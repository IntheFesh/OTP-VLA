"""
otp/train/train_otp_soft.py

Hydra-driven training loop for OTP-Soft (OTPSoftModel).

Config search path: configs/  (relative to repo root).
Default config: otp_soft_frozen.yaml

Usage:
  # Stub sanity (100 steps, synthetic data, CPU-safe):
  python -m otp.train.train_otp_soft --config-name otp_soft_stub

  # Real training (frozen backbone):
  python -m otp.train.train_otp_soft \\
      data.root=data/object_poses \\
      data.grasp_affordance_dir=data/grasp_affordances

§1.2 startup snapshot: [ENV] / [CFG] / [CKPT] / [DATA] printed before first step.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from otp.train.utils import (
    CSVLogger,
    SyntheticDataset,
    assemble_batch,
    build_scheduler,
    collate_fn,
    print_startup,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(message)s",
)
logger = logging.getLogger("train_otp_soft")


def _run(cfg) -> None:
    """Core training logic; separated so it can be called without Hydra."""
    from otp.models.otp_soft_model import OTPSoftModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = cfg.train.precision == "bf16" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # ---- Model ---- #
    try:
        from omegaconf import OmegaConf
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    except Exception:
        model_cfg = dict(cfg.model) if hasattr(cfg.model, "items") else cfg.model

    model = OTPSoftModel(model_cfg).to(device)
    print_startup(cfg, model, device)

    backbone_mode = cfg.model.backbone_mode
    if backbone_mode == "frozen":
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        logger.info("Backbone frozen (mode=%s)", backbone_mode)

    # ---- Dims ---- #
    head_cfg = dict(cfg.model.otp_head) if hasattr(cfg.model, "otp_head") else {}
    num_objects = head_cfg.get("num_objects", 5)
    horizon = head_cfg.get("horizon", 8)
    geom_cfg = dict(cfg.model.geometry_encoder) if hasattr(cfg.model, "geometry_encoder") else {}
    dec_cfg = dict(cfg.model.decoder) if hasattr(cfg.model, "decoder") else {}
    num_grasps = dec_cfg.get("num_grasps_per_object", 8)
    num_points = geom_cfg.get("num_points", 256)

    # ---- Dataset ---- #
    synthetic = getattr(cfg, "synthetic_data", False)
    if synthetic:
        synthetic_steps = getattr(cfg, "synthetic_steps", 500)
        num_samples = synthetic_steps * cfg.train.batch_size
        dataset = SyntheticDataset(
            num_samples=num_samples,
            num_objects=num_objects,
            horizon=horizon,
            num_grasps=num_grasps,
            num_points=num_points,
        )
        loader = DataLoader(
            dataset, batch_size=cfg.train.batch_size,
            shuffle=True, collate_fn=collate_fn, num_workers=0,
        )
    else:
        from otp.data.libero_loader import LIBEROOTPDataset
        norm_path = getattr(cfg.data, "normalizer_path", None)
        dataset = LIBEROOTPDataset(
            root=Path(cfg.data.root),
            suite=cfg.data.suite,
            grasp_affordance_dir=Path(cfg.data.grasp_affordance_dir),
            normalizer_path=Path(norm_path) if norm_path else None,
            num_grasps_per_object=num_grasps,
            num_points=num_points,
            horizon=horizon,
        )
        n_workers = getattr(cfg.data, "num_workers", 4)
        loader = DataLoader(
            dataset, batch_size=cfg.train.batch_size,
            shuffle=True, collate_fn=collate_fn,
            num_workers=n_workers, pin_memory=(device.type == "cuda"),
        )
    logger.info("Dataset: %d samples (%d batches/epoch)", len(dataset), len(loader))

    # ---- Optimizer ---- #
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    num_steps_cfg = getattr(cfg.train, "num_steps", None)
    max_steps = num_steps_cfg or (cfg.train.num_epochs * len(loader))
    scheduler = build_scheduler(optimizer, cfg.train.warmup_steps, max_steps,
                                cfg.train.lr_schedule)

    # ---- W&B ---- #
    use_wandb = False
    wandb_cfg = getattr(cfg, "wandb", None)
    if wandb_cfg and getattr(wandb_cfg, "enabled", False):
        try:
            import wandb
            wandb.init(
                project=wandb_cfg.project,
                entity=getattr(wandb_cfg, "entity", None) or None,
                name=getattr(wandb_cfg, "run_name", None) or None,
                config=model_cfg,
            )
            use_wandb = True
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging")

    # ---- Output ---- #
    out_dir = Path(getattr(cfg, "output_dir", "results/otp_soft"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_log = CSVLogger(out_dir / "train_log.csv")

    # ---- Training loop ---- #
    step = 0
    t0 = time.time()
    loss_window: list[float] = []

    model.train()
    epoch = 0
    while True:
        epoch += 1
        for raw_batch in loader:
            if max_steps is not None and step >= max_steps:
                break

            batch = assemble_batch(raw_batch, device, amp_dtype, num_objects)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
                out = model(batch)

            loss = out["total_loss"]
            if loss is None:
                logger.warning("step %d: NaN guard fired, skipping", step)
                step += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            loss_val = float(loss.detach())
            loss_window.append(loss_val)

            if step % cfg.train.log_every == 0:
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                head_val = float(out["otp_head_loss"].detach()) \
                    if out["otp_head_loss"] is not None else float("nan")
                dec_val = float(out["decoder_loss"].detach()) \
                    if out["decoder_loss"] is not None else float("nan")

                row = {
                    "step": step, "epoch": epoch,
                    "total_loss": loss_val, "otp_head_loss": head_val,
                    "decoder_loss": dec_val, "lr": lr_now,
                    "elapsed_s": f"{elapsed:.1f}",
                }
                csv_log.log(row)
                logger.info("step=%d  total=%.4f  head=%.4f  dec=%.4f  lr=%.2e",
                            step, loss_val, head_val, dec_val, lr_now)
                if use_wandb:
                    import wandb
                    wandb.log(row, step=step)

            ckpt_every = getattr(cfg.train, "ckpt_every", 1000)
            if step % ckpt_every == 0:
                ckpt_path = out_dir / f"ckpt_step{step:07d}.pt"
                torch.save({"step": step,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict()},
                           ckpt_path)
                logger.info("Checkpoint saved: %s", ckpt_path)

        if max_steps is not None and step >= max_steps:
            break
        if num_steps_cfg is None and epoch >= cfg.train.num_epochs:
            break

    elapsed_total = time.time() - t0
    first = loss_window[0] if loss_window else float("nan")
    last = loss_window[-1] if loss_window else float("nan")
    drop = (1.0 - last / first) * 100 if first > 0 else 0.0
    print()
    print(f"Training complete in {elapsed_total:.1f}s  ({step} steps)")
    print(f"  first loss : {first:.4f}")
    print(f"  last  loss : {last:.4f}")
    print(f"  drop       : {drop:.1f}%")
    csv_log.close()

    if use_wandb:
        import wandb
        wandb.finish()


def main() -> None:
    try:
        import hydra
        from omegaconf import DictConfig

        @hydra.main(
            config_path="../../configs",
            config_name="otp_soft_frozen",
            version_base="1.2",
        )
        def _hydra_entry(cfg: DictConfig) -> None:
            _run(cfg)

        _hydra_entry()
    except ImportError:
        logger.error(
            "hydra-core is not installed. "
            "Install with: pip install hydra-core\n"
            "Or import _run() directly for programmatic use."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
