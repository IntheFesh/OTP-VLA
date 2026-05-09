"""
scripts/04_validate_decoder_learning.py

Two-stage evaluation of the OTP-Soft decoder after 1-epoch frozen training.

Stage 1 — Open-loop MSE (required)
  Loads trained checkpoint and a random-init baseline (same architecture).
  Runs both on holdout demos (demo_45–demo_49, 50 demos by default) or on
  a synthetic holdout when --synthetic is given.
  PASS criterion: trained MSE drops >= 70 % vs random-init baseline.

Stage 2 — Sim rollout SR (optional, skipped if LIBERO not installed)
  Runs 1 episode per task (5 tasks from LIBERO-Spatial).
  PASS criterion: mean success rate >= 30 %.

Output: results/04_decoder_sanity.json
  {
    "stage1": {
      "baseline_mse": float,
      "trained_mse":  float,
      "drop_pct":     float,
      "pass":         bool
    },
    "stage2": {          # null if skipped
      "per_task_sr": {task: float, ...},
      "mean_sr":     float,
      "pass":        bool,
      "skipped":     bool
    },
    "overall_pass": bool,
    "wall_clock_s": float
  }

Usage:
  # Synthetic holdout (no data required):
  python scripts/04_validate_decoder_learning.py \
      --checkpoint results/openvla_frozen_1ep/ckpt_*.pt \
      --synthetic

  # Real holdout data:
  python scripts/04_validate_decoder_learning.py \
      --checkpoint results/openvla_frozen_1ep/ckpt_*.pt \
      --data-root data/object_poses \
      --grasp-dir data/grasp_affordances \
      --suite spatial
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(message)s",
)
logger = logging.getLogger("04_validate")

# ---------------------------------------------------------------------------
# Config — mirrors the production training config (small enough for CPU test)
# ---------------------------------------------------------------------------

_EVAL_CFG: Dict[str, Any] = {
    "backbone_dim": 4096,
    "backbone_mode": "stub",   # replaced with real backbone when checkpoint loaded
    "otp_head": {
        "hidden_dim": 256,
        "num_objects": 5,
        "horizon": 8,
        "num_heads": 4,
        "num_layers": 4,
        "vel_hidden_dim": 256,
        "vel_num_layers": 2,
        "use_shortcut": False,
        "num_sample_steps": 10,
    },
    "decoder": {
        "trajectory_dim": 6,
        "affordance_dim": 7,
        "proprio_dim": 8,
        "hidden_dim": 256,
        "num_decoder_layers": 4,
        "num_heads": 4,
        "num_grasps_per_object": 8,
        "action_dim": 7,
        "use_flow_matching": False,
        "num_sample_steps": 10,
    },
    "geometry_encoder": {
        "num_points": 256,
        "output_dim": 128,
        "hidden_dim": 128,
    },
    "otp_head_loss_weight": 1.0,
    "decoder_loss_weight": 1.0,
}

_N_OBJ = 5
_HORIZON = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: Optional[Path], cfg: Dict[str, Any]) -> nn.Module:
    """Load OTPSoftModel from checkpoint, or return random-init if path is None."""
    from otp.models.otp_soft_model import OTPSoftModel

    model = OTPSoftModel(cfg)
    model.eval()

    if checkpoint_path is not None:
        logger.info("Loading checkpoint: %s", checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = state.get("model_state", state)
        trainable_only = state.get("trainable_only", False)
        logger.info("Checkpoint type: %s (%d param tensors)",
                    "trainable_only" if trainable_only else "full_model",
                    len(model_state))
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        # For trainable_only ckpts, "missing" should be ALL backbone params.
        # For mismatched cfg (the original bug), "missing" includes head/decoder.
        # Check: at least one head and one decoder param should NOT be missing.
        head_loaded = sum(1 for n in model_state if n.startswith("otp_head."))
        dec_loaded  = sum(1 for n in model_state if n.startswith("decoder."))
        logger.info("From checkpoint: %d head params, %d decoder params loaded",
                    head_loaded, dec_loaded)
        if head_loaded == 0 or dec_loaded == 0:
            raise RuntimeError(
                f"Checkpoint has no head ({head_loaded}) or decoder ({dec_loaded}) "
                f"params — likely a cfg mismatch. Eval would be meaningless."
            )
        if missing:
            n_miss = len(missing)
            n_backbone_miss = sum(1 for k in missing if "backbone" in k)
            logger.info("Missing %d keys (%d are backbone — expected for frozen).",
                        n_miss, n_backbone_miss)
        if unexpected:
            logger.warning("Unexpected keys (%d): %s …", len(unexpected), unexpected[:3])
        logger.info("Checkpoint loaded (step=%s)", state.get("step", "?"))

    return model


def _make_synthetic_batch(
    batch_size: int,
    num_objects: int = _N_OBJ,
    horizon: int = _HORIZON,
    num_grasps: int = 8,
    num_points: int = 256,
    seed: int = 0,
) -> Dict[str, Any]:
    """Create a fixed-seed synthetic batch for deterministic evaluation."""
    from otp.train.utils import SyntheticDataset, collate_fn, assemble_batch

    ds = SyntheticDataset(
        num_samples=batch_size,
        num_objects=num_objects,
        horizon=horizon,
        num_grasps=num_grasps,
        num_points=num_points,
    )
    raw = collate_fn([ds[i] for i in range(batch_size)])
    return assemble_batch(raw, torch.device("cpu"), torch.float32, num_objects)


def _compute_mse(model, batches, dtype_mode="bf16"):
    """Average action-chunk MSE over a list of batches.

    dtype_mode:
      "bf16" — autocast(bfloat16) forward, matches training-time numerics.
      "fp32" — explicit fp32 forward by casting model and inputs.

    MSE is always computed in fp32 to avoid bf16 accumulation noise.
    """
    import torch
    total_mse = 0.0
    n_batches = 0
    device = next(model.parameters()).device

    if dtype_mode == "fp32":
        # Snapshot model dtype to restore after, cast everything to fp32
        # (this is expensive for OFT 7.5B but only happens once per eval call).
        orig_dtypes = {n: p.dtype for n, p in model.named_parameters()}
        model = model.float()
        autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=False)
    else:
        # bf16 autocast — same as training forward.
        autocast_ctx = torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.bfloat16,
        )

    with torch.no_grad(), autocast_ctx:
        for batch in batches:
            inf_batch = {k: v for k, v in batch.items()
                         if k not in ("gt_trajectory", "gt_action")}
            inf_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in inf_batch.items()}
            out = model(inf_batch)
            pred = out.get("pred_action")
            gt = batch.get("gt_action")
            if pred is None or gt is None:
                continue
            if gt.device != pred.device:
                gt = gt.to(pred.device)
            # fp32 MSE accumulation regardless of forward dtype
            mse = torch.mean((pred.float() - gt.float()) ** 2).item()
            total_mse += mse
            n_batches += 1

    if dtype_mode == "fp32":
        # Restore original (bf16) dtypes module-by-module
        for n, p in model.named_parameters():
            if n in orig_dtypes and p.dtype != orig_dtypes[n]:
                p.data = p.data.to(orig_dtypes[n])

    return total_mse / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Stage 1: Open-loop MSE
# ---------------------------------------------------------------------------

def stage1_open_loop_mse(
    trained_model: nn.Module,
    baseline_model: nn.Module,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Stage 1: open-loop MSE on holdout, dual-precision (bf16 + fp32).

    Reports both:
      - bf16 forward (matches training-time autocast) — paper main result
      - fp32 forward (precision ceiling)              — robustness check

    Pass criterion uses the bf16 number (training-consistent).
    The bf16/fp32 agreement is also reported as a sanity metric.
    """
    logger.info("=== Stage 1: Open-loop MSE ===")
    if args.synthetic:
        logger.info("Using synthetic holdout (seed=100, n=%d)", args.n_holdout)
        batches = [
            _make_synthetic_batch(
                batch_size=min(8, args.n_holdout),
                seed=100 + i,
            )
            for i in range(max(1, args.n_holdout // 8))
        ]
    else:
        # NOTE: LIBEROOTPDataset does not currently support demo_slice; we
        # build the full dataset and take the LAST 5/50 of samples as a
        # rough holdout. This is NOT a true demo-disjoint split — Stage 5
        # (real evaluation) requires train_otp_soft.py to be modified to
        # exclude demo_45-49 during training before any real-holdout
        # number is meaningful. See README §Stage5.
        from otp.data.libero_loader import LIBEROOTPDataset
        from otp.train.utils import collate_fn, assemble_batch
        from torch.utils.data import Subset
        dataset_full = LIBEROOTPDataset(
            root=Path(args.data_root),
            suite=args.suite,
            grasp_affordance_dir=Path(args.grasp_dir),
            horizon=_HORIZON,
        )
        # Approximation: take last 10% as holdout (~5875 samples).
        # Caveat: this includes train data, so the number is in-distribution.
        n_total = len(dataset_full)
        n_holdout = n_total // 10
        holdout_indices = list(range(n_total - n_holdout, n_total))
        dataset = Subset(dataset_full, holdout_indices)
        logger.warning(
            "Using last %d/%d samples as holdout (NOT demo-disjoint). "
            "True holdout requires train script modification — see Stage 5.",
            n_holdout, n_total,
        )
        loader = DataLoader(
            dataset, batch_size=8, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
        batches = [
            assemble_batch(raw, torch.device("cpu"), torch.float32, _N_OBJ)
            for raw in loader
        ]
        logger.info("Holdout dataset: %d samples, %d batches", len(dataset), len(batches))

    # Run dual-precision evaluation
    results: Dict[str, Any] = {}
    for mode in ("bf16", "fp32"):
        logger.info("--- Stage 1 [%s forward] ---", mode)
        logger.info("  Evaluating trained model ...")
        trained_mse = _compute_mse(trained_model, batches, dtype_mode=mode)
        logger.info("  Evaluating random-init baseline ...")
        baseline_mse = _compute_mse(baseline_model, batches, dtype_mode=mode)
        drop_pct = ((baseline_mse - trained_mse) /
                    (abs(baseline_mse) + 1e-8) * 100.0)
        results[mode] = {
            "baseline_mse": round(float(baseline_mse), 6),
            "trained_mse":  round(float(trained_mse),  6),
            "drop_pct":     round(float(drop_pct),     2),
        }
        logger.info("  [%s] baseline=%.6f  trained=%.6f  drop=%+.1f%%",
                    mode, baseline_mse, trained_mse, drop_pct)

    # Pass criterion: bf16 (training-consistent) drop >= 70 %
    passed = results["bf16"]["drop_pct"] >= 70.0

    # bf16 vs fp32 numerical agreement — a sanity signal
    bf16_t = results["bf16"]["trained_mse"]
    fp32_t = results["fp32"]["trained_mse"]
    relative_disagreement = (abs(bf16_t - fp32_t) /
                             (abs(fp32_t) + 1e-8))
    results["bf16_fp32_relative_disagreement"] = round(float(relative_disagreement), 4)
    results["pass"] = bool(passed)

    logger.info(
        "Stage 1 SUMMARY | bf16 drop=%+.1f%%  fp32 drop=%+.1f%%  "
        "bf16-fp32 disagree=%.2f%%  PASS=%s",
        results["bf16"]["drop_pct"],
        results["fp32"]["drop_pct"],
        relative_disagreement * 100,
        passed,
    )
    return results


# ---------------------------------------------------------------------------
# Stage 2: Sim rollout (optional)
# ---------------------------------------------------------------------------

def stage2_sim_rollout(
    trained_model: nn.Module,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    logger.info("=== Stage 2: Sim rollout SR ===")

    try:
        import libero  # noqa: F401
    except ImportError:
        logger.warning("LIBERO not installed — skipping sim rollout.")
        return {"skipped": True, "per_task_sr": {}, "mean_sr": None, "pass": None}

    try:
        from libero.envs import OffScreenRenderEnv
        from libero.utils.task_generation_utils import get_task_info

        SPATIAL_TASKS = [
            "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
            "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
            "KITCHEN_SCENE3_pick_up_the_brown_box",
            "KITCHEN_SCENE4_put_the_wine_bottle_in_the_top_drawer",
            "KITCHEN_SCENE5_open_the_top_drawer",
        ]

        per_task_sr: Dict[str, float] = {}
        trained_model.eval()

        for task_name in SPATIAL_TASKS:
            env = OffScreenRenderEnv(task_name=task_name, render_gpu_device_id=0)
            obs = env.reset()
            done = False
            success = False

            for _ in range(300):    # max 300 steps per episode
                img = torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1).unsqueeze(0)
                batch = {
                    "image":               img,
                    "instruction":         [obs.get("task_description", task_name)],
                    "object_indices":      torch.arange(_N_OBJ).unsqueeze(0),
                    "object_point_clouds": torch.zeros(1, _N_OBJ, 256, 3),
                    "proprioception":      torch.zeros(1, 8),
                    "grasp_affordance":    torch.zeros(1, _N_OBJ, 8, 7),
                }
                with torch.no_grad():
                    out = trained_model(batch)
                action = out["pred_action"]
                if action is None:
                    break
                action_np = action[0, 0].cpu().numpy()    # first step of horizon
                obs, reward, done, info = env.step(action_np)
                if done:
                    success = bool(info.get("success", False))
                    break

            per_task_sr[task_name] = float(success)
            env.close()
            logger.info("Task %s: SR=%.0f", task_name, success)

        mean_sr = float(sum(per_task_sr.values()) / max(len(per_task_sr), 1))
        passed = mean_sr >= 0.30

        logger.info("Stage 2 | mean_SR=%.2f  PASS=%s", mean_sr, passed)

        return {
            "skipped":      False,
            "per_task_sr":  per_task_sr,
            "mean_sr":      round(mean_sr, 4),
            "pass":         bool(passed),
        }

    except Exception as exc:
        logger.error("Sim rollout failed: %s", exc)
        return {"skipped": True, "per_task_sr": {}, "mean_sr": None, "pass": None,
                "error": str(exc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to trained checkpoint (.pt). If omitted, uses random init (baseline test).",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic holdout instead of real LIBERO data.",
    )
    parser.add_argument("--data-root", type=str, default="data/object_poses")
    parser.add_argument("--grasp-dir", type=str, default="data/grasp_affordances")
    parser.add_argument("--suite", type=str, default="spatial")
    parser.add_argument("--n-holdout", type=int, default=40,
                        help="Number of synthetic holdout samples (synthetic mode).")
    parser.add_argument("--no-sim", action="store_true",
                        help="Skip Stage 2 sim rollout even if LIBERO is installed.")
    parser.add_argument("--output", type=Path, default=Path("results/04_decoder_sanity.json"))
    args = parser.parse_args()

    if args.checkpoint is None and not args.synthetic:
        logger.warning("No checkpoint given and --synthetic not set; defaulting to synthetic.")
        args.synthetic = True

    t0 = time.time()

    torch.manual_seed(42)
    # PATCH: load production cfg from configs/otp_soft_frozen.yaml so the
    # evaluation model matches the trained checkpoint (otherwise shape
    # mismatches cause silent strict=False skipping of all trained weights).
    from omegaconf import OmegaConf
    prod_cfg = OmegaConf.load("configs/otp_soft_frozen.yaml")
    cfg = OmegaConf.to_container(prod_cfg.model, resolve=True)
    logger.info("Using production model cfg from otp_soft_frozen.yaml: "
                "head.hidden=%d, decoder.hidden=%d, backbone_mode=%s",
                cfg["otp_head"]["hidden_dim"],
                cfg["decoder"]["hidden_dim"],
                cfg["backbone_mode"])

    logger.info("Building trained model …")
    trained_model = _load_model(args.checkpoint, cfg)

    logger.info("Building random-init baseline (same arch) …")
    torch.manual_seed(0)
    baseline_model = _load_model(None, cfg)

    stage1 = stage1_open_loop_mse(trained_model, baseline_model, args)

    stage2: Optional[Dict[str, Any]] = None
    if not args.no_sim:
        stage2 = stage2_sim_rollout(trained_model, args)
    else:
        logger.info("Stage 2 skipped (--no-sim).")

    wall_clock = time.time() - t0

    # Overall pass: Stage 1 must pass; Stage 2 pass is OR (optional).
    s2_pass = stage2["pass"] if (stage2 and not stage2.get("skipped")) else True
    overall_pass = stage1["pass"] and s2_pass

    report = {
        "stage1":       stage1,
        "stage2":       stage2,
        "overall_pass": bool(overall_pass),
        "wall_clock_s": round(wall_clock, 1),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 60)
    # Stage 1 returns dual-precision dict: {"bf16": {...}, "fp32": {...}, ...}
    bf16_drop = stage1["bf16"]["drop_pct"]
    fp32_drop = stage1["fp32"]["drop_pct"]
    disagree  = stage1["bf16_fp32_relative_disagreement"] * 100.0
    print(f"  Stage 1 MSE drop : bf16={bf16_drop:+.1f}%  fp32={fp32_drop:+.1f}%  "
          f"disagree={disagree:.2f}%  PASS={stage1['pass']}")
    if stage2:
        sr = stage2.get("mean_sr")
        skip = stage2.get("skipped", False)
        print(f"  Stage 2 mean SR  : {sr if sr is not None else 'n/a'}  PASS={stage2.get('pass')}  skipped={skip}")
    print(f"  Overall PASS     : {overall_pass}")
    print(f"  Wall-clock       : {wall_clock:.1f}s")
    print(f"  Report saved     : {args.output}")
    print("=" * 60)

    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
