"""
scripts/diagnostic/30_mini_e2e_check.py

Mini E2E 7-criteria evaluator. Decides variant (c) keep/drop per §V.D
ablation pre-registration §3.

Pre-conditions (run separately, sequentially):
  1) python -m otp.train.train_otp_soft --config-name otp_soft_minie2e_variantc
     → produces results/minie2e_variantc_<ts1>/
  2) python -m otp.train.train_otp_soft --config-name otp_soft_minie2e_pathB
     → produces results/minie2e_pathB_<ts2>/

Then:
  python scripts/diagnostic/30_mini_e2e_check.py \
      --variantc-run results/minie2e_variantc_<ts1> \
      --pathB-run    results/minie2e_pathB_<ts2>

Exit code:
  0 — ALL 7 criteria PASS → launch full 21-retrain ablation
  1 — ANY criterion FAIL → launch 18-retrain fallback (drop variant c)

Criteria (per preregistration §3.4):
  1. Config propagation (loss magnitude signature)
  2. Loss descent (≥50%, monotonic last 30)
  3. Both runs completed (ckpts + logs exist)
  4. Checkpoint size (<500 MB, trainable-only)
  5. Sample quality relative (variant c ≥20% better on ≥4/7 dims)
  6. Numerical stability (no NaN/Inf)
  7. Determinism (bit-exact 2x forward, trained ckpt)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Pre-registered constants
VARIANTC_LOSS_RANGE = (0.5, 1.5)
PATHB_LOSS_RANGE = (1.0, 3.0)
SAMPLE_QUALITY_IMPROVEMENT_PCT = 20.0
SAMPLE_QUALITY_MIN_DIMS = 4
ACTION_DIMS = 7
LOSS_DESCENT_MIN_PCT = 50.0
LOSS_DESCENT_WINDOW_STEPS = 100
MONOTONIC_LAST_N = 30
MONOTONIC_NOISE_TOLERANCE_PCT = 10.0
CKPT_MAX_MB = 500.0
DETERMINISM_TOLERANCE = 1e-6

logger = logging.getLogger("mini_e2e_check")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)


def load_train_log(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "train_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"train_log.csv not found in {run_dir}")
    return pd.read_csv(csv_path)


def find_latest_ckpt(run_dir: Path) -> Optional[Path]:
    ckpts = sorted(run_dir.glob("ckpt_step*.pt"))
    return ckpts[-1] if ckpts else None


def load_model_from_ckpt(
    ckpt_path: Path,
    config_name: str,
    config_dir: str = "/root/autodl-tmp/OTP-VLA/configs",
    device: str = "cuda",
) -> Tuple[nn.Module, Any]:
    """Hydra-compose config + construct OTPSoftModel + load ckpt."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from otp.models.otp_soft_model import OTPSoftModel

    logger.info(f"Hydra compose: {config_name}")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = OTPSoftModel(model_cfg)
    model.eval()

    logger.info(f"Loading ckpt: {ckpt_path.name}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state = state.get("model_state", state)

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    head_loaded = sum(1 for n in model_state if n.startswith("otp_head."))
    dec_loaded = sum(1 for n in model_state if n.startswith("decoder."))
    logger.info(f"  Loaded: {head_loaded} head + {dec_loaded} decoder params")
    if head_loaded == 0 or dec_loaded == 0:
        raise RuntimeError(
            f"Ckpt has no head ({head_loaded}) or decoder ({dec_loaded}) params"
        )

    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    return model, cfg


def build_eval_dataloader(cfg, batch_size: int = 4, max_batches: int = 12):
    """Build deterministic LIBERO eval loader (training subset, no shuffle)."""
    from otp.data.libero_loader import LIBEROOTPDataset
    from otp.train.utils import collate_fn
    from torch.utils.data import DataLoader, Subset

    dataset = LIBEROOTPDataset(
        root=Path(cfg.data.root),
        suite=cfg.data.suite,
        grasp_affordance_dir=Path(cfg.data.grasp_affordance_dir),
        normalizer_path=(
            Path(cfg.data.normalizer_path)
            if cfg.data.get("normalizer_path") else None
        ),
        num_grasps_per_object=cfg.model.decoder.get("num_grasps_per_object", 8),
        num_points=cfg.model.geometry_encoder.get("num_points", 256),
        horizon=cfg.model.otp_head.get("horizon", 8),
        train_end_demo=cfg.data.get("train_end_demo", 45),
        eval_start_demo=None,
    )
    n = min(len(dataset), max_batches * batch_size)
    subset = Subset(dataset, list(range(n)))
    return DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )


# =============================================================================
# Criteria
# =============================================================================

def check_criterion_1(variantc_df: pd.DataFrame, pathB_df: pd.DataFrame) -> Dict:
    vc = float(variantc_df["total_loss"].head(5).mean())
    pb = float(pathB_df["total_loss"].head(5).mean())
    vc_in = VARIANTC_LOSS_RANGE[0] <= vc <= VARIANTC_LOSS_RANGE[1]
    pb_in = PATHB_LOSS_RANGE[0] <= pb <= PATHB_LOSS_RANGE[1]
    vc_outside_pb = not (PATHB_LOSS_RANGE[0] <= vc <= PATHB_LOSS_RANGE[1])
    return {
        "name": "Criterion 1: Loss magnitude signature",
        "passed": vc_in and pb_in and vc_outside_pb,
        "details": {
            "variantc_early_loss": vc,
            "pathB_early_loss": pb,
            "variantc_in_range": vc_in,
            "pathB_in_range": pb_in,
            "variantc_outside_pathB_range": vc_outside_pb,
        },
    }


def check_criterion_2(variantc_df: pd.DataFrame) -> Dict:
    df = variantc_df[variantc_df["step"] <= LOSS_DESCENT_WINDOW_STEPS]
    if len(df) < 5:
        return {
            "name": "Criterion 2: Loss descent",
            "passed": False,
            "details": {"error": f"Only {len(df)} logged steps in window"},
        }
    s = float(df["total_loss"].head(3).mean())
    e = float(df["total_loss"].tail(3).mean())
    pct = 100.0 * (s - e) / max(abs(s), 1e-9)

    last = variantc_df.tail(max(3, MONOTONIC_LAST_N // 10))
    diffs = np.diff(last["total_loss"].values)
    max_inc_pct = (
        100.0 * float(np.max(diffs)) / max(abs(float(np.mean(last["total_loss"]))), 1e-9)
        if len(diffs) > 0 else 0.0
    )
    return {
        "name": "Criterion 2: Loss descent + monotonic",
        "passed": pct >= LOSS_DESCENT_MIN_PCT
                  and max_inc_pct < MONOTONIC_NOISE_TOLERANCE_PCT,
        "details": {
            "loss_start_3step": s,
            "loss_end_3step": e,
            "descent_pct": pct,
            "monotonic_max_increase_pct": max_inc_pct,
        },
    }


def check_criterion_3(pathB_run: Path, variantc_run: Path) -> Dict:
    pb = find_latest_ckpt(pathB_run)
    vc = find_latest_ckpt(variantc_run)
    return {
        "name": "Criterion 3: Both runs completed",
        "passed": all([
            pb is not None, vc is not None,
            (pathB_run / "train_log.csv").exists(),
            (variantc_run / "train_log.csv").exists(),
        ]),
        "details": {
            "pathB_ckpt": str(pb) if pb else None,
            "variantc_ckpt": str(vc) if vc else None,
        },
    }


def check_criterion_4(variantc_run: Path, pathB_run: Path) -> Dict:
    vc = find_latest_ckpt(variantc_run)
    pb = find_latest_ckpt(pathB_run)
    if not vc or not pb:
        return {
            "name": "Criterion 4: Checkpoint size",
            "passed": False,
            "details": {"error": "Ckpt missing"},
        }
    vc_mb = vc.stat().st_size / 1e6
    pb_mb = pb.stat().st_size / 1e6
    return {
        "name": "Criterion 4: Checkpoint size",
        "passed": vc_mb < CKPT_MAX_MB and pb_mb < CKPT_MAX_MB,
        "details": {
            "variantc_mb": float(vc_mb), "pathB_mb": float(pb_mb),
            "limit_mb": CKPT_MAX_MB,
        },
    }


@torch.no_grad()
def compute_per_dim_l1_std(model: nn.Module, loader, device) -> Tuple[np.ndarray, int]:
    preds_all, gts_all = [], []
    n = 0
    for batch in loader:
        batch_dev = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        try:
            out = model(batch_dev)
        except Exception as e:
            logger.warning(f"Forward failed: {e}")
            continue
        pred = out.get("pred_action")
        gt = batch_dev.get("action")
        if pred is None or gt is None:
            continue
        if pred.shape != gt.shape:
            logger.warning(f"Shape mismatch: pred={pred.shape} gt={gt.shape}")
            continue
        preds_all.append(pred.float().cpu())
        gts_all.append(gt.float().cpu())
        n += pred.shape[0]
    if not preds_all:
        return np.full(ACTION_DIMS, np.inf), 0
    preds = torch.cat(preds_all, dim=0)
    gts = torch.cat(gts_all, dim=0)
    l1 = (preds - gts).abs().mean(dim=(0, 1))
    std = gts.std(dim=(0, 1)) + 1e-9
    return (l1 / std).numpy(), n


def check_criterion_5(
    variantc_run: Path, pathB_run: Path,
    variantc_config: str, pathB_config: str,
) -> Dict:
    vc_ckpt = find_latest_ckpt(variantc_run)
    pb_ckpt = find_latest_ckpt(pathB_run)
    if not vc_ckpt or not pb_ckpt:
        return {
            "name": "Criterion 5: Sample quality relative",
            "passed": False,
            "details": {"error": "Ckpt missing"},
        }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        vc_model, vc_cfg = load_model_from_ckpt(vc_ckpt, variantc_config, device=device.type)
        pb_model, pb_cfg = load_model_from_ckpt(pb_ckpt, pathB_config, device=device.type)
        loader = build_eval_dataloader(vc_cfg, batch_size=4, max_batches=12)
    except Exception as e:
        return {
            "name": "Criterion 5: Sample quality relative",
            "passed": False,
            "details": {"error": f"Model/loader build failed: {e}"},
        }
    vc_ratio, vc_n = compute_per_dim_l1_std(vc_model, loader, device)
    loader2 = build_eval_dataloader(vc_cfg, batch_size=4, max_batches=12)
    pb_ratio, pb_n = compute_per_dim_l1_std(pb_model, loader2, device)
    if vc_n == 0 or pb_n == 0:
        return {
            "name": "Criterion 5: Sample quality relative",
            "passed": False,
            "details": {"error": "No predictions produced", "vc_n": vc_n, "pb_n": pb_n},
        }
    improvement_pct = 100.0 * (pb_ratio - vc_ratio) / np.maximum(pb_ratio, 1e-9)
    dims_improved = int((improvement_pct >= SAMPLE_QUALITY_IMPROVEMENT_PCT).sum())
    return {
        "name": "Criterion 5: Sample quality relative",
        "passed": dims_improved >= SAMPLE_QUALITY_MIN_DIMS,
        "details": {
            "variantc_per_dim_l1_std": vc_ratio.tolist(),
            "pathB_per_dim_l1_std": pb_ratio.tolist(),
            "improvement_pct_per_dim": improvement_pct.tolist(),
            "dims_improved": dims_improved,
            "min_dims_required": SAMPLE_QUALITY_MIN_DIMS,
            "vc_samples": vc_n,
            "pb_samples": pb_n,
        },
    }


def check_criterion_6(variantc_df: pd.DataFrame, pathB_df: pd.DataFrame) -> Dict:
    def check(df: pd.DataFrame) -> Tuple[bool, Dict]:
        cols = ["total_loss", "otp_head_loss", "decoder_loss"]
        problems = {}
        for c in cols:
            if c not in df.columns:
                continue
            vals = pd.to_numeric(df[c], errors='coerce').values
            n_nan = int(np.isnan(vals).sum())
            n_inf = int(np.isinf(vals).sum())
            if n_nan or n_inf:
                problems[c] = {"nan": n_nan, "inf": n_inf}
        return len(problems) == 0, problems
    vc_ok, vc_probs = check(variantc_df)
    pb_ok, pb_probs = check(pathB_df)
    return {
        "name": "Criterion 6: Numerical stability",
        "passed": vc_ok and pb_ok,
        "details": {
            "variantc_clean": vc_ok, "variantc_problems": vc_probs,
            "pathB_clean": pb_ok, "pathB_problems": pb_probs,
        },
    }


def check_criterion_7(variantc_run: Path, variantc_config: str) -> Dict:
    vc_ckpt = find_latest_ckpt(variantc_run)
    if not vc_ckpt:
        return {
            "name": "Criterion 7: Determinism",
            "passed": False,
            "details": {"error": "No ckpt"},
        }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, cfg = load_model_from_ckpt(vc_ckpt, variantc_config, device=device.type)
        model.eval()
        loader = build_eval_dataloader(cfg, batch_size=2, max_batches=1)
        batch = next(iter(loader))
        batch_dev = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        with torch.no_grad():
            out1 = model(batch_dev)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        with torch.no_grad():
            out2 = model(batch_dev)
        p1 = out1.get("pred_action")
        p2 = out2.get("pred_action")
        if p1 is None or p2 is None:
            return {
                "name": "Criterion 7: Determinism",
                "passed": False,
                "details": {"error": "No pred_action"},
            }
        diff = float((p1 - p2).abs().max().item())
        return {
            "name": "Criterion 7: Determinism",
            "passed": diff < DETERMINISM_TOLERANCE,
            "details": {"max_abs_diff": diff, "tolerance": DETERMINISM_TOLERANCE},
        }
    except Exception as e:
        return {
            "name": "Criterion 7: Determinism",
            "passed": False,
            "details": {"error": f"Check failed: {e}"},
        }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variantc-run", type=Path, required=True)
    parser.add_argument("--pathB-run", type=Path, required=True)
    parser.add_argument("--variantc-config", default="otp_soft_minie2e_variantc")
    parser.add_argument("--pathB-config", default="otp_soft_minie2e_pathB")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.variantc_run.exists():
        logger.error(f"variantc_run not found: {args.variantc_run}")
        return 2
    if not args.pathB_run.exists():
        logger.error(f"pathB_run not found: {args.pathB_run}")
        return 2

    output_path = args.output or Path(
        f"results/mini_e2e_report_{int(time.time())}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Mini E2E 7-criteria gate")
    logger.info(f"  variantc: {args.variantc_run}")
    logger.info(f"  pathB:    {args.pathB_run}")

    try:
        variantc_df = load_train_log(args.variantc_run)
        pathB_df = load_train_log(args.pathB_run)
    except Exception as e:
        logger.error(f"Failed to load train logs: {e}")
        return 2

    logger.info(f"  variantc log: {len(variantc_df)} rows")
    logger.info(f"  pathB log:    {len(pathB_df)} rows")

    results = [
        check_criterion_1(variantc_df, pathB_df),
        check_criterion_2(variantc_df),
        check_criterion_3(args.pathB_run, args.variantc_run),
        check_criterion_4(args.variantc_run, args.pathB_run),
        check_criterion_5(args.variantc_run, args.pathB_run,
                          args.variantc_config, args.pathB_config),
        check_criterion_6(variantc_df, pathB_df),
        check_criterion_7(args.variantc_run, args.variantc_config),
    ]

    print("\n" + "=" * 72)
    print("MINI E2E 7-CRITERIA EVALUATION REPORT")
    print("=" * 72)

    pass_count = fail_count = 0
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{status}] {r['name']}")
        for k, v in r["details"].items():
            print(f"  {k}: {v}")
        if r["passed"]:
            pass_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 72)
    print(f"SUMMARY: {pass_count} PASS / {fail_count} FAIL of 7")
    decision = (
        "KEEP variant (c) → launch 21 retrains"
        if fail_count == 0
        else "DROP variant (c) → launch 18 retrains"
    )
    print(f"DECISION: {decision}")
    print("=" * 72 + "\n")

    output_path.write_text(json.dumps({
        "timestamp": time.time(),
        "variantc_run": str(args.variantc_run),
        "pathB_run": str(args.pathB_run),
        "criteria_results": results,
        "summary": {
            "pass_count": pass_count,
            "fail_count": fail_count,
            "decision": decision,
        },
    }, indent=2, default=str))
    logger.info(f"Report: {output_path}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
