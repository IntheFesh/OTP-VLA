"""
CFM action distribution diagnostic — paper-ready evidence for §V.B.

Decomposes "is OTP-Soft single-sample inference viable in sim?" into two
quantitative measurements per action dimension:

  Path 1: Marginal distribution shape
    - 100 conditions × 10 samples = 1000 samples per dim per split
    - Plotted as overlay histogram with GT marginal as gray reference
    - Answers: does CFM cover the GT action range overall?

  Path 2: Conditional variance + GT coverage
    - 30 conditions × 100 samples per condition
    - Per-condition per-dim std → normalized by train-set Q99-Q1 GT range
    - Coverage: fraction of (timestep, dim) cells where GT ∈ [CFM_min, CFM_max]
    - Stat power: n=30 conditions detects ≥27% mean-std difference between
      train and holdout (covers expected 20-30% generalization gap)
    - Answers: at a fixed condition, how spread are CFM samples? Does GT
      fall inside the CFM spread?

Reference range (paper convention):
    gt_range_per_dim = q99(train_actions) - q01(train_actions)
    Computed ONCE from train split actions.npz files (raw, no dataset
    abstraction). Both train and holdout use this same train-derived range
    as the reference scale, so train vs holdout comparison reflects only
    numerator change (conditional std), not normalization shift.

RNG (diagnostic-side, namespace-isolated from predictor episode_seed):
    DIAGNOSTIC_BASE_SEED = args.seed (default 42)
    sample_seed = DIAGNOSTIC_BASE_SEED * 10**7 + cond_idx * 10**3 + sample_idx
    Each sample is reproducible bit-exact within bf16/CUDA tolerance.

Diagnostic dataloader:
    DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    - shuffle=False: condition_idx i ↔ deterministic (demo, frame) tuple
    - num_workers=0: avoid worker_init_fn RNG complexity
    - Conditions selected via stride sampling across dataset to span demos
      rather than all-from-first-demo

Output (in --output_dir):
    marginal_overlay.png         (1 fig, 7 subplots: per-dim, train+holdout+GT)
    conditional_summary.png      (7 subplots, per-dim conditional spread paired
                                  train/holdout with GT dots — paper-ready)
    conditional_variance.json    (full numerical data per spec v3 schema)
    conditional_coverage.json    (per-condition coverage rates)
    per_condition_dump.npz       (raw samples for offline re-analysis)
    README.md                    (headline numbers + paper-ready summary)

Usage:
    python scripts/diagnostic/05c_action_distribution_histogram.py \
        --ckpt $(cat .paper_ready_ckpt) \
        --config configs/otp_soft_4e.yaml \
        --output-dir results/cfm_diagnostic_$(date +%Y%m%d_%H%M%S) \
        --seed 42

Estimated runtime:
    Path 1: 100 × 10 × 2 split = 2000 forward
    Path 2:  30 ×100 × 2 split = 6000 forward
    Total: ~8000 forward × ~200ms = ~27 min compute
    + GT range + plotting: ~10 min
    Wall time: ~40 min
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Set up logging early
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(message)s",
)
logger = logging.getLogger("cfm_histogram")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIM_LABELS = [
    "Δpos_x", "Δpos_y", "Δpos_z",   # Scaled OSC position delta
    "Δrot_x", "Δrot_y", "Δrot_z",   # Scaled OSC axis-angle orientation delta
    "Gripper",                       # Binary gripper command
]
ACTION_DIM = 7
HORIZON = 8

N_CONDITIONS_MARGINAL = 100
N_SAMPLES_PER_CONDITION_MARGINAL = 10
N_CONDITIONS_CONDITIONAL = 30
N_SAMPLES_PER_CONDITION_CONDITIONAL = 100


# ---------------------------------------------------------------------------
# RNG isolation
# ---------------------------------------------------------------------------
def make_sample_seed(base_seed: int, cond_idx: int, sample_idx: int) -> int:
    """Deterministic per-sample seed scheme.

    Layout: base_seed * 10^7 + cond_idx * 10^3 + sample_idx
    Note: namespace isolated from predictor episode_seed namespace.
    """
    return base_seed * 10_000_000 + cond_idx * 1_000 + sample_idx


def set_sample_rng(seed: int) -> None:
    """Set all RNG sources for one diagnostic sample."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# GT range computation (paper-ready normalization reference)
# ---------------------------------------------------------------------------
def compute_train_gt_range(
    npz_root: Path,
    train_end_demo: int = 45,
) -> np.ndarray:
    """Compute Q99-Q1 range per action dim from TRAIN split.

    Reads raw `actions` arrays from per-demo npz files, applies the same
    per-task numeric sort + slice as LIBEROOTPDataset uses, computes
    quantile range over all flattened (T, 7) actions.

    Returns:
        gt_range: (7,) float32 — Q99(actions[:, d]) - Q01(actions[:, d])
    """
    npz_paths = sorted(Path(npz_root).glob("*/demo_*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No demo_*.npz under {npz_root}")

    by_task: Dict[str, List[Path]] = defaultdict(list)
    for p in npz_paths:
        by_task[p.parent.name].append(p)

    def demo_num(p: Path) -> int:
        m = re.match(r"demo_(\d+)$", p.stem)
        if m is None:
            raise ValueError(f"Unexpected demo filename: {p}")
        return int(m.group(1))

    train_paths = []
    for task in sorted(by_task.keys()):
        sorted_paths = sorted(by_task[task], key=demo_num)
        train_paths.extend(sorted_paths[:train_end_demo])

    logger.info("Computing GT range from %d train demos across %d tasks",
                len(train_paths), len(by_task))

    all_actions = []
    for npz_path in train_paths:
        data = np.load(npz_path, allow_pickle=False)
        all_actions.append(np.asarray(data["actions"], dtype=np.float32))

    flat = np.concatenate(all_actions, axis=0)  # (Σ T_demo, 7)
    logger.info("GT actions: total %d timesteps", flat.shape[0])

    q99 = np.quantile(flat, 0.99, axis=0)
    q01 = np.quantile(flat, 0.01, axis=0)
    gt_range = (q99 - q01).astype(np.float32)

    logger.info("GT range per dim: %s", [f"{x:.4f}" for x in gt_range])
    return gt_range


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_from_ckpt(
    ckpt_path: Path,
    config_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict]:
    """Load OTPSoftModel and restore trainable params from ckpt.

    Frozen backbone weights are loaded from HF cache automatically when
    OTPSoftModel is constructed with backbone_mode='frozen'; this function
    overlays the trainable_only ckpt on top.

    Returns:
        model: in eval mode, on device
        cfg: OmegaConf config (for downstream reference: num_objects, etc.)
    """
    from omegaconf import OmegaConf
    from otp.models.otp_soft_model import OTPSoftModel

    # Load config — Hydra config has 'defaults' that we resolve here
    cfg = OmegaConf.load(config_path)
    if "defaults" in cfg:
        # Resolve 'defaults: - otp_soft_frozen - _self_' inheritance manually
        # since we're not going through hydra.main.
        # Hydra semantics: each str in defaults is a yaml filename to merge
        # as base, except '_self_' which is the special token meaning "this file".
        # Order matters: merge base first, then overlay self.
        defaults = cfg.pop("defaults")
        bases = []
        self_position = None
        for i, d in enumerate(defaults):
            if isinstance(d, str):
                if d == "_self_":
                    self_position = i
                else:
                    bases.append(d)
            elif isinstance(d, dict):
                # dict-style defaults (e.g. {hydra: default}) — skip
                pass

        # Merge: bases in order, then _self_ (overlay current cfg) at its position.
        # Standard pattern: bases come first, _self_ last → bases overlaid by current cfg.
        merged = OmegaConf.create({})
        for base_name in bases:
            base_path = config_path.parent / f"{base_name}.yaml"
            if not base_path.exists():
                raise FileNotFoundError(
                    f"Defaults base '{base_name}' not found at {base_path}"
                )
            base_cfg = OmegaConf.load(base_path)
            # Recursively resolve base's own defaults
            if "defaults" in base_cfg:
                base_defaults = base_cfg.pop("defaults")
                for bd in base_defaults:
                    if isinstance(bd, str) and bd != "_self_":
                        bd_path = config_path.parent / f"{bd}.yaml"
                        if bd_path.exists():
                            merged = OmegaConf.merge(merged, OmegaConf.load(bd_path))
            merged = OmegaConf.merge(merged, base_cfg)
            logger.info("Resolved defaults: merged base %s", base_name)
        # Overlay current cfg (the _self_ part) on top
        cfg = OmegaConf.merge(merged, cfg)

    # Build model (this loads the frozen backbone from HF cache)
    logger.info("Building OTPSoftModel...")
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = OTPSoftModel(model_cfg)

    # Load trainable params from ckpt
    logger.info("Loading ckpt: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt.get("trainable_only", False), \
        "Expected trainable_only=True ckpt"
    model_state = ckpt["model_state"]

    # Use strict=False because:
    #   1. frozen backbone weights are NOT in this trainable_only ckpt
    #      (loaded from HF cache when OTPSoftModel was constructed)
    #   2. ShortcutFlowMatching internally references velocity_net as
    #      self.model, causing PyTorch to register the same tensor under
    #      two paths in state_dict (otp_head.velocity_net.* and
    #      otp_head.flow_matcher.model.*). Trainable_only ckpt saves only
    #      the canonical path; the duplicate reference is auto-populated
    #      when the canonical path loads (verified via tensor identity).
    missing, unexpected = model.load_state_dict(model_state, strict=False)

    backbone_missing = [k for k in missing if k.startswith("backbone.")]
    flow_matcher_dup = [k for k in missing if ".flow_matcher.model." in k]
    real_missing = [
        k for k in missing
        if not k.startswith("backbone.") and ".flow_matcher.model." not in k
    ]
    if real_missing:
        raise RuntimeError(
            f"Real missing keys (not backbone, not flow_matcher dup ref): "
            f"{real_missing[:5]}..."
        )
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys in ckpt: {unexpected[:5]}..."
        )
    logger.info(
        "Loaded %d trainable params "
        "(skipped %d frozen backbone, %d flow_matcher dup refs)",
        len(model_state), len(backbone_missing), len(flow_matcher_dup),
    )

    model = model.to(device)
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Dataset / dataloader (diagnostic configuration)
# ---------------------------------------------------------------------------
def build_diagnostic_loader(
    cfg,
    split: str,
    train_end_demo: int = 45,
) -> DataLoader:
    """Build a deterministic, single-process DataLoader for diagnostic.

    split: "train" → demos 0:45 per task
           "holdout" → demos 45: per task
    """
    from otp.data.libero_loader import LIBEROOTPDataset
    from otp.train.utils import collate_fn

    norm_path = getattr(cfg.data, "normalizer_path", None)
    head_cfg = dict(cfg.model.otp_head) if hasattr(cfg.model, "otp_head") else {}
    geom_cfg = dict(cfg.model.geometry_encoder) if hasattr(cfg.model, "geometry_encoder") else {}
    dec_cfg = dict(cfg.model.decoder) if hasattr(cfg.model, "decoder") else {}

    kwargs = dict(
        root=Path(cfg.data.root),
        suite=cfg.data.suite,
        grasp_affordance_dir=Path(cfg.data.grasp_affordance_dir),
        normalizer_path=Path(norm_path) if norm_path else None,
        num_grasps_per_object=dec_cfg.get("num_grasps_per_object", 8),
        num_points=geom_cfg.get("num_points", 256),
        horizon=head_cfg.get("horizon", 8),
    )
    if split == "train":
        kwargs["train_end_demo"] = train_end_demo
    elif split == "holdout":
        kwargs["eval_start_demo"] = train_end_demo
    else:
        raise ValueError(f"split must be 'train' or 'holdout', got {split!r}")

    ds = LIBEROOTPDataset(**kwargs)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn, pin_memory=False,
    )
    return loader


def select_condition_indices(n_total: int, n_select: int) -> np.ndarray:
    """Stride sampling across dataset to span multiple demos.

    Returns n_select indices uniformly distributed in [0, n_total)."""
    if n_select >= n_total:
        return np.arange(n_total)
    return np.linspace(0, n_total - 1, n_select).astype(np.int64)


# ---------------------------------------------------------------------------
# Forward + sample collection
# ---------------------------------------------------------------------------
def assemble_inference_batch(raw, device, amp_dtype, num_objects):
    """Same logic as train_otp_soft.assemble_batch but DROPS gt_* fields
    (we want inference-mode forward, not loss computation)."""
    from otp.utils.lie_algebra import so3_to_quat
    B = raw["image"].shape[0]
    image = raw["image"]
    instruction = raw["instruction"]
    object_indices = torch.arange(num_objects).unsqueeze(0).expand(B, -1)
    ee = raw["ee_pose"].float()
    pos = ee[:, :3, 3]
    quat = so3_to_quat(ee[:, :3, :3])
    gripper = torch.zeros(B, 1, dtype=torch.float32)
    proprio = torch.cat([pos, quat, gripper], dim=-1)
    return {
        "image":               image.to(device),
        "instruction":         instruction,
        "object_indices":      object_indices.to(device),
        "object_point_clouds": raw["object_point_clouds"].to(device, amp_dtype),
        "proprioception":      proprio.to(device, amp_dtype),
        "grasp_affordance":    raw["grasp_affordance"].to(device, amp_dtype),
        # NOTE: gt_trajectory and gt_action are intentionally NOT included
        # so model.forward runs in inference mode (returns pred_action only).
    }


def collect_samples_for_condition(
    model: torch.nn.Module,
    batch: Dict,
    n_samples: int,
    base_seed: int,
    cond_idx: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    num_objects: int,
) -> np.ndarray:
    """Run model n_samples times on the same condition.

    Each sample uses make_sample_seed(base_seed, cond_idx, sample_idx).

    Returns:
        samples: (n_samples, H, 7) float32
    """
    samples = np.zeros((n_samples, HORIZON, ACTION_DIM), dtype=np.float32)
    inference_batch = assemble_inference_batch(
        batch, device, amp_dtype, num_objects
    )
    for s_idx in range(n_samples):
        seed = make_sample_seed(base_seed, cond_idx, s_idx)
        set_sample_rng(seed)
        # autocast required: backbone outputs bf16 but trainable params are fp32;
        # train script uses the same wrapper at line 205. use_bf16 mirrored to
        # device.type=='cuda' (CPU mode falls back to fp32 via amp_dtype).
        use_amp = (device.type == "cuda" and amp_dtype == torch.bfloat16)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(inference_batch)
        pred = out["pred_action"]
        if pred is None:
            raise RuntimeError(
                f"pred_action is None at cond_idx={cond_idx}, sample={s_idx}. "
                f"OTP head may have returned NaN guard."
            )
        samples[s_idx] = pred[0].float().cpu().numpy()
    return samples


# ---------------------------------------------------------------------------
# Coverage metric
# ---------------------------------------------------------------------------
def compute_coverage(samples: np.ndarray, gt_action: np.ndarray) -> float:
    """Fraction of (timestep, dim) cells where GT ∈ [CFM_min, CFM_max].

    Args:
        samples:   (n_samples, H, 7) CFM outputs
        gt_action: (H, 7) ground truth at this condition

    Returns:
        coverage in [0, 1]
    """
    cfm_min = samples.min(axis=0)  # (H, 7)
    cfm_max = samples.max(axis=0)  # (H, 7)
    inside = (gt_action >= cfm_min) & (gt_action <= cfm_max)
    return float(inside.mean())


# ---------------------------------------------------------------------------
# Main diagnostic loops
# ---------------------------------------------------------------------------
def run_path_1_marginal(
    model, loader, condition_indices, n_samples_per, base_seed, device,
    amp_dtype, num_objects,
) -> Tuple[np.ndarray, np.ndarray]:
    """Path 1: Marginal histogram data.

    Returns:
        cfm_samples: (N_cond × n_samples, H, 7)
        gt_actions:  (N_cond, H, 7)  — for marginal overlay
    """
    all_cfm = []
    all_gt = []
    loader_iter = iter(loader)
    cur_idx = 0
    for cond_idx, target_idx in enumerate(condition_indices):
        # Advance loader to target_idx
        while cur_idx < target_idx:
            next(loader_iter)
            cur_idx += 1
        batch = next(loader_iter)
        cur_idx += 1

        gt_action = batch["action_chunk"][0].numpy()  # (H, 7)
        all_gt.append(gt_action)

        samples = collect_samples_for_condition(
            model, batch, n_samples_per, base_seed, cond_idx,
            device, amp_dtype, num_objects,
        )
        all_cfm.append(samples)

        if (cond_idx + 1) % 20 == 0:
            logger.info("  Path 1: %d/%d conditions", cond_idx + 1, len(condition_indices))

    return np.concatenate(all_cfm, axis=0), np.stack(all_gt, axis=0)


def run_path_2_conditional(
    model, loader, condition_indices, n_samples_per, base_seed, device,
    amp_dtype, num_objects, gt_range,
) -> Dict:
    """Path 2: Conditional variance + coverage.

    Returns dict with:
        per_condition_std:        (N_cond, H, 7)
        per_condition_mean_std:   (N_cond, 7)  — averaged over H
        per_condition_coverage:   (N_cond,)
        per_condition_samples:    (N_cond, n_samples, H, 7)  (for dump)
        per_condition_gt:         (N_cond, H, 7)
        normalized_cond_std_per_dim:  (7,)  — mean over conditions / gt_range
        conditional_std_per_dim:      (7,)  — mean over conditions, raw
    """
    n_cond = len(condition_indices)
    per_cond_std = np.zeros((n_cond, HORIZON, ACTION_DIM), dtype=np.float32)
    per_cond_mean_std = np.zeros((n_cond, ACTION_DIM), dtype=np.float32)
    per_cond_cov = np.zeros(n_cond, dtype=np.float32)
    per_cond_samples = np.zeros((n_cond, n_samples_per, HORIZON, ACTION_DIM), dtype=np.float32)
    per_cond_gt = np.zeros((n_cond, HORIZON, ACTION_DIM), dtype=np.float32)

    loader_iter = iter(loader)
    cur_idx = 0
    for cond_idx, target_idx in enumerate(condition_indices):
        while cur_idx < target_idx:
            next(loader_iter)
            cur_idx += 1
        batch = next(loader_iter)
        cur_idx += 1

        gt_action = batch["action_chunk"][0].numpy()
        per_cond_gt[cond_idx] = gt_action

        samples = collect_samples_for_condition(
            model, batch, n_samples_per, base_seed, cond_idx,
            device, amp_dtype, num_objects,
        )
        per_cond_samples[cond_idx] = samples

        # Per-(timestep, dim) std across samples
        per_cond_std[cond_idx] = samples.std(axis=0)
        # Per-dim std averaged over H timesteps
        per_cond_mean_std[cond_idx] = per_cond_std[cond_idx].mean(axis=0)
        # Coverage
        per_cond_cov[cond_idx] = compute_coverage(samples, gt_action)

        if (cond_idx + 1) % 5 == 0:
            logger.info("  Path 2: %d/%d conditions, last cond mean std=%s, cov=%.3f",
                        cond_idx + 1, n_cond,
                        [f"{x:.4f}" for x in per_cond_mean_std[cond_idx]],
                        per_cond_cov[cond_idx])

    # Aggregate
    cond_std_per_dim = per_cond_mean_std.mean(axis=0)  # (7,)
    normalized_cond_std = cond_std_per_dim / gt_range

    return {
        "per_condition_std": per_cond_std,
        "per_condition_mean_std": per_cond_mean_std,
        "per_condition_coverage": per_cond_cov,
        "per_condition_samples": per_cond_samples,
        "per_condition_gt": per_cond_gt,
        "conditional_std_per_dim": cond_std_per_dim,
        "normalized_cond_std_per_dim": normalized_cond_std,
    }


# ---------------------------------------------------------------------------
# Plotting (paper-ready)
# ---------------------------------------------------------------------------
def plot_marginal_overlay(
    cfm_train: np.ndarray, cfm_holdout: np.ndarray,
    gt_train: np.ndarray, gt_holdout: np.ndarray,
    output_path: Path,
):
    """7 subplots, per-dim CFM histograms train+holdout overlaid with GT marginal."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=150)
    axes = axes.flatten()

    # All actions flattened to (N, 7)
    cfm_train_flat = cfm_train.reshape(-1, ACTION_DIM)
    cfm_hold_flat = cfm_holdout.reshape(-1, ACTION_DIM)
    gt_train_flat = gt_train.reshape(-1, ACTION_DIM)
    gt_hold_flat = gt_holdout.reshape(-1, ACTION_DIM)

    for d in range(ACTION_DIM):
        ax = axes[d]
        # GT marginal as gray reference (pooled train+holdout GT)
        gt_combined = np.concatenate([gt_train_flat[:, d], gt_hold_flat[:, d]])
        ax.hist(gt_combined, bins=30, alpha=0.4, color="gray", label="GT", density=True)
        ax.hist(cfm_train_flat[:, d], bins=30, alpha=0.5, color="C0",
                label="CFM (train)", density=True, histtype="step", linewidth=1.5)
        ax.hist(cfm_hold_flat[:, d], bins=30, alpha=0.5, color="C1",
                label="CFM (holdout)", density=True, histtype="step", linewidth=1.5)
        ax.set_title(f"Dim {d}: {DIM_LABELS[d]}", fontsize=11)
        ax.set_xlabel("Action value (scaled OSC)")
        if d == 0:
            ax.set_ylabel("Density")
            ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)

    # Hide unused 8th subplot
    axes[7].axis("off")

    fig.suptitle(
        "OTP-Soft CFM marginal action distribution (5-epoch ckpt, step 16000)\n"
        "GT pooled (gray) vs CFM samples by split. Actions in scaled OSC input space.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_conditional_summary(
    samples_train: np.ndarray, gt_train: np.ndarray,
    samples_hold: np.ndarray, gt_hold: np.ndarray,
    gt_range: np.ndarray, output_path: Path,
):
    """7-subplot per-dim conditional summary (paper-ready).

    For each of 7 action dimensions, one subplot:
      x-axis: condition index 0..N_cond-1
      y-axis: action value (raw OSC scaled space)
      For each condition, paired box plot: train (blue) and holdout (orange),
        showing CFM 100-sample spread (over H × N_samples values for this dim).
      GT action at each condition (timestep 0): red dot at condition x position.
      Reference lines: train Q01 / Q99 (gray dashed) — the normalization range.

    Args:
        samples_*: (N_cond, n_samples, H, 7) CFM outputs
        gt_*:      (N_cond, H, 7) ground truth at each condition
        gt_range:  (7,) train Q99-Q1 used as normalization reference
    """
    import matplotlib.pyplot as plt

    n_cond = samples_train.shape[0]
    assert samples_hold.shape[0] == n_cond, "train/holdout n_cond mismatch"

    fig, axes = plt.subplots(2, 4, figsize=(24, 10), dpi=150)
    axes = axes.flatten()

    # Pre-compute Q01/Q99 from train GT marginals (per dim) for reference lines.
    # Note: gt_range = q99 - q01 already passed in; we recompute the absolute
    # bounds here from gt_train for the dashed reference lines.
    gt_train_flat = gt_train.reshape(-1, ACTION_DIM)
    train_q99 = np.quantile(gt_train_flat, 0.99, axis=0)
    train_q01 = np.quantile(gt_train_flat, 0.01, axis=0)

    cond_x = np.arange(n_cond)
    offset = 0.18  # paired box plot offset

    for d in range(ACTION_DIM):
        ax = axes[d]

        # Build per-condition flattened CFM samples for this dim.
        # For each condition, stack (n_samples × H) values together.
        train_per_cond = [samples_train[c, :, :, d].flatten() for c in range(n_cond)]
        hold_per_cond = [samples_hold[c, :, :, d].flatten() for c in range(n_cond)]
        gt_per_cond_train = gt_train[:, 0, d]  # (N_cond,) timestep-0 GT
        gt_per_cond_hold = gt_hold[:, 0, d]

        # Reference lines for train Q01/Q99
        ax.axhline(train_q01[d], color="gray", linestyle="--", linewidth=0.8,
                   alpha=0.6, label=f"train Q01/Q99 (range={gt_range[d]:.3f})")
        ax.axhline(train_q99[d], color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

        # Train box plots (left of each x position)
        bp_t = ax.boxplot(
            train_per_cond, positions=cond_x - offset, widths=0.32,
            showfliers=False, patch_artist=True, manage_ticks=False,
        )
        for patch in bp_t["boxes"]:
            patch.set_facecolor("C0")
            patch.set_alpha(0.55)
        for med in bp_t["medians"]:
            med.set_color("navy")

        # Holdout box plots (right of each x position)
        bp_h = ax.boxplot(
            hold_per_cond, positions=cond_x + offset, widths=0.32,
            showfliers=False, patch_artist=True, manage_ticks=False,
        )
        for patch in bp_h["boxes"]:
            patch.set_facecolor("C1")
            patch.set_alpha(0.55)
        for med in bp_h["medians"]:
            med.set_color("darkorange")

        # GT red dots (paired: at train and holdout positions)
        ax.scatter(cond_x - offset, gt_per_cond_train,
                   color="red", s=18, zorder=10, label="GT (train)")
        ax.scatter(cond_x + offset, gt_per_cond_hold,
                   color="darkred", marker="x", s=22, zorder=10, label="GT (holdout)")

        ax.set_title(f"Dim {d}: {DIM_LABELS[d]}", fontsize=11)
        ax.set_xlabel("Condition index")
        if d % 4 == 0:
            ax.set_ylabel("Action value (scaled OSC)")
        ax.set_xticks(np.arange(0, n_cond, max(1, n_cond // 10)))
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, axis="y")
        if d == 0:
            ax.legend(fontsize=8, loc="best")

    # Hide unused 8th subplot
    axes[7].axis("off")

    fig.suptitle(
        f"OTP-Soft CFM conditional spread per dimension ({n_cond} conditions, "
        f"100 samples each, paired train/holdout)\n"
        "Box: CFM sample spread. Red dots: GT action (timestep 0). "
        "Dashed lines: train Q01/Q99 reference range.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Path to ckpt_step*.pt file")
    parser.add_argument("--config", type=Path,
                        default=Path("configs/otp_soft_4e.yaml"),
                        help="Path to model config yaml")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write outputs")
    parser.add_argument("--seed", type=int, default=42,
                        help="DIAGNOSTIC_BASE_SEED for sample seeding (default 42)")
    parser.add_argument("--data-split", choices=["train", "holdout", "both"],
                        default="both")
    parser.add_argument("--train-end-demo", type=int, default=45)
    parser.add_argument("--n-cond-marginal", type=int,
                        default=N_CONDITIONS_MARGINAL)
    parser.add_argument("--n-samples-marginal", type=int,
                        default=N_SAMPLES_PER_CONDITION_MARGINAL)
    parser.add_argument("--n-cond-conditional", type=int,
                        default=N_CONDITIONS_CONDITIONAL)
    parser.add_argument("--n-samples-conditional", type=int,
                        default=N_SAMPLES_PER_CONDITION_CONDITIONAL)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    DIAGNOSTIC_BASE_SEED = args.seed
    logger.info("DIAGNOSTIC_BASE_SEED = %d", DIAGNOSTIC_BASE_SEED)

    # Set initial RNG (the per-sample seeding will override during loops)
    set_sample_rng(DIAGNOSTIC_BASE_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # ---- 1. Compute GT range from train (single source of normalization) ----
    logger.info("===== Phase 1: Compute GT range from train =====")
    gt_range = compute_train_gt_range(
        npz_root=Path("data/object_poses"),
        train_end_demo=args.train_end_demo,
    )

    # ---- 2. Load model ----
    logger.info("===== Phase 2: Load model =====")
    model, cfg = load_model_from_ckpt(args.ckpt, args.config, device)
    head_cfg = dict(cfg.model.otp_head) if hasattr(cfg.model, "otp_head") else {}
    num_objects = head_cfg.get("num_objects", 5)

    # ---- 3. Run diagnostic loops per split ----
    splits_to_run = ["train", "holdout"] if args.data_split == "both" else [args.data_split]
    results: Dict[str, Dict] = {}

    for split in splits_to_run:
        logger.info("===== Phase 3.%s: Build loader =====", split)
        loader = build_diagnostic_loader(cfg, split, args.train_end_demo)
        n_total = len(loader.dataset)
        logger.info("  %s split: %d total windows", split, n_total)

        # Stride sample condition indices
        marginal_cond_idx = select_condition_indices(n_total, args.n_cond_marginal)
        conditional_cond_idx = select_condition_indices(n_total, args.n_cond_conditional)

        # Path 1: Marginal
        logger.info("===== Phase 4.%s: Path 1 marginal (n_cond=%d × n_samples=%d) =====",
                    split, args.n_cond_marginal, args.n_samples_marginal)
        cfm_marginal, gt_marginal = run_path_1_marginal(
            model, loader, marginal_cond_idx, args.n_samples_marginal,
            DIAGNOSTIC_BASE_SEED, device, amp_dtype, num_objects,
        )

        # Path 2: Conditional
        logger.info("===== Phase 5.%s: Path 2 conditional (n_cond=%d × n_samples=%d) =====",
                    split, args.n_cond_conditional, args.n_samples_conditional)
        # Reset loader iterator (it's been consumed)
        loader = build_diagnostic_loader(cfg, split, args.train_end_demo)
        cond_results = run_path_2_conditional(
            model, loader, conditional_cond_idx, args.n_samples_conditional,
            DIAGNOSTIC_BASE_SEED + 1,  # offset seed namespace from Path 1
            device, amp_dtype, num_objects, gt_range,
        )

        results[split] = {
            "marginal_cfm": cfm_marginal,
            "marginal_gt": gt_marginal,
            "conditional": cond_results,
        }

    # ---- 4. Write JSON outputs ----
    logger.info("===== Phase 6: Write outputs =====")

    # conditional_variance.json (paper-ready schema)
    cv_data = {
        "gt_range_per_dim": gt_range.tolist(),
        "gt_range_source": "train Q99-Q1 (canonical reference for both splits)",
        "dim_labels": DIM_LABELS,
        "n_conditions": args.n_cond_conditional,
        "n_samples_per_condition": args.n_samples_conditional,
    }
    for split in splits_to_run:
        c = results[split]["conditional"]
        cv_data[split] = {
            "conditional_std_per_dim": c["conditional_std_per_dim"].tolist(),
            "normalized_cond_std_per_dim": c["normalized_cond_std_per_dim"].tolist(),
        }
    with open(args.output_dir / "conditional_variance.json", "w") as f:
        json.dump(cv_data, f, indent=2)

    # conditional_coverage.json
    cov_data = {"dim_labels": DIM_LABELS}
    for split in splits_to_run:
        c = results[split]["conditional"]
        cov_data[split] = {
            "per_condition_coverage": c["per_condition_coverage"].tolist(),
            "mean_coverage": float(c["per_condition_coverage"].mean()),
        }
    with open(args.output_dir / "conditional_coverage.json", "w") as f:
        json.dump(cov_data, f, indent=2)

    # per_condition_dump.npz (raw samples for offline re-analysis)
    npz_dump = {"gt_range": gt_range, "dim_labels": np.array(DIM_LABELS)}
    for split in splits_to_run:
        c = results[split]["conditional"]
        npz_dump[f"{split}_samples"] = c["per_condition_samples"]
        npz_dump[f"{split}_gt"] = c["per_condition_gt"]
        npz_dump[f"{split}_marginal_cfm"] = results[split]["marginal_cfm"]
        npz_dump[f"{split}_marginal_gt"] = results[split]["marginal_gt"]
    np.savez_compressed(args.output_dir / "per_condition_dump.npz", **npz_dump)

    # ---- 5. Plots ----
    if args.data_split == "both":
        logger.info("Plotting marginal overlay...")
        plot_marginal_overlay(
            cfm_train=results["train"]["marginal_cfm"],
            cfm_holdout=results["holdout"]["marginal_cfm"],
            gt_train=results["train"]["marginal_gt"],
            gt_holdout=results["holdout"]["marginal_gt"],
            output_path=args.output_dir / "marginal_overlay.png",
        )

        logger.info("Plotting conditional summary (7-subplot per-dim, paper-ready)...")
        plot_conditional_summary(
            samples_train=results["train"]["conditional"]["per_condition_samples"],
            gt_train=results["train"]["conditional"]["per_condition_gt"],
            samples_hold=results["holdout"]["conditional"]["per_condition_samples"],
            gt_hold=results["holdout"]["conditional"]["per_condition_gt"],
            gt_range=gt_range,
            output_path=args.output_dir / "conditional_summary.png",
        )

    # ---- 6. README.md ----
    write_readme(args.output_dir, args, gt_range, results, splits_to_run)

    logger.info("===== DONE =====")
    logger.info("Output: %s", args.output_dir)


def write_readme(output_dir, args, gt_range, results, splits):
    """Headline numbers + paper-ready summary."""
    lines = [
        f"# CFM Action Distribution Diagnostic",
        f"",
        f"- **Ckpt**: `{args.ckpt}`",
        f"- **Config**: `{args.config}`",
        f"- **DIAGNOSTIC_BASE_SEED**: {args.seed}",
        f"- **Splits**: {', '.join(splits)}",
        f"- **train_end_demo**: {args.train_end_demo}",
        f"",
        f"## Reference scale: GT range (TRAIN Q99-Q1)",
        f"",
        f"| Dim | Label | GT range |",
        f"|-----|-------|----------|",
    ]
    for d, lbl in enumerate(DIM_LABELS):
        lines.append(f"| {d} | {lbl} | {gt_range[d]:.4f} |")

    lines.append(f"")
    lines.append(f"## Path 2: Normalized conditional std")
    lines.append(f"")
    lines.append(f"`normalized = mean(per_condition_std) / gt_range`")
    lines.append(f"")
    lines.append(f"Interpretation:")
    lines.append(f"- ≈ 1.0: CFM spread = GT range, single sample is lottery")
    lines.append(f"- ≈ 0.5: moderately spread, mean estimator helps")
    lines.append(f"- ≈ 0.1: tightly concentrated, single sample likely OK")
    lines.append(f"")
    header = "| Dim | Label |"
    sep = "|---|---|"
    for split in splits:
        header += f" {split} normalized |"
        sep += "---|"
    lines.append(header)
    lines.append(sep)
    for d, lbl in enumerate(DIM_LABELS):
        row = f"| {d} | {lbl} |"
        for split in splits:
            v = results[split]["conditional"]["normalized_cond_std_per_dim"][d]
            row += f" {v:.4f} |"
        lines.append(row)

    lines.append(f"")
    lines.append(f"### Summary mean (over dims)")
    for split in splits:
        m = results[split]["conditional"]["normalized_cond_std_per_dim"].mean()
        lines.append(f"- **{split}**: {m:.4f}")

    lines.append(f"")
    lines.append(f"## Path 2: Coverage (CFM spread ⊇ GT)")
    lines.append(f"")
    lines.append(f"Fraction of (timestep, dim) cells where GT ∈ [CFM_min, CFM_max].")
    lines.append(f"")
    for split in splits:
        cov = results[split]["conditional"]["per_condition_coverage"]
        lines.append(f"- **{split}** (n={len(cov)} conditions): "
                     f"mean = {cov.mean():.4f}, "
                     f"min = {cov.min():.4f}, "
                     f"max = {cov.max():.4f}")

    if "train" in splits and "holdout" in splits:
        train_norm = results["train"]["conditional"]["normalized_cond_std_per_dim"]
        hold_norm = results["holdout"]["conditional"]["normalized_cond_std_per_dim"]
        ratio = hold_norm / train_norm
        lines.append(f"")
        lines.append(f"## Train vs holdout ratio")
        lines.append(f"")
        lines.append(f"`ratio = holdout_normalized / train_normalized`")
        lines.append(f"")
        lines.append(f"| Dim | Label | Ratio |")
        lines.append(f"|-----|-------|-------|")
        for d, lbl in enumerate(DIM_LABELS):
            lines.append(f"| {d} | {lbl} | {ratio[d]:.3f} |")
        lines.append(f"")
        lines.append(f"- Mean ratio over dims: {ratio.mean():.3f}")
        lines.append(f"- Ratio > 1.0 → holdout has higher CFM spread (gen gap)")
        lines.append(f"- Ratio ≈ 1.0 → no observable gen gap in CFM variance")

    lines.append(f"")
    lines.append(f"## Files in this directory")
    lines.append(f"")
    lines.append(f"- `conditional_variance.json`: full numerical data, paper-ready schema")
    lines.append(f"- `conditional_coverage.json`: per-condition coverage rates")
    lines.append(f"- `per_condition_dump.npz`: raw samples for offline re-analysis")
    lines.append(f"- `marginal_overlay.png`: 7-subplot per-dim distribution comparison")
    lines.append(f"- `conditional_summary.png`: 7-subplot per-dim conditional spread, paper-ready")

    with open(output_dir / "README.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
