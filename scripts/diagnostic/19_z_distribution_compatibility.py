"""
Path B Step 3: z-distribution compatibility test.

Per V7 §IV Step 3 (first-class). Compares the trajectory output distribution of:
  (a) V3 trained CFM head — loaded from test1_Z.npy (existing Phase 0 dump)
  (b) Path B random-init deterministic head — sampled here

Decision rule:
  - If distributions are similar (cosine sim of mean vectors > 0.5 AND
    per-dim std ratios in [0.5, 2.0]): V3 decoder weights MAY be inheritable
    as initialization for Path B retrain.
  - If not similar: Path B decoder must train from scratch.

This is a risk-check per V7 frozen protocol. Random-init Path B is expected
to differ substantially from trained V3 (this is the predicted outcome);
script formally records the comparison as decision rationale.

Output: results/phase0/z_distribution_compatibility.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from otp.models.otp_head import OTPHead

OUT_DIR = REPO_ROOT / "results" / "phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("Path B Step 3: z-Distribution Compatibility Test")
    print("=" * 70)

    # ── Load V3 z dump ──────────────────────────────────────────────
    v3_path = OUT_DIR / "test1_Z.npy"
    v3_z = np.load(v3_path)  # (5 tasks, 20 seeds, 5 obj, 8 H, 6 SE3)
    print(f"\nV3 z (loaded from {v3_path.name}):")
    print(f"  shape: {v3_z.shape}")

    # Flatten V3 z over tasks+seeds → (100, 5, 8, 6) for direct comparison
    v3_z_flat = v3_z.reshape(-1, 5, 8, 6)  # (100, 5, 8, 6)
    print(f"  flattened: {v3_z_flat.shape}")

    # ── Build Path B random-init head ──────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)
    head_pathB = OTPHead(
        backbone_dim=4096, hidden_dim=512, num_objects=5, horizon=8,
        deterministic=True,
    ).eval()

    n_trainable = sum(p.numel() for p in head_pathB.parameters() if p.requires_grad)
    print(f"\nPath B head (deterministic, random-init):")
    print(f"  trainable params: {n_trainable / 1e6:.2f}M")

    # ── Generate 100 Path B z samples ───────────────────────────────
    # Path B is deterministic given input → varying input gives varying output.
    # Use random backbone input to probe the random-init output distribution.
    print("\nSampling 100 Path B trajectories on random backbone input...")
    B = 100
    S = 100  # seq len
    backbone_hidden = torch.randn(B, S, 4096) * 0.5  # similar scale to real
    attn_mask = torch.ones(B, S, dtype=torch.bool)
    object_indices = torch.zeros(B, 5, dtype=torch.long)

    with torch.no_grad():
        out = head_pathB(backbone_hidden, attn_mask, object_indices, gt_trajectory=None)
    pathB_z = out["trajectories"].numpy()  # (100, 5, 8, 6)
    print(f"  Path B z shape: {pathB_z.shape}")

    # ── Compute distribution statistics ─────────────────────────────
    print("\n" + "=" * 70)
    print("Distribution comparison")
    print("=" * 70)

    # Overall statistics
    v3_mean = v3_z_flat.mean()
    v3_std = v3_z_flat.std()
    v3_min = v3_z_flat.min()
    v3_max = v3_z_flat.max()
    pathB_mean = pathB_z.mean()
    pathB_std = pathB_z.std()
    pathB_min = pathB_z.min()
    pathB_max = pathB_z.max()

    print(f"\nOverall (over all (obj, H, dim)):")
    print(f"  V3:     mean={v3_mean:+.4f}  std={v3_std:.4f}  range=[{v3_min:+.3f}, {v3_max:+.3f}]")
    print(f"  PathB:  mean={pathB_mean:+.4f}  std={pathB_std:.4f}  range=[{pathB_min:+.3f}, {pathB_max:+.3f}]")
    print(f"  std ratio (V3/PathB): {v3_std / pathB_std:.3f}")

    # Per-trajectory norm
    v3_norms = np.linalg.norm(v3_z_flat.reshape(len(v3_z_flat), -1), axis=1)
    pathB_norms = np.linalg.norm(pathB_z.reshape(len(pathB_z), -1), axis=1)
    print(f"\nPer-sample flat-vector norm (n=100):")
    print(f"  V3 norm:    mean={v3_norms.mean():.2f}  std={v3_norms.std():.2f}")
    print(f"  PathB norm: mean={pathB_norms.mean():.2f}  std={pathB_norms.std():.2f}")
    print(f"  norm ratio (V3/PathB): {v3_norms.mean() / pathB_norms.mean():.3f}")

    # Per-dim (240-dim flat) std comparison
    v3_flat = v3_z_flat.reshape(len(v3_z_flat), -1)         # (100, 240)
    pathB_flat = pathB_z.reshape(len(pathB_z), -1)            # (100, 240)
    v3_per_dim_std = v3_flat.std(axis=0)                     # (240,)
    pathB_per_dim_std = pathB_flat.std(axis=0)
    per_dim_ratio = v3_per_dim_std / (pathB_per_dim_std + 1e-9)
    print(f"\nPer-dim std (240 dims):")
    print(f"  V3 std:    mean={v3_per_dim_std.mean():.4f}, range [{v3_per_dim_std.min():.4f}, {v3_per_dim_std.max():.4f}]")
    print(f"  PathB std: mean={pathB_per_dim_std.mean():.4f}, range [{pathB_per_dim_std.min():.4f}, {pathB_per_dim_std.max():.4f}]")
    print(f"  Per-dim std ratio: mean {per_dim_ratio.mean():.3f}, median {np.median(per_dim_ratio):.3f}")

    # Mean vector cosine similarity (over 240-dim)
    v3_mean_vec = v3_flat.mean(axis=0)                       # (240,)
    pathB_mean_vec = pathB_flat.mean(axis=0)
    cos_sim = np.dot(v3_mean_vec, pathB_mean_vec) / (
        np.linalg.norm(v3_mean_vec) * np.linalg.norm(pathB_mean_vec) + 1e-9
    )
    print(f"\nMean-vector cosine similarity: {cos_sim:+.4f}")

    # Covariance eigenvalue spectrum (top-5 eigvals as sanity)
    print(f"\nCovariance eigenvalue spectrum (top 5):")
    v3_cov = np.cov(v3_flat.T)                                # (240, 240)
    pathB_cov = np.cov(pathB_flat.T)
    v3_eigvals = np.linalg.eigvalsh(v3_cov)[::-1][:5]
    pathB_eigvals = np.linalg.eigvalsh(pathB_cov)[::-1][:5]
    print(f"  V3 top-5:    {v3_eigvals}")
    print(f"  PathB top-5: {pathB_eigvals}")

    # ── Decision ────────────────────────────────────────────────────
    median_ratio = float(np.median(per_dim_ratio))
    can_inherit = (abs(cos_sim) > 0.5) and (0.5 <= median_ratio <= 2.0)

    print("\n" + "=" * 70)
    print("Decision (V7 §IV Step 3)")
    print("=" * 70)
    print(f"  cosine sim > 0.5: {abs(cos_sim) > 0.5}  (actual {cos_sim:+.4f})")
    print(f"  per-dim std ratio in [0.5, 2.0]: {0.5 <= median_ratio <= 2.0}  (actual {median_ratio:.3f})")
    print(f"  V3 decoder weights inheritable: {can_inherit}")
    if not can_inherit:
        print(f"\n  → DECISION: Path B decoder retrains from scratch.")
        print(f"     (Expected outcome: random-init head produces a distribution")
        print(f"      structurally different from trained V3 head.)")
    else:
        print(f"\n  → DECISION: V3 decoder weights may be used as initialization.")

    summary = {
        "version": "z_distribution_compatibility_v1",
        "v3_z_source": str(v3_path),
        "v3_z_shape": list(v3_z.shape),
        "pathB_z_shape": list(pathB_z.shape),
        "v3_overall_mean": float(v3_mean),
        "v3_overall_std": float(v3_std),
        "v3_norm_mean": float(v3_norms.mean()),
        "pathB_overall_mean": float(pathB_mean),
        "pathB_overall_std": float(pathB_std),
        "pathB_norm_mean": float(pathB_norms.mean()),
        "std_ratio_v3_over_pathB": float(v3_std / pathB_std),
        "norm_ratio_v3_over_pathB": float(v3_norms.mean() / pathB_norms.mean()),
        "per_dim_std_median_ratio": median_ratio,
        "mean_vector_cosine_similarity": float(cos_sim),
        "v3_top5_eigvals": v3_eigvals.tolist(),
        "pathB_top5_eigvals": pathB_eigvals.tolist(),
        "can_inherit_v3_decoder_weights": bool(can_inherit),
        "decision_threshold_cosine": 0.5,
        "decision_threshold_std_ratio_lo": 0.5,
        "decision_threshold_std_ratio_hi": 2.0,
    }
    out_path = OUT_DIR / "z_distribution_compatibility.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {out_path}")


if __name__ == "__main__":
    main()
