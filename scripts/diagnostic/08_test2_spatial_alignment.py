"""
Phase 0 Test 2: Spatial alignment — does head z position correlate with target object world position?

V6 revisions:
- M2: verdict logic uses permutation p ONLY. R² reported as effect size but not gating verdict.
      Rationale: permutation p measures "signal exists above chance", R² measures "fraction of
      variance explained". Combining with AND produces logically incoherent verdicts in
      small-n regimes.

Spec (V6 design §2.4):
- 10 LIBERO-Spatial tasks (doubled from V3's 5), 20 seeds per task, mean to μ_t^pos (3-dim)
- Affine regression μ = A p + b, report R² + 1000-shuffle permutation p

FROZEN judgement [M2]:
- permutation p < 0.01           → head spatial-aware (problem in decoder)
- permutation p ∈ [0.01, 0.05]   → borderline, joint judgement with Test 1 (document)
- permutation p > 0.05           → head failed to learn spatial structure

R² is reported alongside as effect size.

Output: results/phase0/test2_spatial_alignment.json
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from otp.eval.predictor import OTPSoftPredictor
from otp.data.libero_loader import build_loader, get_target_object_world_position

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_TASKS = 10
N_SEEDS = 20
N_PERM = 1000
ALPHA_STRONG = 0.01
ALPHA_BORDERLINE_UPPER = 0.05


def collect_mu_pos(predictor, task_ids, n_seeds):
    """Returns μ of shape (K, 3) — mean over seeds of z[obj=target, t=0, :3]."""
    K = len(task_ids)
    mu = np.zeros((K, 3), dtype=np.float64)
    for ti, task_id in enumerate(task_ids):
        loader = build_loader(predictor.config_path, batch_size=1, split="train",
                              task_ids=[task_id])
        batch = next(iter(loader))
        target_obj_idx = 0
        z_samples = []
        for si in range(n_seeds):
            with torch.no_grad():
                z = predictor.head_forward_only(batch, seed=si)
            z_samples.append(z[0, target_obj_idx, 0, :3].cpu().numpy())
        mu[ti] = np.mean(z_samples, axis=0)
        print(f"  task {task_id}: μ_pos = {mu[ti]}")
    return mu


def affine_regress(P, mu):
    """Fit μ = A P + b via least squares. Return R²."""
    K = P.shape[0]
    P_aug = np.hstack([P, np.ones((K, 1))])
    coef, *_ = np.linalg.lstsq(P_aug, mu, rcond=None)
    mu_pred = P_aug @ coef
    ss_res = ((mu - mu_pred) ** 2).sum()
    ss_tot = ((mu - mu.mean(axis=0)) ** 2).sum()
    R2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return R2


def permutation_test(P, mu, n_perm=N_PERM, seed=20260511):
    """Shuffle P row indices, recompute R². Return p-value (fraction of null ≥ observed)."""
    rng = np.random.default_rng(seed)
    R2_obs = affine_regress(P, mu)
    K = P.shape[0]
    R2_null = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(K)
        R2_null[i] = affine_regress(P[perm], mu)
    pval = float((R2_null >= R2_obs).mean())
    return R2_obs, pval, R2_null


def main():
    ckpt = (REPO_ROOT / ".paper_ready_ckpt").read_text().strip()
    print(f"=== Phase 0 Test 2: Spatial Alignment (V6) ===\nCheckpoint: {ckpt}\n")

    predictor = OTPSoftPredictor(
        ckpt_path=ckpt,
        config_path=str(REPO_ROOT / "configs" / "otp_soft_30e.yaml"),
        grasp_affordance_dir=str(REPO_ROOT / "data" / "grasp_affordance"),
        device="cuda",
        log_diagnostics=False,
    )
    predictor.config_path = REPO_ROOT / "configs" / "otp_soft_30e.yaml"

    task_ids = list(range(K_TASKS))

    print("Extracting target object world positions...")
    P = np.array([get_target_object_world_position(tid) for tid in task_ids])
    print(f"P shape: {P.shape}")
    print(f"P range: x [{P[:, 0].min():.3f}, {P[:, 0].max():.3f}], "
          f"y [{P[:, 1].min():.3f}, {P[:, 1].max():.3f}], "
          f"z [{P[:, 2].min():.3f}, {P[:, 2].max():.3f}]")

    print(f"\nCollecting μ_pos for {K_TASKS} tasks, {N_SEEDS} seeds each...")
    mu = collect_mu_pos(predictor, task_ids, N_SEEDS)
    np.save(OUT_DIR / "test2_mu_pos.npy", mu)
    np.save(OUT_DIR / "test2_P.npy", P)

    print(f"\nRunning affine regression + {N_PERM}-permutation test...")
    R2_obs, pval, R2_null = permutation_test(P, mu)
    print(f"  R² observed: {R2_obs:.4f}    (reported as effect size)")
    print(f"  Permutation p-value: {pval:.4f}")
    print(f"  R² null mean ± std: {R2_null.mean():.3f} ± {R2_null.std():.3f}")
    print(f"  R² null 95-pctile: {np.percentile(R2_null, 95):.3f}")

    # [M2] Verdict uses permutation p only
    if pval < ALPHA_STRONG:
        verdict = "head_spatial_aware"
    elif pval <= ALPHA_BORDERLINE_UPPER:
        verdict = "borderline_joint_with_test1"
    else:
        verdict = "head_no_spatial_structure"

    summary = {
        "checkpoint": ckpt,
        "version": "V6",
        "n_tasks": K_TASKS,
        "n_seeds": N_SEEDS,
        "n_perm": N_PERM,
        "alpha_strong": ALPHA_STRONG,
        "alpha_borderline_upper": ALPHA_BORDERLINE_UPPER,
        "R2_observed": float(R2_obs),
        "R2_null_mean": float(R2_null.mean()),
        "R2_null_std": float(R2_null.std()),
        "R2_null_p95": float(np.percentile(R2_null, 95)),
        "permutation_pval": float(pval),
        "verdict": verdict,
        "verdict_logic": "M2: permutation p only, R² is effect size",
        "P_world": P.tolist(),
        "mu_pos": mu.tolist(),
    }
    out_path = OUT_DIR / "test2_spatial_alignment.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {out_path}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
