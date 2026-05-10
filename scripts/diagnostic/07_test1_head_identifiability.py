"""
Phase 0 Test 1: Head identifiability via per-slice MANOVA + per-dim F-test.

V6 revisions:
- M1: MANOVA via statsmodels.multivariate.manova, Rao-F approximation (not hand-Bartlett)
- A1: report η² effect size alongside F and p
- A2: second-layer per-dim test is exploratory, no FDR (rotation dims share quaternion
      dependence; m=6 makes BH ≈ Bonferroni; framed as collapse localization diagnostic)

Spec (V6 design §2.2 & §2.3):
- 5 tasks × 20 seeds = 100 head forward passes, head only
- z shape (5, 8, 6) → 40 slices of dim 6
- Layer 1: per-slice 6-dim MANOVA via statsmodels Wilks Λ + Rao-F, BH-FDR across 40, q=0.05
- Layer 2 (exploratory, no FDR): for slices NOT BH-significant, per-dim univariate
  F-test (df1=4, df2=95), report F, p, η². No multiple-comparison correction at this layer.

Critical values (FROZEN per pre-registration):
- F(4, 95): F_0.05 = 2.469, F_0.001 = 5.128
- MANOVA BH-FDR q = 0.05, layer 1 across 40 slices only

Output: results/phase0/test1_head_identifiability.json
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.multivariate.manova import MANOVA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from otp.eval.predictor import OTPSoftPredictor
from otp.data.libero_loader import build_loader

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# FROZEN constants
K_TASKS = 5
N_SEEDS = 20
N_OBJ = 5
H = 8
D = 6
F_CRIT_005 = 2.469
F_CRIT_0001 = 5.128
FDR_Q = 0.05


def collect_z(predictor, task_ids, n_seeds):
    """Collect z of shape (K, N, N_obj, H, D)."""
    Z = np.zeros((len(task_ids), n_seeds, N_OBJ, H, D), dtype=np.float64)
    for ti, task_id in enumerate(task_ids):
        loader = build_loader(predictor.config_path, batch_size=1, split="train",
                              task_ids=[task_id])
        batch = next(iter(loader))
        for si in range(n_seeds):
            with torch.no_grad():
                z = predictor.head_forward_only(batch, seed=si)
            Z[ti, si] = z[0].cpu().numpy()
        print(f"  collected task {task_id}: {n_seeds} seeds")
    return Z


def manova_via_statsmodels(Z_slice):
    """[M1] Per-slice 6-dim MANOVA via statsmodels.

    Z_slice: (K, N, D) array.
    Returns dict with all 4 MANOVA test statistics + primary (Wilks Λ Rao-F) p-value.
    """
    K, N, D_ = Z_slice.shape
    rows = []
    for k in range(K):
        for n in range(N):
            row = {f"d{j}": Z_slice[k, n, j] for j in range(D_)}
            row["task"] = k
            rows.append(row)
    df = pd.DataFrame(rows)
    formula = " + ".join([f"d{j}" for j in range(D_)]) + " ~ C(task)"

    try:
        mv = MANOVA.from_formula(formula, data=df)
        result = mv.mv_test()
        stat_df = result.results["C(task)"]["stat"]

        wilks = stat_df.loc["Wilks' lambda"]
        pillai = stat_df.loc["Pillai's trace"]
        hotelling = stat_df.loc["Hotelling-Lawley trace"]
        roy = stat_df.loc["Roy's greatest root"]

        return {
            "wilks_lambda": float(wilks["Value"]),
            "wilks_F": float(wilks["F Value"]),
            "wilks_df_num": float(wilks["Num DF"]),
            "wilks_df_den": float(wilks["Den DF"]),
            "wilks_pval": float(wilks["Pr > F"]),
            "pillai_trace": float(pillai["Value"]),
            "pillai_pval": float(pillai["Pr > F"]),
            "hotelling_trace": float(hotelling["Value"]),
            "hotelling_pval": float(hotelling["Pr > F"]),
            "roy_root": float(roy["Value"]),
            "roy_pval": float(roy["Pr > F"]),
            "pval_primary": float(wilks["Pr > F"]),
        }
    except Exception as e:
        # Fallback if statsmodels balks (singular covariance, etc.)
        return {
            "wilks_lambda": float("nan"),
            "wilks_F": float("nan"),
            "wilks_df_num": float("nan"),
            "wilks_df_den": float("nan"),
            "wilks_pval": 1.0,
            "pillai_trace": float("nan"),
            "pillai_pval": 1.0,
            "hotelling_trace": float("nan"),
            "hotelling_pval": 1.0,
            "roy_root": float("nan"),
            "roy_pval": 1.0,
            "pval_primary": 1.0,
            "manova_error": str(e),
        }


def univariate_F_with_effect_size(Z_slice_dim):
    """[A1] One-way ANOVA F + η² effect size.

    Z_slice_dim: (K, N) array. Returns (F, p, η²).
    """
    K, N = Z_slice_dim.shape
    N_total = K * N
    mu_group = Z_slice_dim.mean(axis=1)
    mu_grand = Z_slice_dim.mean()
    SSB = N * ((mu_group - mu_grand) ** 2).sum()
    SSW = ((Z_slice_dim - mu_group.reshape(-1, 1)) ** 2).sum()
    MSB = SSB / (K - 1)
    MSW = SSW / (N_total - K)
    F_val = MSB / (MSW + 1e-12)
    pval = 1.0 - stats.f.cdf(F_val, K - 1, N_total - K)
    eta2 = SSB / (SSB + SSW + 1e-12)  # [A1]
    return float(F_val), float(pval), float(eta2)


def bh_fdr(pvals, q=FDR_Q):
    """Benjamini-Hochberg. Returns boolean mask of significant entries."""
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)
    thresh = q * ranks / n
    sorted_p = pvals[order]
    below = sorted_p <= thresh
    if not below.any():
        return np.zeros(n, dtype=bool)
    max_idx = np.where(below)[0].max()
    sig_in_sorted = np.zeros(n, dtype=bool)
    sig_in_sorted[:max_idx + 1] = True
    sig = np.zeros(n, dtype=bool)
    sig[order] = sig_in_sorted
    return sig


def main():
    ckpt = (REPO_ROOT / ".paper_ready_ckpt").read_text().strip()
    print(f"=== Phase 0 Test 1: Head Identifiability (V6) ===")
    print(f"Checkpoint: {ckpt}\n")

    predictor = OTPSoftPredictor(
        ckpt_path=ckpt,
        config_path=str(REPO_ROOT / "configs" / "otp_soft_30e.yaml"),
        grasp_affordance_dir=str(REPO_ROOT / "data" / "grasp_affordance"),
        device="cuda",
        log_diagnostics=False,
    )
    predictor.config_path = REPO_ROOT / "configs" / "otp_soft_30e.yaml"

    task_ids = list(range(K_TASKS))
    print(f"Collecting z for tasks {task_ids}, {N_SEEDS} seeds each...")
    Z = collect_z(predictor, task_ids, N_SEEDS)
    np.save(OUT_DIR / "test1_Z.npy", Z)

    # Layer 1: per-slice MANOVA via statsmodels
    print(f"\nLayer 1: per-slice MANOVA via statsmodels ({N_OBJ * H} slices)...")
    slice_records = []
    for o in range(N_OBJ):
        for t in range(H):
            Z_slice = Z[:, :, o, t, :]
            manova_result = manova_via_statsmodels(Z_slice)
            manova_result["obj"] = o
            manova_result["t"] = t
            slice_records.append(manova_result)

    pvals = np.array([r["pval_primary"] for r in slice_records])
    sig_mask = bh_fdr(pvals, q=FDR_Q)
    for r, s in zip(slice_records, sig_mask):
        r["bh_significant"] = bool(s)

    n_sig = int(sig_mask.sum())
    print(f"  BH-FDR significant slices (Wilks Λ Rao-F): {n_sig} / {N_OBJ * H}")

    # Layer 2 [A2]: exploratory per-dim F + η², no FDR
    print(f"\nLayer 2 (exploratory, no FDR): per-dim F + η² on {N_OBJ * H - n_sig} "
          f"non-significant slices...")
    for r in slice_records:
        if r["bh_significant"]:
            continue
        o, t = r["obj"], r["t"]
        per_dim = []
        for d in range(D):
            Z_dim = Z[:, :, o, t, d]
            F_val, pval, eta2 = univariate_F_with_effect_size(Z_dim)
            verdict = ("identifiable" if F_val >= F_CRIT_0001
                       else "partial" if F_val >= F_CRIT_005
                       else "collapse")
            per_dim.append({
                "dim": d,
                "F": F_val,
                "pval_raw": pval,
                "eta2": eta2,
                "verdict": verdict,
            })
        r["per_dim_exploratory"] = per_dim

    if n_sig >= 1:
        global_verdict = "head_has_identifiability"
    else:
        global_verdict = "head_global_collapse"

    summary = {
        "checkpoint": ckpt,
        "version": "V6",
        "n_tasks": K_TASKS,
        "n_seeds": N_SEEDS,
        "critical_values": {
            "F_005": F_CRIT_005,
            "F_0001": F_CRIT_0001,
            "BH_q": FDR_Q,
            "manova_method": "statsmodels.multivariate.manova (Wilks Λ Rao-F)",
        },
        "n_slices_total": N_OBJ * H,
        "n_slices_significant": n_sig,
        "global_verdict": global_verdict,
        "slices": slice_records,
    }
    out_path = OUT_DIR / "test1_head_identifiability.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {out_path}")
    print(f"Global verdict: {global_verdict}")


if __name__ == "__main__":
    main()
