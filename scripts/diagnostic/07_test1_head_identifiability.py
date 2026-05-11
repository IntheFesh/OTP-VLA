"""
Phase 0 Test 1: Head identifiability via per-slice MANOVA + per-dim F-test (V6.1).

V6 frozen items (per notes/2026-05-11_phase0_preregistration_v6.md):
- 5 tasks × 20 seeds = 100 head forwards, head only
- Per-slice 6-dim MANOVA via statsmodels Wilks Λ Rao-F
- BH-FDR across 40 slices, q = 0.05
- Second layer: per-dim F + η² exploratory (no FDR, m=6 makes BH ≈ Bonferroni)
- F(4, 95) critical: F_{0.05} = 2.469, F_{0.001} = 5.128

V6.1 interface fixes:
- Use LIBEROOTPDataset + collate_fn + assemble_batch (matches train loop)
- Use dataset._index + dataset.demo_records for task-level sample filtering
- predictor.head_forward_only as the head-only entry point

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
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from otp.eval.predictor import OTPSoftPredictor
from otp.data.libero_loader import LIBEROOTPDataset
from otp.train.utils import collate_fn, assemble_batch

PAPER_CKPT_FILE = REPO_ROOT / ".paper_ready_ckpt"
CONFIG_PATH = REPO_ROOT / "configs" / "otp_soft_30e.yaml"
OUT_DIR = REPO_ROOT / "results" / "phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# FROZEN per V6 pre-registration §2.2, §2.3
K_TASKS = 5
N_SEEDS = 20
N_OBJ = 5
H = 8
D = 6
F_CRIT_005 = 2.469
F_CRIT_0001 = 5.128
FDR_Q = 0.05

# LIBERO-Spatial 10 task names (sorted by ls order)
PROBE_TASK_NAMES = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
]


def load_resolved_config():
    config_dir = str(REPO_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="otp_soft_30e")
    return cfg


def build_train_dataset(cfg):
    norm_path = getattr(cfg.data, "normalizer_path", None)
    return LIBEROOTPDataset(
        root=Path(cfg.data.root),
        suite=cfg.data.suite,
        grasp_affordance_dir=Path(cfg.data.grasp_affordance_dir),
        normalizer_path=Path(norm_path) if norm_path else None,
        num_grasps_per_object=cfg.model.decoder.num_grasps_per_object,
        num_points=cfg.model.geometry_encoder.num_points,
        horizon=cfg.model.otp_head.horizon,
        train_end_demo=45,
    )


def get_one_batch_from_task(dataset, task_name, device, amp_dtype, num_objects=5):
    """Return a forward-ready B=1 batch from given task. Uses first matching sample."""
    for i in range(len(dataset)):
        demo_idx, _frame = dataset._index[i]
        record = dataset.demo_records[demo_idx]
        if record["npz_path"].parent.name == task_name:
            sample = dataset[i]
            raw_batch = collate_fn([sample])
            return assemble_batch(raw_batch, device, amp_dtype, num_objects)
    raise RuntimeError(f"No samples from task '{task_name}'")


def collect_z_array(predictor, dataset, task_names, n_seeds, num_objects=5):
    """Collect z of shape (K, N, N_obj, H, D). Same batch per task (frame 0), n_seeds different CFM noise."""
    K = len(task_names)
    Z = np.zeros((K, n_seeds, N_OBJ, H, D), dtype=np.float64)
    for ti, name in enumerate(task_names):
        batch = get_one_batch_from_task(
            dataset, name, predictor.device, predictor.amp_dtype, num_objects=num_objects
        )
        for si in range(n_seeds):
            traj = predictor.head_forward_only(batch, seed=si)  # (1, 5, 8, 6)
            Z[ti, si] = traj[0].cpu().numpy()
        print(f"  task {ti}: {n_seeds} seeds collected")
    return Z


def manova_via_statsmodels(Z_slice):
    """Per-slice 6-dim MANOVA via statsmodels. Z_slice: (K, N, D).

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
        return {
            "wilks_lambda": float("nan"), "wilks_F": float("nan"),
            "wilks_df_num": float("nan"), "wilks_df_den": float("nan"),
            "wilks_pval": 1.0, "pillai_trace": float("nan"), "pillai_pval": 1.0,
            "hotelling_trace": float("nan"), "hotelling_pval": 1.0,
            "roy_root": float("nan"), "roy_pval": 1.0,
            "pval_primary": 1.0, "manova_error": str(e),
        }


def univariate_F_with_effect_size(Z_slice_dim):
    """A1: F + p + η² for one dim. Z_slice_dim: (K, N)."""
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
    eta2 = SSB / (SSB + SSW + 1e-12)
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
    print("=== Phase 0 Test 1: Head Identifiability (V6.1) ===\n")
    ckpt = PAPER_CKPT_FILE.read_text().strip()
    print(f"Checkpoint: {ckpt}\n")

    cfg = load_resolved_config()
    dataset = build_train_dataset(cfg)
    print(f"Dataset: {len(dataset)} samples, {len(dataset.demo_records)} demo records\n")

    predictor = OTPSoftPredictor(
        ckpt_path=ckpt,
        config_path=str(CONFIG_PATH),
        grasp_affordance_dir=str(Path(cfg.data.grasp_affordance_dir)),
        device=torch.device("cuda"),
        log_diagnostics=False,
    )
    predictor.reset(episode_seed=0)
    print(f"Predictor: device={predictor.device}, amp_dtype={predictor.amp_dtype}\n")

    print(f"Collecting z for {K_TASKS} tasks × {N_SEEDS} seeds...")
    Z = collect_z_array(
        predictor, dataset, PROBE_TASK_NAMES[:K_TASKS], N_SEEDS,
        num_objects=cfg.model.otp_head.num_objects,
    )
    np.save(OUT_DIR / "test1_Z.npy", Z)
    print(f"Z shape: {Z.shape}, saved to test1_Z.npy\n")

    # Layer 1: per-slice MANOVA
    print(f"Layer 1: per-slice MANOVA across {N_OBJ * H} = {N_OBJ * H} slices (statsmodels Wilks Λ Rao-F)...")
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
    print(f"  BH-FDR significant slices: {n_sig} / {N_OBJ * H}")
    if n_sig > 0:
        sig_indices = np.where(sig_mask)[0]
        sample_show = sig_indices[:5]
        for idx in sample_show:
            r = slice_records[idx]
            print(f"    sig slice (obj={r['obj']}, t={r['t']}): "
                  f"Λ={r['wilks_lambda']:.4f}, F={r['wilks_F']:.2f}, p={r['pval_primary']:.2e}")

    # Layer 2 [A2]: per-dim F + η², no FDR — exploratory
    print(f"\nLayer 2 (exploratory, no FDR): per-dim F + η² on {N_OBJ * H - n_sig} non-significant slices...")
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
                "dim": d, "F": F_val, "pval_raw": pval, "eta2": eta2,
                "verdict": verdict,
            })
        r["per_dim_exploratory"] = per_dim

    # Aggregate verdicts: count how many of (per-dim cells over all non-sig slices) fall into each category
    if n_sig < N_OBJ * H:
        all_per_dim = []
        for r in slice_records:
            if not r["bh_significant"]:
                all_per_dim.extend(r["per_dim_exploratory"])
        v_counts = {}
        for d in all_per_dim:
            v_counts[d["verdict"]] = v_counts.get(d["verdict"], 0) + 1
        print(f"  Per-dim verdict counts in non-sig slices: {v_counts}")
        # Print median F and η² across non-sig per-dim cells
        all_F = [d["F"] for d in all_per_dim]
        all_eta2 = [d["eta2"] for d in all_per_dim]
        print(f"  F (per-dim, non-sig slices): median={np.median(all_F):.3f}, "
              f"max={np.max(all_F):.3f}, p95={np.percentile(all_F, 95):.3f}")
        print(f"  η² (per-dim, non-sig slices): median={np.median(all_eta2):.3f}, "
              f"max={np.max(all_eta2):.3f}")

    # Global verdict per V6 §2.3
    if n_sig >= 1:
        global_verdict = "head_has_identifiability"
    else:
        global_verdict = "head_global_collapse"

    summary = {
        "checkpoint": ckpt,
        "version": "V6.1",
        "n_tasks": K_TASKS, "n_seeds": N_SEEDS,
        "probe_task_names": PROBE_TASK_NAMES[:K_TASKS],
        "critical_values": {
            "F_005": F_CRIT_005, "F_0001": F_CRIT_0001, "BH_q": FDR_Q,
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
