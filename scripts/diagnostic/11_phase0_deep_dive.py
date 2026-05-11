"""
Phase 0 Deep Dive: Mechanism-level analysis after Test 1 + Test 2.

This script is NOT part of the V6 frozen pre-registration battery. It addresses
four follow-up questions raised by the Test 1 + Test 2 results:

Q1 (Leverage analysis on Test 2 R² = 0.524, p = 0.164):
    Is the spatial alignment dominated by outliers (task 7, task 8)?
    - Per-dim Pearson & Spearman correlation (P[:,j], μ[:,j]) for j ∈ {x, y, z}
    - Leave-one-out: drop each task in turn, refit affine, recompute R² & permutation p
    - Identify high-leverage tasks via Cook's distance proxy

Q2 (Geometry of head z output across tasks):
    Test 1 says obj=0 strongly discriminates tasks. What is the geometric structure?
    - PCA on per-task mean z slice (obj=0, t=0..4 flattened, 30-dim)
    - Pairwise distance matrix among 5 task means
    - Mantel test: corr(z-pairwise-distances, P-pairwise-distances)
      → tests Q1 and Q2 jointly: if Mantel sig, head encodes task identity in a
        spatial-geometric way; if not, head encodes task identity in some other
        feature space orthogonal to target world position.

Q3 (Head output vs ground-truth trajectory):
    Did CFM training converge to the demonstration distribution mode?
    - For each probe task, compare head μ over 20 seeds vs mean of gt_trajectory
      over many demos of the same task
    - L2 distance per (obj, t) cell
    - Decision signal: head close to GT → problem is decoder-only; head far from
      GT → CFM training itself failed (deeper issue)

Q4 (Numerical amplitude distribution sanity):
    head output vs gt_trajectory amplitude per dimension
    - Are head outputs in same numerical range as training targets?
    - If amplitudes mismatch (head saturates or under-shoots), it's a normalization
      or output-head issue, not a representation issue

Output: results/phase0/deep_dive.json + console summary
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from scipy import stats
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

PROBE_TASK_NAMES = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
]
N_SEEDS = 20
N_DEMOS_PER_TASK_FOR_GT = 10  # for Q3: average gt_trajectory across many demos
N_PERM = 1000


# ============================================================================
# Helpers (reused from Test 1/2)
# ============================================================================

def load_config():
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(config_name="otp_soft_30e")


def build_dataset(cfg):
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


def get_sample_indices_for_task(dataset, task_name, n_samples, frame_idx_filter=0):
    """Return up to n_samples sample indices for given task (preferring frame=frame_idx_filter)."""
    matched_frame0 = []
    matched_other = []
    for i in range(len(dataset)):
        demo_idx, frame_idx = dataset._index[i]
        record = dataset.demo_records[demo_idx]
        if record["npz_path"].parent.name == task_name:
            if frame_idx == frame_idx_filter:
                matched_frame0.append(i)
            else:
                matched_other.append(i)
            if len(matched_frame0) >= n_samples:
                break
    return matched_frame0[:n_samples] if matched_frame0 else matched_other[:n_samples]


def get_batch_from_sample_idx(dataset, idx, device, amp_dtype, num_objects=5):
    sample = dataset[idx]
    raw = collate_fn([sample])
    return assemble_batch(raw, device, amp_dtype, num_objects)


def extract_target_pos(npz_path):
    npz = np.load(npz_path, allow_pickle=False)
    pos = npz["object_poses"][0, 0, :3, 3].astype(np.float64)
    npz.close()
    return pos


def affine_R2(P, mu):
    K = P.shape[0]
    P_aug = np.hstack([P, np.ones((K, 1))])
    coef, *_ = np.linalg.lstsq(P_aug, mu, rcond=None)
    mu_pred = P_aug @ coef
    ss_res = ((mu - mu_pred) ** 2).sum()
    ss_tot = ((mu - mu.mean(axis=0)) ** 2).sum()
    return 1.0 - ss_res / (ss_tot + 1e-12)


def permutation_p(P, mu, n_perm=N_PERM, seed=20260511):
    rng = np.random.default_rng(seed)
    R2_obs = affine_R2(P, mu)
    K = P.shape[0]
    R2_null = np.zeros(n_perm)
    for i in range(n_perm):
        R2_null[i] = affine_R2(P[rng.permutation(K)], mu)
    return R2_obs, float((R2_null >= R2_obs).mean()), R2_null


# ============================================================================
# Q1: Leverage analysis on Test 2
# ============================================================================

def run_Q1_leverage_analysis(P, mu):
    """Per-dim correlations + leave-one-out + leverage proxy."""
    print("\n" + "=" * 70)
    print("Q1: Leverage analysis on Test 2 R² = 0.524, p = 0.164")
    print("=" * 70)

    K = P.shape[0]
    results = {"per_dim_corr": {}, "leave_one_out": [], "leverage_ranking": []}

    # Per-dim Pearson + Spearman
    print("\nPer-dim Pearson & Spearman correlation (P_j, μ_j):")
    for j, name in enumerate(["x", "y", "z"]):
        pearson_r, pearson_p = stats.pearsonr(P[:, j], mu[:, j])
        spearman_r, spearman_p = stats.spearmanr(P[:, j], mu[:, j])
        results["per_dim_corr"][name] = {
            "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        }
        print(f"  {name}-dim: Pearson r={pearson_r:+.3f} (p={pearson_p:.3f}), "
              f"Spearman r={spearman_r:+.3f} (p={spearman_p:.3f})")

    # Leave-one-out R² and permutation p
    print(f"\nLeave-one-out (drop each task, refit on remaining {K-1}):")
    R2_full, p_full, _ = permutation_p(P, mu)
    print(f"  Full ({K} tasks): R² = {R2_full:.4f}, perm p = {p_full:.4f}")
    for k in range(K):
        keep = [i for i in range(K) if i != k]
        P_loo = P[keep]
        mu_loo = mu[keep]
        R2_loo, p_loo, _ = permutation_p(P_loo, mu_loo, seed=20260511 + k)
        delta_R2 = R2_loo - R2_full
        results["leave_one_out"].append({
            "dropped_task": k, "task_name": PROBE_TASK_NAMES[k][:60],
            "R2_remaining": float(R2_loo), "perm_p_remaining": float(p_loo),
            "delta_R2": float(delta_R2),
        })
        flag = " ← drives R² down" if delta_R2 > 0.15 else (" ← drives R² up" if delta_R2 < -0.15 else "")
        print(f"  drop task {k}: R² = {R2_loo:.4f} (Δ={delta_R2:+.4f}), "
              f"p = {p_loo:.4f}{flag}")

    # Leverage ranking by |delta_R2|
    leverage_sorted = sorted(results["leave_one_out"], key=lambda r: abs(r["delta_R2"]), reverse=True)
    results["leverage_ranking"] = [r["dropped_task"] for r in leverage_sorted]
    print(f"\nLeverage ranking (most influential first): {results['leverage_ranking']}")

    return results


# ============================================================================
# Q2: Geometry of head z output via Mantel test
# ============================================================================

def run_Q2_geometry_mantel(z_per_task, P):
    """PCA + Mantel test on z geometry vs target position geometry.

    z_per_task: (K, 30) — per-task mean of z[obj=0, t=0..4, :] flattened
    P:          (K, 3)  — per-task target world position
    """
    print("\n" + "=" * 70)
    print("Q2: Geometry of head z output (obj=0, t=0..4) across tasks")
    print("=" * 70)

    K = z_per_task.shape[0]
    results = {}

    # PCA on z_per_task
    z_centered = z_per_task - z_per_task.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(z_centered, full_matrices=False)
    explained_ratio = (S ** 2) / (S ** 2).sum()
    print(f"\nPCA on z_per_task (K={K} task means, 30-dim each):")
    print(f"  Explained variance ratio (top 5): "
          f"{[f'{r:.3f}' for r in explained_ratio[:5]]}")
    print(f"  Top-3 cumulative: {explained_ratio[:3].sum():.3f}")
    results["pca_explained_ratio"] = explained_ratio.tolist()
    results["pca_top3_cumulative"] = float(explained_ratio[:3].sum())

    # Project to top-3 PCs for visualization (saved)
    z_pca3 = U[:, :3] * S[:3]
    results["z_pca3"] = z_pca3.tolist()
    print(f"  z projected to top-3 PCs (per task):")
    for k in range(K):
        print(f"    task {k}: [{z_pca3[k, 0]:+.3f}, {z_pca3[k, 1]:+.3f}, {z_pca3[k, 2]:+.3f}]")

    # Pairwise distance matrices
    from scipy.spatial.distance import pdist, squareform
    D_z = squareform(pdist(z_per_task, metric="euclidean"))
    D_P = squareform(pdist(P, metric="euclidean"))

    # Vectorize upper triangle for correlation
    iu = np.triu_indices(K, k=1)
    d_z_vec = D_z[iu]
    d_P_vec = D_P[iu]

    # Mantel test via permutation
    mantel_r, _ = stats.pearsonr(d_z_vec, d_P_vec)
    rng = np.random.default_rng(42)
    n_perm = 1000
    null_r = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(K)
        D_P_perm = squareform(pdist(P[perm], metric="euclidean"))
        null_r[i], _ = stats.pearsonr(D_z[iu], D_P_perm[iu])
    mantel_p = float((null_r >= mantel_r).mean())

    print(f"\nMantel test (pairwise distance correlation, {n_perm}-shuffle):")
    print(f"  Mantel r = {mantel_r:+.4f}")
    print(f"  Mantel p = {mantel_p:.4f}")
    print(f"  Null distribution: mean = {null_r.mean():+.3f}, p95 = {np.percentile(null_r, 95):+.3f}")
    results["mantel_r"] = float(mantel_r)
    results["mantel_p"] = float(mantel_p)
    results["mantel_null_mean"] = float(null_r.mean())
    results["D_z_summary"] = {
        "min": float(D_z[iu].min()), "max": float(D_z[iu].max()),
        "mean": float(D_z[iu].mean()),
    }
    results["D_P_summary"] = {
        "min": float(D_P[iu].min()), "max": float(D_P[iu].max()),
        "mean": float(D_P[iu].mean()),
    }

    # Interpretation
    if mantel_p < 0.01:
        interp = "H2.b: head encodes task identity in spatial-geometric way (strong evidence)"
    elif mantel_p < 0.05:
        interp = "H2.b weak: marginal spatial-geometric encoding"
    else:
        interp = "H2.a or H2.c: head encodes task identity in a feature space orthogonal to spatial position"
    results["interpretation"] = interp
    print(f"\nInterpretation: {interp}")

    return results


# ============================================================================
# Q3: Head output vs ground-truth trajectory
# ============================================================================

def run_Q3_head_vs_gt(predictor, dataset, num_objects=5):
    """For each probe task, compare head mean output vs ground-truth trajectory mean."""
    print("\n" + "=" * 70)
    print("Q3: Head output vs ground-truth trajectory (CFM training convergence?)")
    print("=" * 70)

    K = len(PROBE_TASK_NAMES)
    head_mean = np.zeros((K, 5, 8, 6), dtype=np.float64)
    gt_mean = np.zeros((K, 5, 8, 6), dtype=np.float64)

    for ti, task_name in enumerate(PROBE_TASK_NAMES):
        sample_indices = get_sample_indices_for_task(
            dataset, task_name, n_samples=N_DEMOS_PER_TASK_FOR_GT, frame_idx_filter=0
        )
        if len(sample_indices) == 0:
            print(f"  task {ti}: no samples found, skipping")
            continue

        # Collect head outputs (N_SEEDS) and gt_trajectories (across demos)
        head_samples = []
        gt_samples = []
        # Use first demo for head probe (consistent with Test 1/2 protocol)
        batch_first = get_batch_from_sample_idx(
            dataset, sample_indices[0], predictor.device, predictor.amp_dtype, num_objects
        )
        for si in range(N_SEEDS):
            traj = predictor.head_forward_only(batch_first, seed=si)
            head_samples.append(traj[0].cpu().numpy())  # (5, 8, 6)
        head_mean[ti] = np.mean(head_samples, axis=0)

        # GT trajectory: average over all sampled demos
        for idx in sample_indices:
            sample = dataset[idx]
            gt_samples.append(sample["gt_trajectory"])  # (5, 8, 6) numpy
        gt_mean[ti] = np.mean(gt_samples, axis=0)

        # Distance: per-(obj, t) L2
        dist = np.linalg.norm(head_mean[ti] - gt_mean[ti], axis=-1)  # (5, 8)
        print(f"  task {ti} ({task_name[:50]}...):")
        print(f"    head μ amplitude:  range [{head_mean[ti].min():+.3f}, {head_mean[ti].max():+.3f}], "
              f"mean abs {np.abs(head_mean[ti]).mean():.3f}")
        print(f"    gt amplitude:      range [{gt_mean[ti].min():+.3f}, {gt_mean[ti].max():+.3f}], "
              f"mean abs {np.abs(gt_mean[ti]).mean():.3f}")
        print(f"    L2 dist (obj=0, t=0..4): {dist[0, :5].tolist()}")
        print(f"    L2 dist mean (all obj/t): {dist.mean():.3f}, max {dist.max():.3f}")

    # Save raw arrays
    np.save(OUT_DIR / "deep_dive_head_mean.npy", head_mean)
    np.save(OUT_DIR / "deep_dive_gt_mean.npy", gt_mean)

    # Summary
    overall_dist = np.linalg.norm(head_mean - gt_mean, axis=-1)  # (K, 5, 8)
    target_dist = overall_dist[:, 0, :]  # focus on obj=0 (target object)

    head_amp = np.abs(head_mean).mean()
    gt_amp = np.abs(gt_mean).mean()
    amp_ratio = head_amp / (gt_amp + 1e-12)

    print(f"\nSummary (all tasks):")
    print(f"  head mean abs amplitude: {head_amp:.3f}")
    print(f"  gt mean abs amplitude:   {gt_amp:.3f}")
    print(f"  amplitude ratio head/gt: {amp_ratio:.3f}")
    print(f"  L2 dist target (obj=0): mean={target_dist.mean():.3f}, max={target_dist.max():.3f}")
    print(f"  L2 dist overall:        mean={overall_dist.mean():.3f}, max={overall_dist.max():.3f}")

    # Verdict
    if overall_dist.mean() < 0.2 and amp_ratio > 0.5 and amp_ratio < 2.0:
        verdict = "head_converged_to_gt"
    elif amp_ratio < 0.3 or amp_ratio > 3.0:
        verdict = "amplitude_mismatch_head_far_from_gt"
    else:
        verdict = "head_partially_converged"

    print(f"\nVerdict: {verdict}")
    return {
        "head_amp_mean_abs": float(head_amp),
        "gt_amp_mean_abs": float(gt_amp),
        "amplitude_ratio_head_over_gt": float(amp_ratio),
        "L2_dist_target_mean": float(target_dist.mean()),
        "L2_dist_target_max": float(target_dist.max()),
        "L2_dist_overall_mean": float(overall_dist.mean()),
        "L2_dist_overall_max": float(overall_dist.max()),
        "verdict": verdict,
    }


# ============================================================================
# Q4: Per-dim amplitude distribution
# ============================================================================

def run_Q4_amplitude_distribution(predictor, dataset, num_objects=5):
    """Per-dim distribution of head output vs gt_trajectory."""
    print("\n" + "=" * 70)
    print("Q4: Per-dim amplitude distribution (head vs gt_trajectory)")
    print("=" * 70)

    # Sample broadly across all probe tasks
    head_vals = {d: [] for d in range(6)}
    gt_vals = {d: [] for d in range(6)}

    for ti, task_name in enumerate(PROBE_TASK_NAMES):
        sample_indices = get_sample_indices_for_task(dataset, task_name, n_samples=5, frame_idx_filter=0)
        if not sample_indices:
            continue
        # Head output via one batch + 5 seeds
        batch = get_batch_from_sample_idx(
            dataset, sample_indices[0], predictor.device, predictor.amp_dtype, num_objects
        )
        for si in range(5):
            traj = predictor.head_forward_only(batch, seed=si)
            arr = traj[0].cpu().numpy().reshape(-1, 6)  # (5*8, 6)
            for d in range(6):
                head_vals[d].extend(arr[:, d].tolist())

        # GT trajectory
        for idx in sample_indices:
            sample = dataset[idx]
            arr = sample["gt_trajectory"].reshape(-1, 6)
            for d in range(6):
                gt_vals[d].extend(arr[:, d].tolist())

    summary = {}
    print(f"\n{'Dim':<5}{'Head mean':>12}{'Head std':>12}{'Head range':>22}"
          f"{'GT mean':>12}{'GT std':>12}{'GT range':>22}")
    print("-" * 100)
    for d in range(6):
        h = np.array(head_vals[d])
        g = np.array(gt_vals[d])
        summary[f"dim_{d}"] = {
            "head_mean": float(h.mean()), "head_std": float(h.std()),
            "head_min": float(h.min()),   "head_max": float(h.max()),
            "gt_mean":   float(g.mean()), "gt_std":   float(g.std()),
            "gt_min":    float(g.min()),  "gt_max":   float(g.max()),
            "std_ratio_h_over_g": float(h.std() / (g.std() + 1e-12)),
        }
        h_range_str = f"[{h.min():+.2f}, {h.max():+.2f}]"
        g_range_str = f"[{g.min():+.2f}, {g.max():+.2f}]"
        print(f"{d:<5}{h.mean():>+12.3f}{h.std():>12.3f}{h_range_str:>22}"
              f"{g.mean():>+12.3f}{g.std():>12.3f}{g_range_str:>22}")

    # Interpretation
    dims_with_amp_issue = [d for d in range(6)
                          if summary[f"dim_{d}"]["std_ratio_h_over_g"] < 0.3
                          or summary[f"dim_{d}"]["std_ratio_h_over_g"] > 3.0]
    print(f"\nDims with amplitude mismatch (head_std / gt_std < 0.3 or > 3.0): {dims_with_amp_issue}")
    summary["dims_with_amp_issue"] = dims_with_amp_issue
    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Phase 0 Deep Dive: Q1 + Q2 + Q3 + Q4")
    print("=" * 70)

    # Setup
    cfg = load_config()
    dataset = build_dataset(cfg)
    predictor = OTPSoftPredictor(
        ckpt_path=PAPER_CKPT_FILE.read_text().strip(),
        config_path=str(CONFIG_PATH),
        grasp_affordance_dir=str(Path(cfg.data.grasp_affordance_dir)),
        device=torch.device("cuda"),
        log_diagnostics=False,
    )
    predictor.reset(episode_seed=0)

    # Load Test 2 outputs (re-use μ_pos and P)
    print("\nLoading Test 2 outputs for Q1...")
    mu = np.load(OUT_DIR / "test2_mu_pos.npy")
    P = np.load(OUT_DIR / "test2_P.npy")
    print(f"  μ shape: {mu.shape}, P shape: {P.shape}")

    # Q1: Leverage
    q1_results = run_Q1_leverage_analysis(P, mu)

    # Q2: needs per-task mean z slice (obj=0, t=0..4)
    # First load Test 1 Z if available, else recompute (5 task only)
    # We need ALL 10 tasks for proper Mantel, so collect fresh for the 10 tasks.
    print("\n" + "=" * 70)
    print("Collecting head z output for all 10 tasks (for Q2 Mantel test)...")
    print("=" * 70)
    K = len(PROBE_TASK_NAMES)
    z_obj0_per_task = np.zeros((K, 30), dtype=np.float64)  # obj=0, t=0..4, 6 dims = 30
    for ti, task_name in enumerate(PROBE_TASK_NAMES):
        sample_indices = get_sample_indices_for_task(dataset, task_name, n_samples=1, frame_idx_filter=0)
        if not sample_indices:
            continue
        batch = get_batch_from_sample_idx(
            dataset, sample_indices[0], predictor.device, predictor.amp_dtype, cfg.model.otp_head.num_objects
        )
        z_seed_samples = []
        for si in range(N_SEEDS):
            traj = predictor.head_forward_only(batch, seed=si)
            z_seed_samples.append(traj[0, 0, :5, :].cpu().numpy().flatten())  # (30,)
        z_obj0_per_task[ti] = np.mean(z_seed_samples, axis=0)
        print(f"  task {ti}: z[obj=0, t=0..4].mean shape={z_obj0_per_task[ti].shape}")

    q2_results = run_Q2_geometry_mantel(z_obj0_per_task, P)

    # Q3: head vs gt_trajectory
    q3_results = run_Q3_head_vs_gt(predictor, dataset, num_objects=cfg.model.otp_head.num_objects)

    # Q4: amplitude distribution
    q4_results = run_Q4_amplitude_distribution(predictor, dataset, num_objects=cfg.model.otp_head.num_objects)

    # Save all
    summary = {
        "version": "deep_dive_v1",
        "n_tasks": K,
        "n_seeds_per_task": N_SEEDS,
        "n_demos_per_task_for_gt": N_DEMOS_PER_TASK_FOR_GT,
        "Q1_leverage": q1_results,
        "Q2_geometry_mantel": q2_results,
        "Q3_head_vs_gt": q3_results,
        "Q4_amplitude": q4_results,
    }
    out_path = OUT_DIR / "deep_dive.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL DONE. Summary written to", out_path)
    print("=" * 70)
    print(f"  Q1: highest-leverage task = {q1_results['leverage_ranking'][0]} "
          f"(Δ R² = {q1_results['leave_one_out'][q1_results['leverage_ranking'][0]]['delta_R2']:+.3f})")
    print(f"  Q2: Mantel r = {q2_results['mantel_r']:+.4f}, p = {q2_results['mantel_p']:.4f}")
    print(f"      → {q2_results['interpretation']}")
    print(f"  Q3: head vs gt L2 (target obj=0) mean = {q3_results['L2_dist_target_mean']:.3f}; "
          f"amp ratio = {q3_results['amplitude_ratio_head_over_gt']:.3f}")
    print(f"      → {q3_results['verdict']}")
    print(f"  Q4: dims with amplitude mismatch = {q4_results['dims_with_amp_issue']}")


if __name__ == "__main__":
    main()
