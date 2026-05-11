"""
M3 Trajectory Representation Robust Check.

Per V7-pre reviewer critique §一:
  The M3 finding r(demo_traj, language) = -0.27 used flattened trajectory
  representation. Reviewer requests robustness across 3 representations:

    R1: Flattened (current, 240-dim = 5 objects × 8 horizon × 6 dim)
    R2: Endpoint (30-dim = 5 objects × 6 dim, last horizon step only)
    R3: Waypoint subsample (90-dim = 5 objects × 3 waypoints {t=0,3,7} × 6 dim)

  Decision:
    If all 3 give r ≈ 0 (or large gap from r_h = 0.76): M3 robust, commit V7
    If endpoint shows r >> 0: demos ARE task-discriminative via endpoint geometry
                              (supports explanation A in §三)
    If any representation gives r >> 0: M3 framing needs revision

Output: results/phase0/m3_robust_check.json
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from otp.data.libero_loader import LIBEROOTPDataset

OUT_DIR = REPO_ROOT / "results" / "phase0"

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
N_DEMOS = 10
N_PERM = 1000


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


def get_demos_for_task(dataset, task_name, n_demos):
    matched = []
    for i in range(len(dataset)):
        demo_idx, frame_idx = dataset._index[i]
        record = dataset.demo_records[demo_idx]
        if record["npz_path"].parent.name == task_name and frame_idx == 0:
            matched.append(i)
            if len(matched) >= n_demos:
                break
    return matched


def mantel_test_two_sided(D1, D2, n_perm=N_PERM, seed=20260511):
    """Mantel test with one-sided and two-sided p-values."""
    K = D1.shape[0]
    iu = np.triu_indices(K, k=1)
    r_obs, _ = stats.pearsonr(D1[iu], D2[iu])
    rng = np.random.default_rng(seed)
    null_r = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(K)
        D2_perm = D2[perm][:, perm]
        null_r[i], _ = stats.pearsonr(D1[iu], D2_perm[iu])
    p_one_pos = float((null_r >= r_obs).mean())
    p_one_neg = float((null_r <= r_obs).mean())
    p_two = float((np.abs(null_r) >= abs(r_obs)).mean())
    return {
        "r": float(r_obs),
        "p_one_sided_positive": p_one_pos,
        "p_one_sided_negative": p_one_neg,
        "p_two_sided": p_two,
        "null_mean": float(null_r.mean()),
        "null_std": float(null_r.std()),
    }


def main():
    print("=" * 70)
    print("M3 Trajectory Representation Robust Check")
    print("=" * 70)

    cfg = load_config()
    dataset = build_dataset(cfg)

    # Collect 3 trajectory representations
    K = len(PROBE_TASK_NAMES)
    flat_reprs = np.zeros((K, 5 * 8 * 6), dtype=np.float64)        # 240-dim
    endpoint_reprs = np.zeros((K, 5 * 6), dtype=np.float64)         # 30-dim (last step)
    waypoint_reprs = np.zeros((K, 5 * 3 * 6), dtype=np.float64)    # 90-dim (t=0,3,7)

    print(f"\nCollecting trajectory representations ({N_DEMOS} demos per task)...")
    for ti, task_name in enumerate(PROBE_TASK_NAMES):
        demo_indices = get_demos_for_task(dataset, task_name, N_DEMOS)
        flat_samples = []
        endpoint_samples = []
        waypoint_samples = []
        for idx in demo_indices:
            sample = dataset[idx]
            traj = sample["gt_trajectory"]  # (5, 8, 6)
            flat_samples.append(traj.flatten())             # 240-dim
            endpoint_samples.append(traj[:, -1, :].flatten())  # 30-dim, last horizon step
            wp = traj[:, [0, 3, 7], :].flatten()             # 90-dim
            waypoint_samples.append(wp)
        flat_reprs[ti] = np.mean(flat_samples, axis=0)
        endpoint_reprs[ti] = np.mean(endpoint_samples, axis=0)
        waypoint_reprs[ti] = np.mean(waypoint_samples, axis=0)
        print(f"  task {ti}: flat L2={np.linalg.norm(flat_reprs[ti]):.2f}, "
              f"endpoint L2={np.linalg.norm(endpoint_reprs[ti]):.2f}, "
              f"waypoint L2={np.linalg.norm(waypoint_reprs[ti]):.2f}")

    # Load language_CLS from Gate 1
    lang_npy = OUT_DIR / "gate1_language_CLS.npy"
    if not lang_npy.exists():
        print(f"ERROR: {lang_npy} not found. Run Gate 1 first.")
        sys.exit(1)
    language_CLS = np.load(lang_npy)
    print(f"\nUsing language_CLS from Gate 1: shape={language_CLS.shape}, "
          f"D_lang mean = {pdist(language_CLS).mean():.4f}")

    # Pairwise distance matrices
    D_lang = squareform(pdist(language_CLS, metric="euclidean"))
    D_flat = squareform(pdist(flat_reprs, metric="euclidean"))
    D_endpoint = squareform(pdist(endpoint_reprs, metric="euclidean"))
    D_waypoint = squareform(pdist(waypoint_reprs, metric="euclidean"))

    iu = np.triu_indices(K, k=1)
    print(f"\nPairwise distance summary:")
    print(f"  D(flat trajectory):     mean={D_flat[iu].mean():.3f}")
    print(f"  D(endpoint trajectory): mean={D_endpoint[iu].mean():.3f}")
    print(f"  D(waypoint trajectory): mean={D_waypoint[iu].mean():.3f}")
    print(f"  D(language):            mean={D_lang[iu].mean():.4f}")

    print(f"\nMantel tests (each with one-sided + two-sided p-values):")
    print(f"  Reference Gate 1: r(h_OFT, language) = +0.7575, p_two ≈ 0.0000")
    print()

    results_per_repr = {}
    for repr_name, D_traj in [
        ("flat", D_flat),
        ("endpoint", D_endpoint),
        ("waypoint", D_waypoint),
    ]:
        m = mantel_test_two_sided(D_traj, D_lang)
        results_per_repr[repr_name] = m
        gap_from_r_h = 0.7575 - m["r"]
        print(f"  R={repr_name}:")
        print(f"    r(demo_traj, language) = {m['r']:+.4f}")
        print(f"    p_one_sided_positive   = {m['p_one_sided_positive']:.4f}  (test: r > random)")
        print(f"    p_one_sided_negative   = {m['p_one_sided_negative']:.4f}  (test: r < random)")
        print(f"    p_two_sided            = {m['p_two_sided']:.4f}")
        print(f"    null distribution:       mean={m['null_mean']:+.3f}, std={m['null_std']:.3f}")
        print(f"    Gap from r_h (Δr)      = {gap_from_r_h:+.4f}")
        print()

    # Verdict: M3 robust if all 3 r values have large gap (Δr > 0.5) from r_h
    deltas = {name: 0.7575 - m["r"] for name, m in results_per_repr.items()}
    min_delta = min(deltas.values())
    max_r = max(m["r"] for m in results_per_repr.values())

    print("=" * 70)
    print("Robustness verdict:")
    print("=" * 70)
    print(f"  All 3 representations: r values = "
          f"{ {n: f'{m[\"r\"]:+.3f}' for n, m in results_per_repr.items()} }")
    print(f"  All gaps from r_h (0.7575): {deltas}")
    print(f"  Min gap (Δr): {min_delta:.4f}")
    print(f"  Max r value across representations: {max_r:+.4f}")

    if min_delta > 0.5 and max_r < 0.4:
        verdict = "M3_ROBUST: all 3 trajectory representations show negligible language correlation"
        framing = ("M3 thesis holds: contrastive framing r_h=0.76 vs r_demo≈0 (regardless of "
                  "representation choice). Commit V7.")
    elif min_delta > 0.3 and max_r < 0.6:
        verdict = "M3_QUALIFIED: some representations show partial language correlation"
        framing = ("M3 thesis qualified: demos carry weak language signal in some "
                  "representations. Need to acknowledge this nuance in paper §V.B.")
    else:
        verdict = "M3_FAILED: some representation gives r >> 0"
        framing = ("M3 thesis collapses: at least one representation shows demos ARE "
                  "task-discriminative w.r.t. language. Need to reframe.")

    print(f"\n  Verdict: {verdict}")
    print(f"  Framing implication: {framing}")

    out_path = OUT_DIR / "m3_robust_check.json"
    with open(out_path, "w") as f:
        json.dump({
            "version": "m3_robust_v1",
            "per_representation": results_per_repr,
            "deltas_from_r_h": deltas,
            "r_h_reference": 0.7575,
            "verdict": verdict,
            "framing_implication": framing,
        }, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
