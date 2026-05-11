"""
Phase 0 Test 2: Spatial alignment (V6.1).

V6 frozen items (per notes/2026-05-11_phase0_preregistration_v6.md):
- K = 10 LIBERO-Spatial tasks (doubled from V3's 5)
- N = 20 seeds per task, mean to μ_t^pos ∈ R^3
- Target object world position p_t from npz["object_poses"][0, 0, :3, 3]
  (obj 0 = target black bowl, frame 0, translation part of SE(3))
- Affine regression μ = A p + b, R² + 1000-shuffle permutation p

V6 frozen verdict (M2: permutation p only):
  p < 0.01:        head_spatial_aware (problem in decoder)
  p ∈ [0.01, 0.05]: borderline (joint with Test 1, document rationale)
  p > 0.05:        head_no_spatial_structure

R² reported as effect size, NOT in verdict logic.

V6.1 interface fixes:
- Use LIBEROOTPDataset + collate_fn + assemble_batch
- Use predictor.head_forward_only
- Target world position extracted from npz file directly (not from a non-existent helper)

Output: results/phase0/test2_spatial_alignment.json
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
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

# FROZEN per V6 pre-registration §2.4
K_TASKS = 10
N_SEEDS = 20
N_PERM = 1000
ALPHA_STRONG = 0.01
ALPHA_BORDERLINE_UPPER = 0.05

# All 10 LIBERO-Spatial task directories (sorted; matches `ls` order)
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


def get_first_sample_idx_for_task(dataset, task_name):
    """Find the index of the first sample (lowest demo, frame 0 preferred) for given task."""
    for i in range(len(dataset)):
        demo_idx, frame_idx = dataset._index[i]
        record = dataset.demo_records[demo_idx]
        if record["npz_path"].parent.name == task_name and frame_idx == 0:
            return i, record["npz_path"]
    # Fallback: first matching sample regardless of frame
    for i in range(len(dataset)):
        demo_idx, _ = dataset._index[i]
        record = dataset.demo_records[demo_idx]
        if record["npz_path"].parent.name == task_name:
            return i, record["npz_path"]
    raise RuntimeError(f"No samples for task '{task_name}'")


def extract_target_world_position(npz_path):
    """Read target object (obj 0) world position at frame 0 from npz.

    npz['object_poses'] shape: (N_obj, T, 4, 4). target = obj 0, frame 0,
    translation = [:3, 3].
    """
    npz = np.load(npz_path, allow_pickle=False)
    object_poses = npz["object_poses"]  # (N_obj, T, 4, 4)
    target_pos = object_poses[0, 0, :3, 3]  # (3,)
    npz.close()
    return np.asarray(target_pos, dtype=np.float64)


def get_batch_for_sample(dataset, sample_idx, device, amp_dtype, num_objects=5):
    sample = dataset[sample_idx]
    raw_batch = collate_fn([sample])
    return assemble_batch(raw_batch, device, amp_dtype, num_objects)


def collect_mu_pos(predictor, dataset, task_names, n_seeds, num_objects=5):
    """For each task, compute mean over n_seeds of trajectory[obj=0, t=0, :3].

    Returns:
        mu: (K, 3)  — mean head spatial output for target object at horizon-step 0
        P:  (K, 3)  — target object world position
    """
    K = len(task_names)
    mu = np.zeros((K, 3), dtype=np.float64)
    P = np.zeros((K, 3), dtype=np.float64)
    for ti, name in enumerate(task_names):
        sample_idx, npz_path = get_first_sample_idx_for_task(dataset, name)
        P[ti] = extract_target_world_position(npz_path)
        batch = get_batch_for_sample(
            dataset, sample_idx, predictor.device, predictor.amp_dtype, num_objects
        )
        # Collect z spatial slice [obj=0, t=0, :3] across seeds, then average
        spatial_samples = []
        for si in range(n_seeds):
            traj = predictor.head_forward_only(batch, seed=si)  # (1, 5, 8, 6)
            spatial_samples.append(traj[0, 0, 0, :3].cpu().numpy())
        mu[ti] = np.mean(spatial_samples, axis=0)
        print(f"  task {ti} ({name[:50]}...):")
        print(f"    target pos = {P[ti]}")
        print(f"    μ pos       = {mu[ti]}")
    return mu, P


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
    """Shuffle P rows, recompute R². Return p-value = fraction of null R² ≥ observed."""
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
    print("=== Phase 0 Test 2: Spatial Alignment (V6.1) ===\n")
    ckpt = PAPER_CKPT_FILE.read_text().strip()
    print(f"Checkpoint: {ckpt}\n")

    cfg = load_resolved_config()
    dataset = build_train_dataset(cfg)
    print(f"Dataset: {len(dataset)} samples\n")

    predictor = OTPSoftPredictor(
        ckpt_path=ckpt,
        config_path=str(CONFIG_PATH),
        grasp_affordance_dir=str(Path(cfg.data.grasp_affordance_dir)),
        device=torch.device("cuda"),
        log_diagnostics=False,
    )
    predictor.reset(episode_seed=0)

    print(f"Collecting μ_pos and P for {K_TASKS} tasks × {N_SEEDS} seeds...\n")
    mu, P = collect_mu_pos(
        predictor, dataset, PROBE_TASK_NAMES, N_SEEDS,
        num_objects=cfg.model.otp_head.num_objects,
    )
    np.save(OUT_DIR / "test2_mu_pos.npy", mu)
    np.save(OUT_DIR / "test2_P.npy", P)

    print(f"\nP range:    x [{P[:,0].min():.3f}, {P[:,0].max():.3f}]  "
          f"y [{P[:,1].min():.3f}, {P[:,1].max():.3f}]  "
          f"z [{P[:,2].min():.3f}, {P[:,2].max():.3f}]")
    print(f"μ range:    x [{mu[:,0].min():.3f}, {mu[:,0].max():.3f}]  "
          f"y [{mu[:,1].min():.3f}, {mu[:,1].max():.3f}]  "
          f"z [{mu[:,2].min():.3f}, {mu[:,2].max():.3f}]")

    print(f"\nRunning affine regression + {N_PERM}-shuffle permutation test...")
    R2_obs, pval, R2_null = permutation_test(P, mu)
    print(f"  R² observed:    {R2_obs:.4f}  (reported as effect size, not in verdict)")
    print(f"  Permutation p:  {pval:.4f}")
    print(f"  Null R² mean:   {R2_null.mean():.3f} ± {R2_null.std():.3f}")
    print(f"  Null R² p95:    {np.percentile(R2_null, 95):.3f}")

    # M2: Verdict uses permutation p only
    if pval < ALPHA_STRONG:
        verdict = "head_spatial_aware"
    elif pval <= ALPHA_BORDERLINE_UPPER:
        verdict = "borderline_joint_with_test1"
    else:
        verdict = "head_no_spatial_structure"

    summary = {
        "checkpoint": ckpt,
        "version": "V6.1",
        "n_tasks": K_TASKS, "n_seeds": N_SEEDS, "n_perm": N_PERM,
        "alpha_strong": ALPHA_STRONG, "alpha_borderline_upper": ALPHA_BORDERLINE_UPPER,
        "R2_observed": float(R2_obs),
        "R2_null_mean": float(R2_null.mean()),
        "R2_null_std": float(R2_null.std()),
        "R2_null_p95": float(np.percentile(R2_null, 95)),
        "permutation_pval": float(pval),
        "verdict": verdict,
        "verdict_logic": "M2: permutation p only; R² is effect size",
        "probe_task_names": PROBE_TASK_NAMES,
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
