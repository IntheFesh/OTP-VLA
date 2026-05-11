"""
Phase 0 Test 0: Sanity check (V6.1 — real interface).

Three checks (revised for actual repo API):
1. Dataset can be instantiated with train_end_demo=45 (V3 train split), and probe
   target task name is in the dataset's task list.
2. predictor.head_forward_only(batch, seed) returns trajectory with shape
   (1, 5, 8, 6); flatten gives 240-dim.
3. Same seed twice → identical trajectory (max abs diff < 1e-5).
   Different seeds → different trajectory (max abs diff > 1e-3).

Run: python scripts/diagnostic/06_test0_sanity.py
Exit 0 if all pass; nonzero with diagnostic message.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from otp.train.utils import collate_fn, assemble_batch

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from otp.eval.predictor import OTPSoftPredictor
from otp.data.libero_loader import LIBEROOTPDataset
from otp.train.utils import collate_fn


PAPER_CKPT_FILE = REPO_ROOT / ".paper_ready_ckpt"
CONFIG_PATH = REPO_ROOT / "configs" / "otp_soft_30e.yaml"

# Test probes use task index 0 (first LIBERO-Spatial task)
PROBE_TASK_NAME = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"


def load_resolved_config():
    """Resolve hydra config (otp_soft_30e inherits from otp_soft_frozen)."""
    config_dir = str(REPO_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="otp_soft_30e")
    return cfg


def build_train_dataset(cfg):
    """Instantiate LIBEROOTPDataset with train_end_demo=45 (V3 train split)."""
    norm_path = getattr(cfg.data, "normalizer_path", None)
    dataset = LIBEROOTPDataset(
        root=Path(cfg.data.root),
        suite=cfg.data.suite,
        grasp_affordance_dir=Path(cfg.data.grasp_affordance_dir),
        normalizer_path=Path(norm_path) if norm_path else None,
        num_grasps_per_object=cfg.model.decoder.num_grasps_per_object,
        num_points=cfg.model.geometry_encoder.num_points,
        horizon=cfg.model.otp_head.horizon,
        train_end_demo=45,  # V3 training split — probe samples are guaranteed in-train
    )
    return dataset


def get_one_batch_from_task(dataset, task_name, device, amp_dtype, num_objects=5):
    """Return a forward-ready single-sample batch (B=1) from a specified task.
    
    Pipeline: dataset → collate_fn (stack) → assemble_batch (add object_indices,
    proprio; move to device). This matches the train loop's exact path.
    """
    target_indices = []
    for i in range(len(dataset)):
        demo_idx, _frame = dataset._index[i]
        record = dataset.demo_records[demo_idx]
        if record["npz_path"].parent.name == task_name:
            target_indices.append(i)
            if len(target_indices) >= 5:
                break
    if not target_indices:
        raise RuntimeError(f"No samples from task '{task_name}' in train split")

    sample = dataset[target_indices[0]]
    raw_batch = collate_fn([sample])  # B=1, post-stack
    batch = assemble_batch(raw_batch, device, amp_dtype, num_objects)
    return batch, len(target_indices)


def move_batch_to_device(batch, device, amp_dtype):
    """Move tensors to device, matching predictor's batch conventions."""
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                # Float tensors → amp_dtype where applicable
                if k in ("object_point_clouds", "proprioception", "grasp_affordance"):
                    moved[k] = v.to(device, amp_dtype)
                else:
                    moved[k] = v.to(device)
            else:
                moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def check_dataset_and_split(cfg, dataset):
    """Verify probe task is in dataset and dataset uses train_end_demo=45."""
    if dataset.train_end_demo != 45:
        return False, f"dataset.train_end_demo = {dataset.train_end_demo}, expected 45"
    
    # demo_records is per-(task, demo) — 10 task × 45 demo = 450 entries
    n_demo_records = len(dataset.demo_records)
    expected_n = 10 * 45  # LIBERO-Spatial 10 tasks, train_end_demo=45
    
    # Check probe task is present, and collect first 3 demo IDs for the probe task
    sample_demos = []
    for record in dataset.demo_records:
        if record["npz_path"].parent.name == PROBE_TASK_NAME:
            sample_demos.append(record["npz_path"].stem)
            if len(sample_demos) >= 3:
                break
    
    if not sample_demos:
        return False, f"probe task '{PROBE_TASK_NAME}' not in dataset.demo_records"
    
    return True, (
        f"train_end_demo=45 OK; demo_records={n_demo_records} "
        f"(expected ~{expected_n}); probe task present; "
        f"first 3 demos: {sample_demos}"
    )


def check_head_forward_output(predictor, dataset):
    """Run head_forward_only once; check shape (1, 5, 8, 6) and flatten = 240."""
    batch, _ = get_one_batch_from_task(
        dataset, PROBE_TASK_NAME, predictor.device, predictor.amp_dtype
    )
    traj = predictor.head_forward_only(batch, seed=0)
    expected = (1, 5, 8, 6)
    if tuple(traj.shape) != expected:
        return False, f"trajectory shape {tuple(traj.shape)} != expected {expected}"
    flat_dim = traj.reshape(traj.shape[0], -1).shape[-1]
    if flat_dim != 240:
        return False, f"flatten dim {flat_dim} != 240"
    return True, f"trajectory shape {tuple(traj.shape)}, flatten dim {flat_dim}"


def check_deterministic_seeding(predictor, dataset):
    """Same seed → identical traj. Different seeds → different traj."""
    batch, _ = get_one_batch_from_task(
        dataset, PROBE_TASK_NAME, predictor.device, predictor.amp_dtype
    )
    t_a1 = predictor.head_forward_only(batch, seed=42)
    t_a2 = predictor.head_forward_only(batch, seed=42)
    t_b = predictor.head_forward_only(batch, seed=1337)
    same_diff = (t_a1 - t_a2).abs().max().item()
    diff_diff = (t_a1 - t_b).abs().max().item()
    if same_diff > 1e-5:
        return False, f"same-seed gave different output, max abs diff = {same_diff:.2e}"
    if diff_diff < 1e-3:
        return False, (
            f"different seeds gave near-identical output, max abs diff = {diff_diff:.2e}; "
            "seeding might be broken"
        )
    return True, f"same-seed diff {same_diff:.2e}, diff-seed diff {diff_diff:.2e}"


def main():
    print("=== Phase 0 Test 0: Sanity Check (V6.1) ===\n")

    if not PAPER_CKPT_FILE.exists():
        print(f"[ERROR] {PAPER_CKPT_FILE} not found")
        sys.exit(1)
    ckpt = PAPER_CKPT_FILE.read_text().strip()
    print(f"Checkpoint: {ckpt}\n")

    print("Loading config (otp_soft_30e + otp_soft_frozen merge)...")
    cfg = load_resolved_config()
    print(f"  data.root           = {cfg.data.root}")
    print(f"  data.suite          = {cfg.data.suite}")
    print(f"  data.grasp_affdir   = {cfg.data.grasp_affordance_dir}")
    print(f"  model.head.horizon  = {cfg.model.otp_head.horizon}")
    print(f"  model.head.n_obj    = {cfg.model.otp_head.num_objects}")
    print()

    print("Building dataset (train_end_demo=45)...")
    dataset = build_train_dataset(cfg)
    print(f"  dataset size: {len(dataset)} samples\n")

    print("Building predictor...")
    predictor = OTPSoftPredictor(
        ckpt_path=ckpt,
        config_path=str(CONFIG_PATH),
        grasp_affordance_dir=str(Path(cfg.data.grasp_affordance_dir)),
        device=torch.device("cuda"),
        log_diagnostics=False,
    )
    # Predictor must be reset (it tracks episode_seed); for head-only probes we
    # don't actually use episode tracking, but call reset to satisfy any
    # internal invariants:
    predictor.reset(episode_seed=0)
    print(f"  predictor device: {predictor.device}, amp_dtype: {predictor.amp_dtype}\n")

    results = []
    for name, fn in [
        ("Dataset & split", lambda: check_dataset_and_split(cfg, dataset)),
        ("head_forward_only shape", lambda: check_head_forward_output(predictor, dataset)),
        ("Deterministic seeding", lambda: check_deterministic_seeding(predictor, dataset)),
    ]:
        try:
            ok, msg = fn()
        except Exception as e:
            import traceback
            ok = False
            msg = f"exception: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {msg}")
        results.append((name, ok, msg))

    all_pass = all(r[1] for r in results)
    print(f"\n=== Result: {'ALL PASS' if all_pass else 'FAILURE'} ===")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
