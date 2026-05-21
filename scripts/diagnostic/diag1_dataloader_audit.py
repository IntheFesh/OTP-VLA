"""
Diagnostic 1: Dataloader Action Audit
======================================

Verify whether the training dataloader applies any undeclared transformation
to actions before they reach the loss computation.

Goal: rule out the simplest possible bug — "dataloader normalizes actions at
training time but predict_chunk doesn't denormalize at eval time".

Hypothesis to test:
  H_null: training_actions == hdf5_raw_actions (byte-identical)
  H_alt:  training_actions = f(hdf5_raw_actions) for some f ≠ identity

Constraint: CPU only, no model training, no source-file modifications.

Author: Day 9 diagnostic
Date: 2026-05-21
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, '/root/autodl-tmp/OTP-VLA')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'

import json
import numpy as np
import h5py


def header(s: str):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


# -----------------------------------------------------------------------
# Step 1: Find pathB seed 42 training config
# -----------------------------------------------------------------------
header("STEP 1: Locate pathB seed 42 training config")

repo_root = Path("/root/autodl-tmp/OTP-VLA")
config_candidates = [
    repo_root / "configs" / "otp_soft_pathB_seed42.yaml",
    repo_root / "configs" / "path_b" / "otp_soft_pathB_seed42.yaml",
]

config_path = None
for c in config_candidates:
    if c.exists():
        config_path = c
        break

if config_path is None:
    # Fallback: search
    found = list(repo_root.rglob("*pathB_seed42*.yaml"))
    if found:
        config_path = found[0]
    else:
        print("ERROR: Cannot find pathB seed 42 config")
        sys.exit(1)

print(f"Config: {config_path}")
print()
print(config_path.read_text())


# -----------------------------------------------------------------------
# Step 2: Read normalizer stats explicitly
# -----------------------------------------------------------------------
header("STEP 2: Read normalizer stats")

normalizer_path = repo_root / "data" / "object_poses" / "meta" / "stats_qpace.json"
if not normalizer_path.exists():
    print(f"NOT FOUND: {normalizer_path}")
    print("Searching for alternatives...")
    alts = list(repo_root.rglob("stats_qpace.json"))
    print(f"Found: {alts}")
    if alts:
        normalizer_path = alts[0]

print(f"Normalizer file: {normalizer_path}")
with open(normalizer_path) as f:
    norm_stats = json.load(f)
print(f"Contents: {json.dumps(norm_stats, indent=2)}")

# Detect if it's identity
am = np.array(norm_stats.get("action_mean", [0]))
as_ = np.array(norm_stats.get("action_std", [1]))
is_identity = np.allclose(am, 0, atol=1e-6) and np.allclose(as_, 1, atol=1e-6)
print(f"\nIs normalizer identity? {is_identity}")
if not is_identity:
    print(f"  action_mean: {am}")
    print(f"  action_std:  {as_}")


# -----------------------------------------------------------------------
# Step 3: Instantiate dataset and pull idx=0 sample
# -----------------------------------------------------------------------
header("STEP 3: Instantiate dataset, take idx=0 sample")

# Use hydra to load config
try:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra().is_initialized():
        GlobalHydra().clear()

    config_dir_str = str(config_path.parent.resolve())
    config_name = config_path.stem  # without .yaml

    print(f"Hydra config dir: {config_dir_str}")
    print(f"Hydra config name: {config_name}")

    with initialize_config_dir(config_dir=config_dir_str, version_base=None):
        cfg = compose(config_name=config_name)

    print("Config loaded via Hydra.")
    print(f"  cfg.data.root: {cfg.data.root if hasattr(cfg, 'data') else 'N/A'}")
    print(f"  cfg.data.suite: {cfg.data.suite if hasattr(cfg.data, 'suite') else 'libero_spatial'}")
    print(f"  cfg.data.horizon: {cfg.data.horizon if hasattr(cfg.data, 'horizon') else 8}")
except Exception as e:
    print(f"Hydra load failed: {e}")
    print("Falling back to manual dataset instantiation...")
    cfg = None

# Instantiate dataset
from otp.data.libero_loader import LIBEROOTPDataset

ds_kwargs = dict(
    root=Path("data/object_poses"),
    suite="libero_spatial",
    libero_hdf5_root=Path("/root/autodl-tmp/libero_data/libero_spatial"),
    grasp_affordance_dir=Path("data/grasp_affordances"),
    horizon=8,
)

if cfg is not None:
    # Override from cfg if available
    if hasattr(cfg, "data"):
        for k in ["root", "suite", "horizon"]:
            if hasattr(cfg.data, k):
                v = getattr(cfg.data, k)
                if v is not None:
                    ds_kwargs[k] = Path(v) if k == "root" else v

print(f"\nDataset kwargs: {ds_kwargs}")

try:
    ds = LIBEROOTPDataset(**ds_kwargs)
    print(f"Dataset loaded: {len(ds)} samples")
except Exception as e:
    print(f"Dataset instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Get idx=0
sample = ds[0]
print(f"\nSample 0 keys: {list(sample.keys())}")
actions_train = sample["action_chunk"]
print(f"\naction_chunk:")
print(f"  shape:  {actions_train.shape}")
print(f"  dtype:  {actions_train.dtype}")
print(f"  min:    {actions_train.min():.6f}")
print(f"  max:    {actions_train.max():.6f}")
print(f"  per-dim mean: {actions_train.mean(axis=0)}")
print(f"\nactions[0] (training, 6 decimals):")
print(f"  {np.array2string(actions_train[0], precision=6, suppress_small=False)}")

# Get episode_id and frame_idx
demo_idx, frame_idx = ds._index[0]
record = ds.demo_records[demo_idx]
print(f"\nIndex 0 metadata:")
print(f"  demo_idx: {demo_idx}")
print(f"  frame_idx: {frame_idx}")
print(f"  npz_path: {record['npz_path']}")
print(f"  hdf5_path: {record.get('hdf5_path', 'N/A')}")
print(f"  demo_key (HDF5): {record.get('demo_key', 'N/A')}")


# -----------------------------------------------------------------------
# Step 4: Load corresponding HDF5 raw actions
# -----------------------------------------------------------------------
header("STEP 4: Load HDF5 raw actions for same episode/frame")

hdf5_path = record.get('hdf5_path')
demo_key = record.get('demo_key')

if hdf5_path is None or demo_key is None:
    # Infer from npz_path
    npz_path = Path(record['npz_path'])
    # NPZ filename like demo_0.npz → demo_0
    inferred_demo_key = npz_path.stem
    # HDF5 path inferred from suite + task name
    task_name = npz_path.parent.name  # parent dir = task name
    hdf5_root = Path("/root/autodl-tmp/libero_data/libero_spatial")
    hdf5_candidates = list(hdf5_root.glob(f"*{task_name}*demo.hdf5"))
    if hdf5_candidates:
        hdf5_path = hdf5_candidates[0]
        demo_key = inferred_demo_key
        print(f"Inferred hdf5_path: {hdf5_path}")
        print(f"Inferred demo_key: {demo_key}")
    else:
        print("Cannot infer HDF5 path. Searching...")
        sys.exit(1)

print(f"HDF5 path: {hdf5_path}")
print(f"Demo key: {demo_key}")
print(f"Frame range: [{frame_idx}, {frame_idx + actions_train.shape[0]}]")

with h5py.File(hdf5_path, "r") as f:
    available_keys = list(f["data"].keys())
    print(f"Available HDF5 demo keys (first 5): {available_keys[:5]}")
    
    if demo_key not in available_keys:
        # Try lookup by index
        if isinstance(demo_idx, int) and demo_idx < len(available_keys):
            demo_key_resolved = available_keys[demo_idx]
            print(f"WARNING: '{demo_key}' not in keys, using index lookup: '{demo_key_resolved}'")
            demo_key = demo_key_resolved
        else:
            demo_key = available_keys[0]
            print(f"WARNING: Using first key '{demo_key}'")
    
    actions_hdf5_full = np.array(f["data"][demo_key]["actions"])
    print(f"\nHDF5 actions shape (full): {actions_hdf5_full.shape}")
    
    # Same window as training sample
    actions_hdf5 = actions_hdf5_full[frame_idx:frame_idx + actions_train.shape[0]]
    print(f"HDF5 actions[{frame_idx}:{frame_idx + actions_train.shape[0]}] shape: {actions_hdf5.shape}")
    print(f"  min:    {actions_hdf5.min():.6f}")
    print(f"  max:    {actions_hdf5.max():.6f}")
    print(f"  per-dim mean: {actions_hdf5.mean(axis=0)}")

print(f"\nHDF5 actions[{frame_idx}] (raw, 6 decimals):")
print(f"  {np.array2string(actions_hdf5[0], precision=6, suppress_small=False)}")


# -----------------------------------------------------------------------
# Step 5: Compare with np.allclose(rtol=1e-4)
# -----------------------------------------------------------------------
header("STEP 5: Strict comparison")

if actions_train.shape != actions_hdf5.shape:
    print(f"SHAPE MISMATCH: train={actions_train.shape} vs hdf5={actions_hdf5.shape}")
    actions_hdf5 = actions_hdf5[:actions_train.shape[0]]
    print(f"  Truncated hdf5 to: {actions_hdf5.shape}")

# Cast to same dtype for comparison
at = np.asarray(actions_train, dtype=np.float32)
ah = np.asarray(actions_hdf5, dtype=np.float32)

match_strict = np.allclose(at, ah, rtol=1e-4, atol=1e-6)
print(f"\nnp.allclose(rtol=1e-4, atol=1e-6): {match_strict}")

# Element-wise diff
diff = at - ah
abs_diff = np.abs(diff)
print(f"\nElement-wise diff stats:")
print(f"  max abs diff: {abs_diff.max():.6f}")
print(f"  mean abs diff: {abs_diff.mean():.6f}")
print(f"  any nonzero: {(abs_diff > 1e-6).any()}")

# Per-dim diff
print(f"\nPer-dim max abs diff: {abs_diff.max(axis=0)}")

# Check for scale factor: ratio at/ah (where ah != 0)
print(f"\nElement-wise ratio at/ah analysis:")
mask = (np.abs(ah) > 1e-3)
if mask.any():
    ratios = at[mask] / ah[mask]
    print(f"  num nonzero hdf5 entries: {mask.sum()}")
    print(f"  ratio mean: {ratios.mean():.6f}")
    print(f"  ratio std:  {ratios.std():.6f}")
    print(f"  ratio min:  {ratios.min():.6f}")
    print(f"  ratio max:  {ratios.max():.6f}")
    constant_scale = ratios.std() < 0.01 * abs(ratios.mean())
    print(f"  Is constant scale factor? {constant_scale}")
    if constant_scale:
        print(f"  IMPLIED SCALE FACTOR: {ratios.mean():.4f}")

# Check sign flip on any dim
print(f"\nPer-dim sign analysis:")
for d in range(at.shape[1]):
    train_sign = np.sign(at[:, d])
    hdf5_sign = np.sign(ah[:, d])
    sign_match = (train_sign == hdf5_sign).mean()
    print(f"  dim {d}: sign_match_rate = {sign_match:.4f}")


# -----------------------------------------------------------------------
# Step 6: Verdict
# -----------------------------------------------------------------------
header("STEP 6: VERDICT")

if match_strict:
    print("✅ YES — training actions == HDF5 raw actions (byte-equal within rtol=1e-4)")
    print()
    print("CONCLUSION:")
    print("  The dataloader does NOT apply any undeclared transformation to actions.")
    print("  Training-time loss is computed against raw LIBERO actions.")
    print("  The 0% SR bug is NOT in the dataloader.")
    print()
    print("  Therefore the action-scale mismatch (~7x at inference) must originate")
    print("  in the MODEL or the INFERENCE/SAMPLING PATH:")
    print("    - CFM velocity field learned wrong distribution, OR")
    print("    - Predict_chunk has a postprocessing bug, OR")
    print("    - Trajectory representation feeding decoder is in a different coordinate frame")
else:
    print("❌ NO — training actions ≠ HDF5 raw actions")
    print()
    print("CONCLUSION:")
    print("  The dataloader is applying an undeclared transformation.")
    print()
    if mask.any() and ratios.std() < 0.01 * abs(ratios.mean()):
        print(f"  Detected transformation: action_train ≈ {ratios.mean():.4f} × action_hdf5")
        print(f"  Inverse needed at inference: predicted_action / {ratios.mean():.4f}")
    print()
    print("  ACTION ITEM:")
    print("  - Compare to predicted output range ([-0.72, 3.06] vs HDF5 [-1.0, 0.88])")
    print("  - If training-action range matches predicted range scale → bug confirmed")
    print("  - Fix: apply inverse transform in predict_chunk output (1-line fix)")


print()
print("=" * 70)
print("Diagnostic 1 complete.")
print("=" * 70)
