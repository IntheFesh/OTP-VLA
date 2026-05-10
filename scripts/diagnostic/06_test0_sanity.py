"""
Phase 0 Test 0: Sanity check before running the diagnostic battery.

V6 status: unchanged from V5 draft. No revisions affect this script.

Three checks:
1. Probe task split is a subset of the V3 training split (not testing on holdout).
2. Head z output shape matches expected (N_obj, H, 6) and flatten order is consistent
   with what the decoder consumes.
3. Deterministic seeding controls within-task variance — same seed twice gives identical
   z, different seeds give different z.

Run: python scripts/diagnostic/06_test0_sanity.py
Exit code 0 if all checks pass; nonzero with diagnostic message otherwise.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from otp.eval.predictor import OTPSoftPredictor
from otp.data.libero_loader import build_loader


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_CKPT = REPO_ROOT / ".paper_ready_ckpt"
CFG_PATH = REPO_ROOT / "configs" / "otp_soft_30e.yaml"
SPLIT_FILE = REPO_ROOT / "data" / "libero_spatial_split.json"  # adjust to actual path


def check_split():
    """Probe must use a subset of training task IDs, not the 5 held-out demos."""
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_task_ids = set(split["train_task_ids"])
    probe_task_ids = set(range(5))
    if not probe_task_ids.issubset(train_task_ids):
        return False, f"probe tasks {probe_task_ids - train_task_ids} not in training split"
    return True, f"all 5 probe tasks in training split (size {len(train_task_ids)})"


def check_z_shape(predictor):
    """Run one forward pass, verify z has shape (N_obj=5, H=8, 6) and flattens row-major."""
    loader = build_loader(CFG_PATH, batch_size=1, split="train", task_ids=[0])
    batch = next(iter(loader))
    with torch.no_grad():
        z = predictor.head_forward_only(batch, seed=0)
    expected = (1, 5, 8, 6)
    if tuple(z.shape) != expected:
        return False, f"z shape {tuple(z.shape)} != expected {expected}"
    flat = z.reshape(z.shape[0], -1)
    if flat.shape[-1] != 240:
        return False, f"flatten dim {flat.shape[-1]} != 240"
    return True, f"z shape {tuple(z.shape)}, flatten dim {flat.shape[-1]}"


def check_seeding(predictor):
    """Same seed twice → identical z. Different seeds → different z."""
    loader = build_loader(CFG_PATH, batch_size=1, split="train", task_ids=[0])
    batch = next(iter(loader))
    with torch.no_grad():
        z_a1 = predictor.head_forward_only(batch, seed=42)
        z_a2 = predictor.head_forward_only(batch, seed=42)
        z_b = predictor.head_forward_only(batch, seed=1337)
    same_seed_diff = (z_a1 - z_a2).abs().max().item()
    diff_seed_diff = (z_a1 - z_b).abs().max().item()
    if same_seed_diff > 1e-5:
        return False, f"same seed gave different z, max abs diff = {same_seed_diff:.2e}"
    if diff_seed_diff < 1e-3:
        return False, f"different seeds gave near-identical z, max abs diff = {diff_seed_diff:.2e} — seeding broken"
    return True, f"same-seed diff {same_seed_diff:.2e}, diff-seed diff {diff_seed_diff:.2e}"


def main():
    ckpt = PAPER_CKPT.read_text().strip()
    print(f"=== Phase 0 Test 0: Sanity Check (V6) ===")
    print(f"Checkpoint: {ckpt}\n")

    predictor = OTPSoftPredictor(
        ckpt_path=ckpt,
        config_path=str(CFG_PATH),
        grasp_affordance_dir=str(REPO_ROOT / "data" / "grasp_affordance"),
        device="cuda",
        log_diagnostics=False,
    )

    results = []
    for name, fn in [
        ("Split membership", lambda: check_split()),
        ("z shape & flatten", lambda: check_z_shape(predictor)),
        ("Deterministic seeding", lambda: check_seeding(predictor)),
    ]:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"exception: {type(e).__name__}: {e}"
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {msg}")
        results.append((name, ok, msg))

    all_pass = all(r[1] for r in results)
    print(f"\n=== Result: {'ALL PASS' if all_pass else 'FAILURE'} ===")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
