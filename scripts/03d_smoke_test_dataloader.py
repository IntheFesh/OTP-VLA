#!/usr/bin/env python
"""
scripts/03d_smoke_test_dataloader.py

Smoke-test the LIBEROOTPDataset end-to-end:
  1. Build the dataset (requires extracted demos + affordance cache).
  2. Iterate the first 5 samples.
  3. Print every key's shape and dtype.
  4. Verify grasp_affordance is not all-zero.
  5. Verify object_point_clouds lie within a reasonable bounding box
     (|x|, |y|, |z| < 5 m).

Usage:
    python scripts/03d_smoke_test_dataloader.py \\
        --root data/object_poses/ \\
        --suite libero_spatial \\
        --grasp-affordance-dir data/grasp_affordances/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(message)s",
)
logger = logging.getLogger("smoke_test")


def _print_startup(args: argparse.Namespace) -> None:
    print("=" * 72)
    print(f"[ENV]  python={sys.version.split()[0]}  cwd={os.getcwd()}")
    print(f"[CFG]  suite={args.suite}  num-samples={args.num_samples}")
    print(f"[DATA] root={args.root}")
    print(f"[CACHE] grasp_affordance_dir={args.grasp_affordance_dir}")
    print("=" * 72)


def _shape_str(x) -> str:
    if isinstance(x, np.ndarray):
        return f"shape={tuple(x.shape)}  dtype={x.dtype}"
    if isinstance(x, list):
        return f"list[len={len(x)}]"
    return f"{type(x).__name__}({x!r})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory containing extracted .hdf5 demos")
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--grasp-affordance-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()
    _print_startup(args)

    from otp.data.libero_loader import LIBEROOTPDataset

    try:
        ds = LIBEROOTPDataset(
            root=args.root,
            suite=args.suite,
            grasp_affordance_dir=args.grasp_affordance_dir,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("dataset construction failed: %s", e)
        sys.exit(1)

    logger.info("dataset built: len=%d", len(ds))

    # Iterate first N samples.
    n = min(args.num_samples, len(ds))
    bad_pcd_count = 0
    bad_aff_count = 0

    for i in range(n):
        sample = ds[i]
        print(f"\n--- sample {i} ---")
        for k, v in sample.items():
            print(f"  {k:25s}  {_shape_str(v)}")

        aff = sample["grasp_affordance"]
        if np.all(aff == 0):
            logger.warning("sample %d: grasp_affordance is ALL-ZERO", i)
            bad_aff_count += 1

        pcd = sample["object_point_clouds"]
        max_abs = float(np.abs(pcd).max())
        if max_abs >= 5.0:
            logger.warning("sample %d: point cloud out of bbox (max=%.3f m)",
                           i, max_abs)
            bad_pcd_count += 1

    print()
    print(f"Smoke test summary:")
    print(f"  samples checked      : {n}")
    print(f"  all-zero affordances : {bad_aff_count}")
    print(f"  oversized pcd        : {bad_pcd_count}")
    if bad_pcd_count or bad_aff_count:
        sys.exit(1)
    print(f"  [OK] dataloader smoke test passed")


if __name__ == "__main__":
    main()
