#!/usr/bin/env python
"""
scripts/03_extract_object_poses.py

Extract per-frame object SE(3) poses from LIBERO demo HDF5 files and write them
to a structured per-demo .npz file under <output_dir>/<task_id>/<demo_id>.npz.

Usage:
    python scripts/03_extract_object_poses.py \\
        --suite libero_spatial \\
        --num-demos 50 \\
        --demos-root third_party/LIBERO/datasets \\
        --output data/object_poses/

Each .npz file contains:
    object_names: array of strings (N_obj,)
    object_poses: float32 (N_obj, T, 4, 4)
    actions:      float32 (T, 7)
    instruction:  str

Startup snapshot (§1.2): prints [ENV]/[CFG]/[CKPT]/[DATA] tags before any work.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(name)s | %(message)s",
)
logger = logging.getLogger("extract_object_poses")


def _print_startup(args: argparse.Namespace) -> None:
    """§1.2 startup snapshot."""
    print("=" * 72)
    print(f"[ENV]  python={sys.version.split()[0]}  cwd={os.getcwd()}")
    print(f"[CFG]  suite={args.suite}  num_demos={args.num_demos}")
    print(f"[DATA] demos_root={args.demos_root}")
    print(f"[OUT]  output_dir={args.output}")
    print(f"[CKPT] (no checkpoint required)")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=str, default="libero_spatial",
                        help="LIBERO benchmark suite name")
    parser.add_argument("--num-demos", type=int, default=50,
                        help="Maximum number of demos to extract per task")
    parser.add_argument("--demos-root", type=Path,
                        default=Path("third_party/LIBERO/datasets"),
                        help="Root directory containing LIBERO HDF5 demos")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for extracted .npz files")
    args = parser.parse_args()

    _print_startup(args)

    from otp.data.object_pose_extractor import ObjectPoseExtractor

    suite_dir = args.demos_root / args.suite
    if not suite_dir.exists():
        logger.error("Suite directory not found: %s", suite_dir)
        logger.error("Hint: ensure LIBERO is cloned and demos downloaded.")
        sys.exit(1)

    demo_paths = sorted(suite_dir.glob("**/*.hdf5"))[: args.num_demos]
    if not demo_paths:
        logger.error("No .hdf5 demos in %s", suite_dir)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    extractor = ObjectPoseExtractor(suite=args.suite)

    summary = {"num_demos": len(demo_paths), "extracted": 0, "failed": []}
    t0 = time.time()

    for i, demo_path in enumerate(demo_paths):
        try:
            data = extractor.extract(demo_path)
        except Exception as e:
            logger.warning("Failed to extract %s: %s", demo_path.name, e)
            summary["failed"].append(str(demo_path.name))
            continue

        out_path = args.output / f"{demo_path.stem}.npz"
        np.savez(
            out_path,
            object_names=np.asarray(data["object_names"]),
            object_poses=data["object_poses"],
            actions=data["actions"],
            instruction=data["instruction"],
        )
        summary["extracted"] += 1
        if (i + 1) % 10 == 0:
            logger.info("  progress: %d / %d", i + 1, len(demo_paths))

    summary["elapsed_sec"] = round(time.time() - t0, 2)
    summary_path = args.output / "extract_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done. Wrote %d / %d demos in %.1fs",
                summary["extracted"], summary["num_demos"], summary["elapsed_sec"])
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
