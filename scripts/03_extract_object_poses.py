#!/usr/bin/env python
"""
scripts/03_extract_object_poses.py

Extract per-frame object SE(3) poses from LIBERO demo HDF5 files via sim
replay, and write a structured per-demo .npz file.

LIBERO HDF5 layout: each task's hdf5 contains 50 demos (demo_0..demo_49).
We replay each one in sim to recover object poses, since the raw hdf5
does not store per-frame object poses (only full MuJoCo state).

Usage:
    python scripts/03_extract_object_poses.py \\
        --suite libero_spatial \\
        --demos-per-task 5 \\
        --demos-root /root/autodl-tmp/libero_data \\
        --output data/object_poses/

Output layout:
    <output>/<task_stem>/<demo_id>.npz
    <output>/extract_summary.json

Each .npz contains:
    object_names: array of strings (N_obj,)
    object_poses: float32 (N_obj, T, 4, 4)   — world-frame SE(3)
    actions:      float32 (T, 7)              — LIBERO OSC delta actions
    states:       float64 (T, 92)             — MuJoCo qpos+qvel
    ee_pos:       float32 (T, 3)
    ee_ori:       float32 (T, 3)              — axis-angle
    gripper_states: float32 (T, 2)
    instruction:  str
    bddl_file:    str
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
    print(f"[CFG]  suite={args.suite}  demos_per_task={args.demos_per_task}")
    print(f"[DATA] demos_root={args.demos_root}")
    print(f"[OUT]  output_dir={args.output}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--demos-per-task", type=int, default=5,
                        help="Number of demos to replay per task hdf5 file.")
    parser.add_argument("--demos-root", type=Path, required=True,
                        help="Directory containing <suite>/*.hdf5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _print_startup(args)

    from otp.data.object_pose_extractor import ObjectPoseExtractor

    suite_dir = args.demos_root / args.suite
    if not suite_dir.exists():
        logger.error("Suite directory not found: %s", suite_dir)
        sys.exit(1)
    hdf5_paths = sorted(suite_dir.glob("*.hdf5"))
    if not hdf5_paths:
        logger.error("No .hdf5 files in %s", suite_dir)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    extractor = ObjectPoseExtractor(suite=args.suite)

    summary = {
        "num_tasks":        len(hdf5_paths),
        "demos_per_task":   args.demos_per_task,
        "tasks_extracted":  [],
        "tasks_failed":     [],
    }
    t0 = time.time()

    for task_idx, hdf5_path in enumerate(hdf5_paths):
        task_stem = hdf5_path.stem.replace("_demo", "")
        demo_keys = [f"demo_{i}" for i in range(args.demos_per_task)]
        logger.info("[%d/%d] Replaying %s — %d demos",
                    task_idx + 1, len(hdf5_paths),
                    hdf5_path.name, args.demos_per_task)

        try:
            data = extractor.extract(hdf5_path, demo_keys=demo_keys)
        except Exception as e:
            logger.error("  FAILED: %s", e)
            summary["tasks_failed"].append({
                "task": hdf5_path.name, "error": str(e),
            })
            continue

        # Write per-demo .npz under <output>/<task_stem>/<demo_id>.npz.
        task_out_dir = args.output / task_stem
        task_out_dir.mkdir(parents=True, exist_ok=True)

        object_names = data["object_names"]
        instruction  = data["instruction"]
        bddl_file    = data["bddl_file"]

        for demo in data["demos"]:
            out_path = task_out_dir / f"{demo['demo_id']}.npz"
            np.savez(
                out_path,
                object_names=np.asarray(object_names),
                object_poses=demo["object_poses"],
                actions=demo["actions"],
                states=demo["states"],
                ee_pos=demo["ee_pos"],
                ee_ori=demo["ee_ori"],
                gripper_states=demo["gripper_states"],
                instruction=instruction,
                bddl_file=bddl_file,
            )

        summary["tasks_extracted"].append({
            "task":          task_stem,
            "demos":         len(data["demos"]),
            "object_names":  object_names,
            "instruction":   instruction,
        })
        logger.info("  OK — %d objects: %s",
                    len(object_names), object_names)

    summary["elapsed_sec"] = round(time.time() - t0, 2)
    with open(args.output / "extract_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done in %.1fs.  %d tasks OK, %d failed.",
                summary["elapsed_sec"],
                len(summary["tasks_extracted"]),
                len(summary["tasks_failed"]))


if __name__ == "__main__":
    main()
