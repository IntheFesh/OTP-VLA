#!/usr/bin/env python
"""
scripts/03b_generate_perturbation_suite.py

Generate the full controlled perturbation suite (paper §V.B) using the public
PerturbationProtocol class.  Reads previously-extracted demos from
data/object_poses/ (produced by 03_extract_object_poses.py) and writes:

    <output>/manifest.json
    <output>/samples/<task>_<type>_<idx>.json
    <output>/samples/<task>_visual_style_<idx>.npy

Usage:
    python scripts/03b_generate_perturbation_suite.py \\
        --suite libero_spatial \\
        --num-samples-per-task-per-type 30 \\
        --extracted-dir data/object_poses/ \\
        --output data/perturbation_suite/

The reported statistics (per-task per-type valid sample counts) are written
to <output>/manifest.json — paper Table § lists these per cell, no averaging.
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
logger = logging.getLogger("generate_perturbation_suite")


def _print_startup(args: argparse.Namespace) -> None:
    """§1.2 startup snapshot."""
    print("=" * 72)
    print(f"[ENV]  python={sys.version.split()[0]}  cwd={os.getcwd()}")
    print(f"[CFG]  suite={args.suite}  "
          f"num_samples_per_task_per_type={args.num_samples_per_task_per_type}  "
          f"seed={args.seed}")
    print(f"[DATA] extracted_dir={args.extracted_dir}")
    print(f"[OUT]  output={args.output}")
    print(f"[VOCAB] {args.vocab_path or '(default)'}")
    print("=" * 72)


def _load_demos_from_extracted(extracted_dir: Path) -> list[dict]:
    """
    Convert extracted .npz files into the dict format that
    PerturbationProtocol.generate_full_suite consumes.
    """
    demos = []
    for npz_path in sorted(extracted_dir.glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        names = list(data["object_names"])
        if not names:
            logger.warning("%s has no objects, skipping", npz_path.name)
            continue

        # Heuristic: target_object = first object in BDDL declaration order.
        target_object = names[0]
        # Use the action at the midpoint as a representative GT action.
        actions = data["actions"]
        mid = len(actions) // 2

        demos.append({
            "task_id":       npz_path.stem,
            "instruction":   str(data["instruction"]) or "pick up the object",
            "target_object": target_object,
            "scene_objects": names,
            "image":         np.zeros((128, 128, 3), dtype=np.uint8),
            "action":        actions[mid].astype(np.float32),
        })
    return demos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--num-samples-per-task-per-type", type=int, default=30)
    parser.add_argument("--max-attempts-per-sample", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extracted-dir", type=Path,
                        default=Path("data/object_poses/"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=None,
                        help="Override perturbation_vocab.yaml location")
    args = parser.parse_args()

    _print_startup(args)

    from otp.data.perturbation_protocol import PerturbationProtocol

    if not args.extracted_dir.exists():
        logger.error("Extracted directory missing: %s", args.extracted_dir)
        logger.error("Run scripts/03_extract_object_poses.py first.")
        sys.exit(1)

    demos = _load_demos_from_extracted(args.extracted_dir)
    if not demos:
        logger.error("No usable demos found under %s", args.extracted_dir)
        sys.exit(1)
    logger.info("Loaded %d demos", len(demos))

    proto = PerturbationProtocol(
        seed=args.seed,
        num_samples_per_task_per_type=args.num_samples_per_task_per_type,
        max_attempts_per_sample=args.max_attempts_per_sample,
        vocab_path=args.vocab_path,
    )

    t0 = time.time()
    result = proto.generate_full_suite(demos, args.output)
    elapsed = time.time() - t0

    logger.info("Done in %.1fs", elapsed)
    logger.info("Manifest: %s", result["manifest_path"])

    # Per-task / per-type valid sample report (paper §V.B).
    print("\n=== Per-task / per-type valid sample counts ===")
    for task_id, by_type in result["stats"].items():
        line = f"  {task_id}: " + "  ".join(
            f"{ptype}={n}" for ptype, n in by_type.items()
        )
        print(line)


if __name__ == "__main__":
    main()
