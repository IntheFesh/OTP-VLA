#!/usr/bin/env python
"""
scripts/03c_extract_grasp_affordances.py

Pre-compute antipodal grasp affordances for every object referenced by any
LIBERO BDDL file under --libero-asset-dir.  Output layout:

    <output_dir>/<object_name>.npz
        grasps:        (8, 7) float32 [pos | quat(wxyz)]
        mesh_vertices: (256, 3) float32
        valid_mask:    (8,)     bool
    <output_dir>/_meshes/<object_name>.mesh.npz       (intermediate)
    <output_dir>/affordance_manifest.json              (summary)

Usage:
    python scripts/03c_extract_grasp_affordances.py \\
        --libero-asset-dir third_party/LIBERO/libero/libero \\
        --output-dir data/grasp_affordances/ \\
        --num-grasps 8 \\
        --gripper-width 0.08 \\
        --friction-coef 0.5

Requires `trimesh` and a checked-out LIBERO repo with BDDL files +
fixture XMLs under `libero/libero/{bddl_files,assets}/`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(name)s | %(message)s",
)
logger = logging.getLogger("extract_grasp_affordances")


def _print_startup(args: argparse.Namespace) -> None:
    """§1.2 startup snapshot."""
    print("=" * 72)
    print(f"[ENV]  python={sys.version.split()[0]}  cwd={os.getcwd()}")
    print(f"[CFG]  num_grasps={args.num_grasps}  "
          f"gripper_width={args.gripper_width}  "
          f"friction_coef={args.friction_coef}  num_points={args.num_points}")
    print(f"[DATA] libero_asset_dir={args.libero_asset_dir}")
    print(f"[OUT]  output_dir={args.output_dir}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-asset-dir", type=Path, required=True,
                        help="LIBERO root containing bddl_files/ and assets/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-grasps", type=int, default=8)
    parser.add_argument("--gripper-width", type=float, default=0.08)
    parser.add_argument("--friction-coef", type=float, default=0.5)
    parser.add_argument("--num-points", type=int, default=256)
    args = parser.parse_args()

    _print_startup(args)

    if not args.libero_asset_dir.exists():
        logger.error("libero-asset-dir not found: %s", args.libero_asset_dir)
        logger.error("Hint: clone LIBERO at third_party/LIBERO and pass "
                     "third_party/LIBERO/libero/libero")
        sys.exit(1)

    from otp.data.grasp_affordance import precompute_all_affordances

    t0 = time.time()
    try:
        written = precompute_all_affordances(
            libero_object_dir=args.libero_asset_dir,
            grasp_output_dir=args.output_dir,
            num_grasps=args.num_grasps,
            gripper_width=args.gripper_width,
            friction_coef=args.friction_coef,
            num_points=args.num_points,
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except ImportError as e:
        logger.error("%s", e)
        logger.error("Install trimesh: pip install trimesh")
        sys.exit(1)
    elapsed = time.time() - t0

    manifest_path = args.output_dir / "affordance_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        n_invalid = sum(
            1 for obj in manifest["objects"]
            if not bool(
                __import__("numpy").load(
                    args.output_dir / f"{obj}.npz"
                )["valid_mask"].any()
            )
        )
    else:
        n_invalid = 0

    print()
    print(f"Done in {elapsed:.1f}s.  Wrote {len(written)} object affordance files.")
    print(f"Manifest: {manifest_path}")
    print(f"Objects with NO valid grasps: {n_invalid}")


if __name__ == "__main__":
    main()
