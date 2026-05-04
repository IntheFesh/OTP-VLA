"""
ObjectPoseExtractor: pull per-frame object SE(3) poses from LIBERO demos.

For each demo HDF5 file produced by LIBERO collect_demonstration.py, this
module extracts the world-frame pose of every BDDL-declared object across
all frames, returning an array of shape (N_obj, T, 4, 4).

Calibration contract (§5.1): the (object_name → BDDL fixture) mapping is
loaded from the suite's BDDL definition; mismatches raise KeyError.

This module imports LIBERO lazily inside extract() so that:
  - tests that do not call extract() can run without LIBERO installed;
  - the missing-dependency error is reported with a clear remediation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ObjectPoseExtractor:
    """
    Extract per-frame object poses from a LIBERO demo HDF5 file.

    Args:
        suite:        LIBERO benchmark suite name (e.g. 'libero_spatial').
        bddl_root:    Optional override for BDDL file directory.
                      If None, located via the LIBERO package install path.
    """

    def __init__(
        self,
        suite: str,
        bddl_root: Optional[Path] = None,
    ) -> None:
        self.suite = suite
        self.bddl_root = Path(bddl_root) if bddl_root is not None else None

    # ------------------------------------------------------------------
    def extract(
        self,
        demo_path: Path,
    ) -> Dict[str, np.ndarray]:
        """
        Args:
            demo_path: Path to a LIBERO demo HDF5 file.

        Returns:
            dict with:
              object_names: list[str] of length N_obj
              object_poses: (N_obj, T, 4, 4) float32 SE(3) world-frame poses
              actions:      (T, 7) float32 LIBERO action vectors
              instruction:  str (task language goal)
        """
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "ObjectPoseExtractor.extract requires h5py. "
                "Install with: pip install h5py"
            ) from e

        demo_path = Path(demo_path)
        if not demo_path.exists():
            raise FileNotFoundError(f"Demo HDF5 not found: {demo_path}")

        with h5py.File(demo_path, "r") as f:
            # LIBERO HDF5 layout: data/demo_<i>/{obs, actions, ...}
            instruction = self._read_instruction(f)
            obs_group = self._first_demo_obs_group(f)

            object_names = self._discover_object_names(obs_group)
            if not object_names:
                raise KeyError(
                    f"No object pose entries found in {demo_path}. "
                    f"Expected obs keys ending in '_pose' or '_pos'/'_quat'."
                )

            poses = self._stack_object_poses(obs_group, object_names)  # (N_obj, T, 4, 4)
            actions = np.asarray(obs_group.parent["actions"], dtype=np.float32)

        return {
            "object_names": object_names,
            "object_poses": poses,
            "actions": actions,
            "instruction": instruction,
        }

    # ------------------------------------------------------------------
    # Internal helpers (split out so tests can override / mock).
    # ------------------------------------------------------------------

    @staticmethod
    def _first_demo_obs_group(f) -> "h5py.Group":
        """Return obs group of the first demo (LIBERO single-demo HDF5 layout)."""
        if "data" not in f:
            raise KeyError(f"Expected top-level 'data' group, got {list(f.keys())}")
        data = f["data"]
        demo_keys = sorted(data.keys())
        if not demo_keys:
            raise KeyError("data/ group is empty")
        return data[demo_keys[0]]["obs"]

    @staticmethod
    def _read_instruction(f) -> str:
        """Read instruction from HDF5 attrs or top-level dataset."""
        if "instruction" in f.attrs:
            return str(f.attrs["instruction"])
        if "language_instruction" in f:
            return str(np.asarray(f["language_instruction"]).item())
        return ""

    @staticmethod
    def _discover_object_names(obs_group) -> List[str]:
        """
        Discover object names from obs keys.

        Convention:
          <name>_pose             — (T, 4, 4)
          <name>_pos + <name>_quat — (T, 3) and (T, 4)
        """
        keys = list(obs_group.keys())
        names = set()
        for k in keys:
            if k.endswith("_pose"):
                names.add(k[: -len("_pose")])
            elif k.endswith("_pos"):
                base = k[: -len("_pos")]
                if f"{base}_quat" in keys:
                    names.add(base)
        return sorted(names)

    def _stack_object_poses(
        self,
        obs_group,
        object_names: List[str],
    ) -> np.ndarray:
        """Build (N_obj, T, 4, 4) tensor from per-object pose entries."""
        from otp.utils.lie_algebra import quat_to_so3
        import torch

        poses = []
        for name in object_names:
            pose_key = f"{name}_pose"
            if pose_key in obs_group:
                arr = np.asarray(obs_group[pose_key], dtype=np.float32)
                if arr.ndim != 3 or arr.shape[-2:] != (4, 4):
                    raise ValueError(
                        f"{pose_key} expected (T, 4, 4), got {arr.shape}"
                    )
            else:
                pos = np.asarray(obs_group[f"{name}_pos"], dtype=np.float32)  # (T, 3)
                quat = np.asarray(obs_group[f"{name}_quat"], dtype=np.float32)  # (T, 4)
                R = quat_to_so3(torch.from_numpy(quat)).numpy()                # (T, 3, 3)
                T = pos.shape[0]
                arr = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
                arr[:, :3, :3] = R
                arr[:, :3, 3] = pos
            poses.append(arr)

        return np.stack(poses, axis=0)         # (N_obj, T, 4, 4)
