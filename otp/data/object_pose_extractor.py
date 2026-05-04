"""
ObjectPoseExtractor: extract per-frame object SE(3) poses from LIBERO demos.

LIBERO demo HDF5 files do NOT directly store per-frame object poses.
Instead they store:
  - states[T, 92]   : full MuJoCo qpos+qvel snapshots
  - actions[T, 7]   : OSC_POSE delta actions
  - obs/ee_pos, obs/ee_ori, etc.

To recover object poses we replay each demo in a LIBERO sim env:
  1. set_init_state(states[0])
  2. for each action in actions: env.step(action), read obs[<name>_pos/_quat]
This gives (N_obj, T, 4, 4) world-frame pose tensor.

Calibration contract (§5.1): the (object_name → BDDL fixture) mapping is
discovered at extract-time from the env obs keys; mismatches across demos
within the same task raise KeyError.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Object-pose obs keys end with one of these suffixes.  We exclude
# "<name>_to_robot0_eef_pos/_quat" which describe relative pose to EE.
_POS_SUFFIX = "_pos"
_QUAT_SUFFIX = "_quat"
_REL_INFIX = "_to_robot0_eef_"   # relative-to-EE keys we must skip


class ObjectPoseExtractor:
    """
    Replay a LIBERO demo HDF5 in sim to recover per-frame object poses.

    Args:
        suite:        LIBERO benchmark suite name (e.g. 'libero_spatial').
        camera_h, camera_w: dummy camera resolution (low → fast replay).
    """

    def __init__(
        self,
        suite: str,
        camera_h: int = 128,
        camera_w: int = 128,
    ) -> None:
        self.suite = suite
        self.camera_h = camera_h
        self.camera_w = camera_w

    # ------------------------------------------------------------------
    def extract(
        self,
        demo_path: Path,
        demo_keys: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """
        Extract per-frame object SE(3) poses from a LIBERO demo HDF5.

        Args:
            demo_path:  Path to a LIBERO demo HDF5 file.
            demo_keys:  Optional list of demo group names (e.g. ['demo_0']).
                        If None, ALL demos in the file are extracted.

        Returns:
            dict with:
              object_names: list[str] of length N_obj
              demos:        list of per-demo dicts, each with:
                  demo_id:      str (e.g. 'demo_0')
                  object_poses: (N_obj, T, 4, 4) float32, world-frame
                  actions:      (T, 7) float32
                  states:       (T, 92) float64 (MuJoCo qpos+qvel)
                  ee_pos:       (T, 3) float32
                  ee_ori:       (T, 3) float32 (axis-angle)
                  gripper_states: (T, 2) float32
              instruction:  str
              bddl_file:    str (BDDL filename used to build the env)
        """
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "ObjectPoseExtractor.extract requires h5py."
            ) from e

        demo_path = Path(demo_path)
        if not demo_path.exists():
            raise FileNotFoundError(f"Demo HDF5 not found: {demo_path}")

        # Lazy LIBERO imports.
        os.environ.setdefault("MUJOCO_GL", "egl")
        from libero.libero.envs import OffScreenRenderEnv

        with h5py.File(demo_path, "r") as f:
            data_group = f["data"]
            instruction = self._read_instruction(f, data_group)
            bddl_path = self._resolve_bddl_path(data_group, demo_path)
            all_demo_keys = sorted(data_group.keys(),
                                   key=lambda s: int(s.split("_")[-1]))
            keys_to_use = demo_keys if demo_keys is not None else all_demo_keys

            # Pre-load demo arrays we need (states/actions/ee/gripper) to
            # close hdf5 before allocating the env (avoid double resource use).
            demo_arrays: Dict[str, dict] = {}
            for k in keys_to_use:
                if k not in data_group:
                    raise KeyError(f"{demo_path}: demo {k} not found")
                d = data_group[k]
                demo_arrays[k] = {
                    "states":  np.asarray(d["states"]),         # (T, 92)
                    "actions": np.asarray(d["actions"]),        # (T, 7)
                    "ee_pos":  np.asarray(d["obs"]["ee_pos"]).astype(np.float32),
                    "ee_ori":  np.asarray(d["obs"]["ee_ori"]).astype(np.float32),
                    "gripper_states": np.asarray(
                        d["obs"]["gripper_states"]).astype(np.float32),
                }

        # Build sim env.
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=self.camera_h,
            camera_widths=self.camera_w,
        )
        try:
            # Discover object names from a single reset.
            env.reset()
            obs0 = env.set_init_state(demo_arrays[keys_to_use[0]]["states"][0])
            object_names = self._discover_object_names(obs0)
            if not object_names:
                raise KeyError(
                    f"No object pose entries found in env obs.  "
                    f"Available keys: {sorted(obs0.keys())}"
                )

            demos_out: List[dict] = []
            for k in keys_to_use:
                arrs = demo_arrays[k]
                poses = self._replay_demo(env, arrs["states"], arrs["actions"],
                                          object_names)
                demos_out.append({
                    "demo_id":        k,
                    "object_poses":   poses,                # (N_obj, T, 4, 4)
                    "actions":        arrs["actions"].astype(np.float32),
                    "states":         arrs["states"],
                    "ee_pos":         arrs["ee_pos"],
                    "ee_ori":         arrs["ee_ori"],
                    "gripper_states": arrs["gripper_states"],
                })
        finally:
            env.close()

        return {
            "object_names": object_names,
            "demos":        demos_out,
            "instruction":  instruction,
            "bddl_file":    bddl_path,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_instruction(f, data_group) -> str:
        """Pull instruction from data group attrs (LIBERO format)."""
        # LIBERO stores it under data.attrs['problem_info'] as JSON.
        if "problem_info" in data_group.attrs:
            import json
            try:
                info = json.loads(str(data_group.attrs["problem_info"]))
                return info.get("language_instruction", "")
            except Exception:
                pass
        if "instruction" in f.attrs:
            return str(f.attrs["instruction"])
        return ""

    def _resolve_bddl_path(self, data_group, demo_path: Path) -> str:
        """Resolve full BDDL file path for this demo's task."""
        # 1) Try explicit attr in hdf5 (often a relative path).
        bddl_attr = data_group.attrs.get("bddl_file_name", "")
        bddl_attr = str(bddl_attr) if bddl_attr else ""
        # 2) Match the hdf5 filename to a bddl file in the LIBERO package.
        from libero.libero.benchmark import get_benchmark_dict
        bm = get_benchmark_dict()[self.suite]()
        hdf5_stem = demo_path.stem.replace("_demo", "")
        for i in range(bm.get_num_tasks()):
            task = bm.get_task(i)
            bddl_full = bm.get_task_bddl_file_path(i)
            bddl_stem = Path(bddl_full).stem
            # Match either by hdf5 filename containment OR
            # by bddl_file_name attr in the hdf5.
            if bddl_stem in hdf5_stem or hdf5_stem in bddl_stem:
                return bddl_full
            if bddl_attr and Path(bddl_attr).stem == bddl_stem:
                return bddl_full
        # Fallback: error.
        raise FileNotFoundError(
            f"Could not match {demo_path.name} to a bddl file in suite "
            f"{self.suite!r}.  attr='{bddl_attr}'"
        )

    @staticmethod
    def _discover_object_names(obs: dict) -> List[str]:
        """
        Discover object base names from env obs keys.

        Convention:
          <name>_pos  + <name>_quat   → real object pose
          Skip <name>_to_robot0_eef_*  (relative-to-EE)
          Skip robot0_*                (robot proprio)
        """
        keys = list(obs.keys())
        names = set()
        for k in keys:
            if not k.endswith(_POS_SUFFIX):
                continue
            if _REL_INFIX in k:
                continue
            if k.startswith("robot0_"):
                continue
            base = k[: -len(_POS_SUFFIX)]
            if f"{base}{_QUAT_SUFFIX}" in keys:
                names.add(base)
        return sorted(names)

    def _replay_demo(
        self,
        env,
        states: np.ndarray,
        actions: np.ndarray,
        object_names: List[str],
    ) -> np.ndarray:
        """
        Replay a single demo by setting init state then stepping actions.

        Returns:
            (N_obj, T, 4, 4) float32 SE(3) world-frame poses.
        """
        from otp.utils.lie_algebra import quat_to_so3
        import torch

        T = len(actions)
        N_obj = len(object_names)
        out = np.zeros((N_obj, T, 4, 4), dtype=np.float32)

        env.reset()
        obs = env.set_init_state(states[0])
        # Extract t=0 poses from the post-set_init_state obs.
        self._fill_pose_frame(out, obs, object_names, t=0, quat_to_so3=quat_to_so3, torch=torch)

        # Replay actions for t=1..T-1.
        for t in range(1, T):
            obs, _, _, _ = env.step(actions[t - 1])
            self._fill_pose_frame(out, obs, object_names, t=t,
                                  quat_to_so3=quat_to_so3, torch=torch)

        return out

    @staticmethod
    def _fill_pose_frame(out, obs, object_names, t, quat_to_so3, torch) -> None:
        """Read object poses at frame t from obs dict and fill out[:, t]."""
        for n_idx, name in enumerate(object_names):
            pos = np.asarray(obs[f"{name}{_POS_SUFFIX}"], dtype=np.float32)
            quat = np.asarray(obs[f"{name}{_QUAT_SUFFIX}"], dtype=np.float32)
            R = quat_to_so3(torch.from_numpy(quat).unsqueeze(0)).squeeze(0).numpy()
            out[n_idx, t, :3, :3] = R
            out[n_idx, t, :3, 3] = pos
            out[n_idx, t, 3, 3] = 1.0
