"""
LIBEROOTPDataset: PyTorch dataset for LIBERO demos with grasp affordance.

Per-sample dict keys:
  image                   (3, H, W)         uint8   RGB
  instruction             str
  action                  (7,)              float32 LIBERO action
  ee_pose                 (4, 4)            float32 EE world pose
  object_poses            (N_obj, 4, 4)     float32 object world poses
  object_names            list[str]
  grasp_affordance        (N_obj, K, 7)     float32 [pos | quat] per grasp
  grasp_affordance_mask   (N_obj, K)        bool    True = valid grasp
  object_point_clouds     (N_obj, P, 3)     float32 mesh vertices

Calibration contracts:
  §3.3 — view_keys resolved at init time; raises if no camera.
  §5.1 — normalizer JSON must exist; raises FileNotFoundError otherwise.
  §5.1 — grasp affordance npz must exist for every object_name encountered;
         silent zero-fill is forbidden — caller must regenerate cache.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LIBEROOTPDataset:
    """
    PyTorch-style dataset over LIBERO demo files with grasp affordance cache.

    Args:
        root:                 Directory containing .hdf5 demo files.
        suite:                Suite name (e.g. 'libero_spatial').
        grasp_affordance_dir: Directory holding precomputed per-object grasp
                              affordance npz files (one file per unique
                              object name, written by
                              precompute_all_affordances).  REQUIRED.
        view_keys:            Camera names to load images from.  Resolved
                              automatically if None (§3.3).
        normalizer_path:      Override for normalizer JSON.  Defaults to
                              <root>/meta/stats_qpace.json (§5.1).
        num_grasps_per_object: K — pad/clip per-object grasps to this count.
        num_points:           P — point cloud vertex count per object.
    """

    def __init__(
        self,
        root: Path,
        suite: str = "libero_spatial",
        grasp_affordance_dir: Optional[Path] = None,
        view_keys: Optional[List[str]] = None,
        normalizer_path: Optional[Path] = None,
        num_grasps_per_object: int = 8,
        num_points: int = 256,
    ) -> None:
        self.root = Path(root)
        self.suite = suite
        self.num_grasps_per_object = num_grasps_per_object
        self.num_points = num_points

        if grasp_affordance_dir is None:
            raise ValueError(
                "LIBEROOTPDataset requires `grasp_affordance_dir`. "
                "Run scripts/03c_extract_grasp_affordances.py first."
            )
        self.grasp_affordance_dir = Path(grasp_affordance_dir)
        if not self.grasp_affordance_dir.exists():
            raise FileNotFoundError(
                f"grasp_affordance_dir not found: {self.grasp_affordance_dir}. "
                f"Run scripts/03c_extract_grasp_affordances.py."
            )

        self.demo_paths = self._scan_demo_paths()

        # §5.1 normalizer.
        norm_path = (
            Path(normalizer_path)
            if normalizer_path is not None
            else self.root / "meta" / "stats_qpace.json"
        )
        if not norm_path.exists():
            raise FileNotFoundError(
                f"Normalizer stats missing: {norm_path}. "
                f"Run scripts/02_compute_normalizer.py first (§5.1)."
            )
        with open(norm_path, "r") as f:
            self.norm_stats: Dict[str, Any] = json.load(f)

        # §3.3 view-key resolution.
        self.view_keys = view_keys if view_keys is not None else self._resolve_view_keys()

        # Per-frame index across all demos.
        self._index: List[Tuple[Path, int]] = self._build_index()

        # Per-object affordance cache (lazy, populated on first use).
        self._affordance_cache: Dict[str, Dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "LIBEROOTPDataset.__getitem__ requires h5py. "
                "Install with: pip install h5py"
            ) from e

        demo_path, frame_idx = self._index[idx]

        with h5py.File(demo_path, "r") as f:
            obs = f["data"][sorted(f["data"].keys())[0]]["obs"]
            actions = np.asarray(f["data"][sorted(f["data"].keys())[0]]["actions"])

            image = self._load_image(obs, frame_idx)
            instruction = self._read_instruction(f)
            action = actions[frame_idx].astype(np.float32)
            ee_pose = self._load_ee_pose(obs, frame_idx)
            object_poses, object_names = self._load_object_poses(obs, frame_idx)

        # Affordance lookup (M-fail-loud per §5.1).
        affordance, affordance_mask, point_clouds = self._lookup_affordance(object_names)

        return {
            "image":                 image,
            "instruction":           instruction,
            "action":                action,
            "ee_pose":               ee_pose,
            "object_poses":          object_poses,
            "object_names":          object_names,
            "grasp_affordance":      affordance,
            "grasp_affordance_mask": affordance_mask,
            "object_point_clouds":   point_clouds,
        }

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _scan_demo_paths(self) -> List[Path]:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        paths = sorted(self.root.glob("**/*.hdf5"))
        if not paths:
            raise FileNotFoundError(
                f"No .hdf5 demos under {self.root}. "
                f"Run scripts/03_extract_object_poses.py first."
            )
        return paths

    def _resolve_view_keys(self) -> List[str]:
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "LIBEROOTPDataset requires h5py to resolve view keys."
            ) from e

        with h5py.File(self.demo_paths[0], "r") as f:
            obs = f["data"][sorted(f["data"].keys())[0]]["obs"]
            candidates = [k for k in obs.keys() if k.endswith("_rgb") or k.endswith("_image")]
            if not candidates:
                raise KeyError(
                    f"No image-like keys in obs of {self.demo_paths[0]}. "
                    f"Found keys: {list(obs.keys())}"
                )
        return sorted(candidates)

    def _build_index(self) -> List[Tuple[Path, int]]:
        try:
            import h5py
        except ImportError:
            return []

        index: List[Tuple[Path, int]] = []
        for path in self.demo_paths:
            with h5py.File(path, "r") as f:
                key = sorted(f["data"].keys())[0]
                T = len(f["data"][key]["actions"])
            for i in range(T):
                index.append((path, i))
        return index

    # ------------------------------------------------------------------
    # Per-frame loaders
    # ------------------------------------------------------------------

    def _load_image(self, obs, frame_idx: int) -> np.ndarray:
        view = self.view_keys[0]
        img = np.asarray(obs[view][frame_idx])
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.transpose(2, 0, 1)
        return img.astype(np.uint8)

    @staticmethod
    def _read_instruction(f) -> str:
        if "instruction" in f.attrs:
            return str(f.attrs["instruction"])
        if "language_instruction" in f:
            return str(np.asarray(f["language_instruction"]).item())
        return ""

    @staticmethod
    def _load_ee_pose(obs, frame_idx: int) -> np.ndarray:
        if "ee_pose" in obs:
            return np.asarray(obs["ee_pose"][frame_idx], dtype=np.float32)
        from otp.utils.lie_algebra import quat_to_so3
        import torch
        pos = np.asarray(obs["ee_pos"][frame_idx], dtype=np.float32)
        quat = np.asarray(obs["ee_quat"][frame_idx], dtype=np.float32)
        R = quat_to_so3(torch.from_numpy(quat).unsqueeze(0)).squeeze(0).numpy()
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    @staticmethod
    def _load_object_poses(obs, frame_idx: int) -> Tuple[np.ndarray, List[str]]:
        from otp.data.object_pose_extractor import ObjectPoseExtractor
        names = ObjectPoseExtractor._discover_object_names(obs)
        poses = []
        for n in names:
            pose_key = f"{n}_pose"
            if pose_key in obs:
                poses.append(np.asarray(obs[pose_key][frame_idx], dtype=np.float32))
            else:
                from otp.utils.lie_algebra import quat_to_so3
                import torch
                pos = np.asarray(obs[f"{n}_pos"][frame_idx], dtype=np.float32)
                quat = np.asarray(obs[f"{n}_quat"][frame_idx], dtype=np.float32)
                R = quat_to_so3(torch.from_numpy(quat).unsqueeze(0)).squeeze(0).numpy()
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = R
                T[:3, 3] = pos
                poses.append(T)
        if poses:
            return np.stack(poses, axis=0), names
        return np.zeros((0, 4, 4), dtype=np.float32), names

    # ------------------------------------------------------------------
    # Grasp affordance loader (cache-backed)
    # ------------------------------------------------------------------

    def _lookup_affordance(
        self,
        object_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For each object name, look up the cached affordance npz.

        Raises:
            FileNotFoundError: If any object name has no corresponding npz
                               under self.grasp_affordance_dir.  Silent
                               zero-fill is explicitly forbidden by §5.1.
        """
        K = self.num_grasps_per_object
        P = self.num_points
        N = len(object_names)

        affordance = np.zeros((N, K, 7), dtype=np.float32)
        affordance_mask = np.zeros((N, K), dtype=bool)
        point_clouds = np.zeros((N, P, 3), dtype=np.float32)

        for i, name in enumerate(object_names):
            cached = self._affordance_cache.get(name)
            if cached is None:
                npz_path = self.grasp_affordance_dir / f"{name}.npz"
                if not npz_path.exists():
                    raise FileNotFoundError(
                        f"Grasp affordance file missing for object {name!r}: "
                        f"{npz_path}.  Re-run "
                        f"scripts/03c_extract_grasp_affordances.py "
                        f"to populate the cache (§5.1: silent zero-fill forbidden)."
                    )
                data = np.load(npz_path)
                cached = {
                    "grasps":        np.asarray(data["grasps"], dtype=np.float32),
                    "valid_mask":    np.asarray(data["valid_mask"], dtype=bool),
                    "mesh_vertices": np.asarray(data["mesh_vertices"], dtype=np.float32),
                }
                self._affordance_cache[name] = cached

            g = cached["grasps"]
            vm = cached["valid_mask"]
            mv = cached["mesh_vertices"]

            # Pad / truncate per-object grasps to K.
            k_avail = min(g.shape[0], K)
            affordance[i, :k_avail] = g[:k_avail]
            affordance_mask[i, :k_avail] = vm[:k_avail]

            # Pad / truncate point cloud to P.
            p_avail = min(mv.shape[0], P)
            point_clouds[i, :p_avail] = mv[:p_avail]
            if p_avail < P:
                # Repeat the last vertex to fill the buffer (deterministic).
                point_clouds[i, p_avail:] = mv[-1:]

        return affordance, affordance_mask, point_clouds


# Backwards-compatible alias for any code/tests written against the old name.
LiberoDataset = LIBEROOTPDataset
