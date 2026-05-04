"""
LiberoDataset: PyTorch Dataset wrapping a directory of LIBERO demo HDF5 files.

Returns per-frame samples with:
  image           (3, 256, 256) uint8 RGB
  instruction     str
  action          (7,) float32 LIBERO action [dx, dy, dz, dax, day, daz, gripper]
  ee_pose         (4, 4) float32 EE world-frame pose
  object_poses    (N_obj, 4, 4) float32 object world-frame poses
  object_names    list[str] of length N_obj

Calibration contract (§5.1): a normaliser stats file
`<root>/meta/stats_qpace.json` must exist; raise FileNotFoundError otherwise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class LiberoDataset:
    """
    PyTorch-style dataset over LIBERO demo files.

    Args:
        root:        Directory containing .hdf5 demo files.
        suite:       Suite name (e.g. 'libero_spatial').
        view_keys:   Camera names to load images from.
                     If None, resolved at init time via _resolve_view_keys
                     (raises if no usable camera found — §3.3).
        normalizer_path: Optional override for normalizer JSON.
                     Defaults to <root>/meta/stats_qpace.json (§5.1).
    """

    def __init__(
        self,
        root: Path,
        suite: str = "libero_spatial",
        view_keys: Optional[List[str]] = None,
        normalizer_path: Optional[Path] = None,
    ) -> None:
        self.root = Path(root)
        self.suite = suite
        self.demo_paths = self._scan_demo_paths()

        # §5.1 — normalizer must exist.
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

        # §3.3 — resolve view keys at init.
        self.view_keys = view_keys if view_keys is not None else self._resolve_view_keys()

        # Build per-frame index across all demos.
        self._index: List[tuple] = self._build_index()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "LiberoDataset.__getitem__ requires h5py. "
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

        return {
            "image":         image,
            "instruction":   instruction,
            "action":        action,
            "ee_pose":       ee_pose,
            "object_poses":  object_poses,
            "object_names":  object_names,
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
        """
        §3.3: probe the first demo for camera-image keys and pick a usable one.

        Conventional LIBERO obs keys are 'agentview_rgb' / 'eye_in_hand_rgb' —
        we keep both if present, else raise.
        """
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "LiberoDataset requires h5py to resolve view keys."
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

    def _build_index(self) -> List[tuple]:
        """Build a flat list of (demo_path, frame_idx) tuples."""
        try:
            import h5py
        except ImportError:
            # Defer index build until first __getitem__ if h5py missing.
            return []

        index = []
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
        """Load (3, H, W) uint8 image from the first available view."""
        view = self.view_keys[0]
        img = np.asarray(obs[view][frame_idx])
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.transpose(2, 0, 1)        # HWC → CHW
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
        # Fallback: assemble from ee_pos + ee_quat.
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
    def _load_object_poses(obs, frame_idx: int):
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
        return np.stack(poses, axis=0) if poses else np.zeros((0, 4, 4), dtype=np.float32), names
