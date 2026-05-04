"""
LIBEROOTPDataset: PyTorch dataset for LIBERO demos with grasp affordance.

Data sources:
  - data/object_poses/<task_stem>/demo_X.npz  (REQUIRED)
        Pre-extracted GT object poses, actions, ee state, instruction
        (produced by scripts/03_extract_object_poses.py via sim replay).
  - <libero_root>/<suite>/*.hdf5              (REQUIRED for image loading)
        Original LIBERO hdf5 demo files (agentview_rgb / eye_in_hand_rgb).
  - data/grasp_affordances/<obj_class>.npz    (REQUIRED)
        Pre-computed grasp affordance per object class
        (produced by scripts/03c_extract_grasp_affordances.py).

Per-sample dict keys:
  image                   (3, H, W)         uint8   RGB
  instruction             str
  action                  (7,)              float32 LIBERO action
  ee_pose                 (4, 4)            float32 EE world pose
  object_poses            (N_obj, 4, 4)     float32 object world poses
  object_names            list[str]                 instance names (with _N suffix)
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
    PyTorch-style dataset over LIBERO demos.

    Args:
        root:                    Pre-extracted object-pose cache directory
                                 (data/object_poses/) — REQUIRED.  Each demo
                                 is one .npz file under <root>/<task>/<demo>.npz.
        suite:                   Suite name (e.g. 'libero_spatial').
        libero_hdf5_root:        LIBERO original hdf5 directory (e.g.
                                 /root/autodl-tmp/libero_data/libero_spatial/)
                                 — REQUIRED for loading images.  npz files do
                                 not contain image data.
        grasp_affordance_dir:    Directory holding precomputed per-object
                                 grasp affordance npz files.  REQUIRED.
        view_keys:               Camera names to load images from.  Resolved
                                 automatically if None.
        normalizer_path:         Override for normalizer JSON.  Defaults to
                                 <root>/meta/stats_qpace.json.
        num_grasps_per_object:   K — pad/clip per-object grasps to this count.
        num_points:              P — point cloud vertex count per object.
    """

    def __init__(
        self,
        root: Path,
        suite: str = "libero_spatial",
        libero_hdf5_root: Optional[Path] = None,
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

        if not self.root.exists():
            raise FileNotFoundError(
                f"Object-pose cache root not found: {self.root}.  Run "
                f"scripts/03_extract_object_poses.py first."
            )

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

        # libero_hdf5_root resolution: prefer explicit arg, then guess from
        # /root/autodl-tmp/libero_data/<suite>/, else require explicit pass.
        if libero_hdf5_root is None:
            guess = Path("/root/autodl-tmp/libero_data") / suite
            if guess.exists():
                libero_hdf5_root = guess
            else:
                raise FileNotFoundError(
                    f"libero_hdf5_root not provided and default {guess} "
                    f"does not exist.  Pass `libero_hdf5_root` explicitly."
                )
        self.libero_hdf5_root = Path(libero_hdf5_root)
        if not self.libero_hdf5_root.exists():
            raise FileNotFoundError(
                f"libero_hdf5_root not found: {self.libero_hdf5_root}"
            )

        # Discover npz demos and their corresponding hdf5 files.
        self.demo_records = self._scan_demos()

        # Normalizer (currently unused by __getitem__; kept for forward compat).
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

        # View-key resolution from the first hdf5.
        self.view_keys = view_keys if view_keys is not None else self._resolve_view_keys()

        # Per-frame index across all demos.
        self._index: List[Tuple[int, int]] = self._build_index()

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
                "LIBEROOTPDataset.__getitem__ requires h5py."
            ) from e

        demo_idx, frame_idx = self._index[idx]
        record = self.demo_records[demo_idx]

        # 1. Load all per-frame data from npz cache.
        npz = np.load(record["npz_path"], allow_pickle=False)
        action = np.asarray(npz["actions"][frame_idx], dtype=np.float32)
        ee_pose = self._build_ee_pose(
            npz["ee_pos"][frame_idx], npz["ee_ori"][frame_idx],
        )
        # object_poses npz layout: (N_obj, T, 4, 4) — transpose to (N_obj, 4, 4)
        object_poses_full = np.asarray(npz["object_poses"], dtype=np.float32)
        object_poses = object_poses_full[:, frame_idx, :, :]
        object_names = [str(n) for n in npz["object_names"]]
        instruction = str(npz["instruction"].item()) if npz["instruction"].ndim == 0 \
                      else str(npz["instruction"])

        # 2. Load image from hdf5 (npz doesn't store images).
        with h5py.File(record["hdf5_path"], "r") as f:
            obs = f["data"][record["demo_key"]]["obs"]
            image = self._load_image(obs, frame_idx)

        # 3. Affordance lookup (fail-loud).
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

    def _scan_demos(self) -> List[Dict[str, Any]]:
        """
        Walk root/<task>/demo_X.npz; for each, locate the corresponding
        hdf5 and demo key.  A demo with missing hdf5 is skipped with warning.
        """
        records: List[Dict[str, Any]] = []
        npz_paths = sorted(self.root.glob("*/demo_*.npz"))
        if not npz_paths:
            raise FileNotFoundError(
                f"No demo_*.npz under {self.root}/<task>/.  Run "
                f"scripts/03_extract_object_poses.py first."
            )

        for npz_path in npz_paths:
            task_stem = npz_path.parent.name
            demo_id = npz_path.stem                          # e.g. "demo_3"
            hdf5_path = self.libero_hdf5_root / f"{task_stem}_demo.hdf5"
            if not hdf5_path.exists():
                logger.warning(
                    "Skipping %s — hdf5 not found: %s", npz_path.name, hdf5_path,
                )
                continue
            records.append({
                "npz_path":  npz_path,
                "hdf5_path": hdf5_path,
                "demo_key":  demo_id,
            })

        if not records:
            raise FileNotFoundError(
                f"No demos resolved.  Checked {len(npz_paths)} npz files "
                f"under {self.root}; none had a corresponding hdf5 in "
                f"{self.libero_hdf5_root}."
            )
        return records

    def _resolve_view_keys(self) -> List[str]:
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "LIBEROOTPDataset requires h5py to resolve view keys."
            ) from e

        first = self.demo_records[0]
        with h5py.File(first["hdf5_path"], "r") as f:
            obs = f["data"][first["demo_key"]]["obs"]
            candidates = [k for k in obs.keys()
                          if k.endswith("_rgb") or k.endswith("_image")]
            if not candidates:
                raise KeyError(
                    f"No image-like keys in obs of {first['hdf5_path']}. "
                    f"Found keys: {list(obs.keys())}"
                )
        # Prefer agentview / front-view if present, else first sorted.
        preferred = [k for k in candidates if "agentview" in k]
        if preferred:
            return sorted(preferred) + sorted(k for k in candidates if k not in preferred)
        return sorted(candidates)

    def _build_index(self) -> List[Tuple[int, int]]:
        """Per-frame index using npz `actions` length (== T per demo)."""
        index: List[Tuple[int, int]] = []
        for demo_idx, record in enumerate(self.demo_records):
            with np.load(record["npz_path"]) as npz:
                T = len(npz["actions"])
            for i in range(T):
                index.append((demo_idx, i))
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
    def _build_ee_pose(pos: np.ndarray, ori: np.ndarray) -> np.ndarray:
        """
        Compose SE(3) from ee_pos (3,) + ee_ori (3,) axis-angle.

        LIBERO npz stores ee_ori as axis-angle (3,) in world frame, not quat.
        """
        from otp.utils.lie_algebra import so3_exp
        import torch

        pos = np.asarray(pos, dtype=np.float32)
        ori = np.asarray(ori, dtype=np.float32)
        R = so3_exp(torch.from_numpy(ori).unsqueeze(0)).squeeze(0).numpy()
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    # ------------------------------------------------------------------
    # Grasp affordance loader (cache-backed)
    # ------------------------------------------------------------------

    def _lookup_affordance(
        self,
        object_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For each object name, look up the cached affordance npz.

        LIBERO instance names are like "akita_black_bowl_1"; the affordance
        cache is keyed by base class "akita_black_bowl".  Strip a trailing
        "_<digit>+" suffix to map instance -> base.

        Raises:
            FileNotFoundError: If any base name has no corresponding npz.
        """
        import re as _re_strip

        K = self.num_grasps_per_object
        P = self.num_points
        N = len(object_names)

        affordance = np.zeros((N, K, 7), dtype=np.float32)
        affordance_mask = np.zeros((N, K), dtype=bool)
        point_clouds = np.zeros((N, P, 3), dtype=np.float32)

        for i, name in enumerate(object_names):
            cached = self._affordance_cache.get(name)
            if cached is None:
                base_name = _re_strip.sub(r"_\d+$", "", name)
                npz_path = self.grasp_affordance_dir / f"{base_name}.npz"
                if not npz_path.exists():
                    raise FileNotFoundError(
                        f"Grasp affordance file missing for object {name!r} "
                        f"(base={base_name!r}): {npz_path}.  Re-run "
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

            k_avail = min(g.shape[0], K)
            affordance[i, :k_avail] = g[:k_avail]
            affordance_mask[i, :k_avail] = vm[:k_avail]

            p_avail = min(mv.shape[0], P)
            point_clouds[i, :p_avail] = mv[:p_avail]
            if p_avail < P:
                point_clouds[i, p_avail:] = mv[-1:]

        return affordance, affordance_mask, point_clouds


# Backwards-compatible alias for any code/tests written against the old name.
LiberoDataset = LIBEROOTPDataset
