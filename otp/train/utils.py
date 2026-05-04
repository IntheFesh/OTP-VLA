"""
otp/train/utils.py

Shared training utilities that do NOT import Hydra or WandB at module level.
Safe to import from tests.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset



# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """
    Returns random tensors with the same structure as LIBEROOTPDataset.
    All values are seeded per-index for reproducibility.
    """

    def __init__(
        self,
        num_samples: int,
        num_objects: int = 5,
        horizon: int = 8,
        num_grasps: int = 8,
        num_points: int = 256,
        img_h: int = 128,
        img_w: int = 128,
    ) -> None:
        self.num_samples = num_samples
        self.N = num_objects
        self.H = horizon
        self.K = num_grasps
        self.P = num_points
        self.img_h = img_h
        self.img_w = img_w

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = np.random.RandomState(idx)
        N, H, K, P = self.N, self.H, self.K, self.P
        return {
            "image":                rng.randint(0, 256, (3, self.img_h, self.img_w),
                                                dtype=np.uint8),
            "instruction":          "pick up the object and place it on the plate",
            "action_chunk":         rng.randn(H, 7).astype(np.float32),
            "ee_pose":              np.eye(4, dtype=np.float32),
            "object_poses":         np.stack([np.eye(4, dtype=np.float32)] * N),
            "object_names":         [f"obj_{i}" for i in range(N)],
            "grasp_affordance":     rng.randn(N, K, 7).astype(np.float32),
            "grasp_affordance_mask": np.ones((N, K), dtype=bool),
            "object_point_clouds":  rng.randn(N, P, 3).astype(np.float32),
            "gt_trajectory":        rng.randn(N, H, 6).astype(np.float32),
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate: keeps strings as lists, stacks ndarray/Tensor fields."""
    out: Dict[str, Any] = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if isinstance(vals[0], str):
            out[k] = vals
        elif isinstance(vals[0], list):
            out[k] = vals
        elif isinstance(vals[0], np.ndarray):
            out[k] = torch.from_numpy(np.stack(vals, axis=0))
        else:
            out[k] = vals
    return out


# Alias used in older import sites.
_collate = collate_fn


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------

def assemble_batch(
    raw: Dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype,
    num_objects: int,
) -> Dict[str, Any]:
    """
    Map LIBEROOTPDataset / SyntheticDataset output → OTPSoftModel.forward input.

    Conversions:
      image        uint8 (B,3,H,W)  →  passed as-is; backbone preprocesses internally
      instruction  list[str]         →  passed as-is; backbone tokenises internally
      action_chunk (B, H, 7)         →  gt_action     (B, H, 7)
      ee_pose      (B, 4, 4)         →  proprioception (B, 8)
      object_names list[list]        →  object_indices (B, N_obj)  [0…N_obj-1]
    """
    from otp.utils.lie_algebra import so3_to_quat

    B = raw["image"].shape[0]

    image: torch.Tensor = raw["image"]                # (B, 3, H, W) uint8
    instruction: List[str] = raw["instruction"]       # list[str], no device move

    object_indices = torch.arange(num_objects).unsqueeze(0).expand(B, -1)

    ee = raw["ee_pose"].float()
    pos = ee[:, :3, 3]
    quat = so3_to_quat(ee[:, :3, :3])
    gripper = torch.zeros(B, 1, dtype=torch.float32)
    proprio = torch.cat([pos, quat, gripper], dim=-1)

    return {
        "image":               image.to(device),
        "instruction":         instruction,
        "object_indices":      object_indices.to(device),
        "object_point_clouds": raw["object_point_clouds"].to(device, amp_dtype),
        "proprioception":      proprio.to(device, amp_dtype),
        "grasp_affordance":    raw["grasp_affordance"].to(device, amp_dtype),
        "gt_trajectory":       raw["gt_trajectory"].to(device, amp_dtype),
        "gt_action":           raw["action_chunk"].to(device, amp_dtype),
    }


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    schedule: str,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        if schedule == "cosine":
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

class CSVLogger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._writer: Optional[csv.DictWriter] = None
        self._f = None

    def log(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            self._f = open(self._path, "w", newline="")
            self._writer = csv.DictWriter(self._f, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()


# ---------------------------------------------------------------------------
# Startup banner (§1.2)
# ---------------------------------------------------------------------------

def print_startup(cfg: Any, model: nn.Module, device: torch.device) -> None:
    import os, sys
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 72)
    print(f"[ENV]  python={sys.version.split()[0]}  cwd={os.getcwd()}")
    print(f"[ENV]  device={device}  torch={torch.__version__}")

    backbone_mode = getattr(cfg.model, "backbone_mode", "stub") \
        if hasattr(cfg, "model") else cfg.get("backbone_mode", "stub")
    precision = getattr(cfg.train, "precision", "fp32") \
        if hasattr(cfg, "train") else cfg.get("precision", "fp32")
    lr = getattr(cfg.train, "lr", 1e-3) \
        if hasattr(cfg, "train") else cfg.get("lr", 1e-3)
    bs = getattr(cfg.train, "batch_size", 4) \
        if hasattr(cfg, "train") else cfg.get("batch_size", 4)

    print(f"[CFG]  backbone_mode={backbone_mode}  precision={precision}")
    print(f"[CFG]  lr={lr}  batch={bs}")
    print(f"[MODEL] params={n_params:_}  trainable={n_trainable:_}")

    synthetic = getattr(cfg, "synthetic_data", False)
    if synthetic:
        print(f"[DATA]  SYNTHETIC  (no real dataset)")
    else:
        root = getattr(getattr(cfg, "data", cfg), "root", "?")
        aff = getattr(getattr(cfg, "data", cfg), "grasp_affordance_dir", "?")
        print(f"[DATA]  root={root}")
        print(f"[DATA]  grasp_affordance_dir={aff}")

    out_dir = getattr(cfg, "output_dir", "results")
    print(f"[OUT]   output_dir={out_dir}")
    print("=" * 72)
