"""
GeometryEncoder: PointNet-style encoder for object mesh point clouds.

Theorem 2 (C3) compliance:
  Inputs are object vertex point clouds — geometry depends solely on the
  scene's 3D structure, not on the language instruction ℓ.  This encoder is
  a valid component of w(o) (task-irrelevant side information).

Architecture:
  Per-point MLP → max pool over points → per-object MLP.
  Permutation invariant by construction (max pool).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GeometryEncoder(nn.Module):
    """
    Encode (B, N_obj, N_pts, 3) point clouds into (B, N_obj, output_dim) features.

    Args:
        num_points: Expected number of points per object (informational; the
                    encoder is invariant to actual N_pts via max pool).
        output_dim: Output feature dimension D_geom.
        hidden_dim: Per-point MLP width.
    """

    def __init__(
        self,
        num_points: int = 256,
        output_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.output_dim = output_dim

        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, point_clouds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_clouds: (B, N_obj, N_pts, 3) float tensor.

        Returns:
            (B, N_obj, output_dim) float tensor.
        """
        if point_clouds.dim() != 4 or point_clouds.shape[-1] != 3:
            raise ValueError(
                f"Expected (B, N_obj, N_pts, 3); got {tuple(point_clouds.shape)}"
            )

        per_point = self.point_mlp(point_clouds)            # (B, N_obj, N_pts, H)
        pooled = per_point.max(dim=2).values                # (B, N_obj, H)
        return self.global_mlp(pooled)                      # (B, N_obj, output_dim)
