"""
TrajectoryParser: converts predicted Lie algebra trajectories to SE(3) matrices.

Interface contract (§REPO_LAYOUT.md Stage 2):
  parse(xi: (N_obj, H, 6)) -> (N_obj, H, 4, 4)
"""

from __future__ import annotations

import torch

from otp.utils.lie_algebra import lie_to_se3


class TrajectoryParser:
    """Decodes (N_obj, H, 6) Lie algebra tensors into (N_obj, H, 4, 4) SE(3) matrices."""

    def parse(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xi: (N_obj, H, 6) Lie algebra trajectory — [omega(3) | t(3)] per step.

        Returns:
            T: (N_obj, H, 4, 4) SE(3) homogeneous transformation matrices.
        """
        N_obj, H, _ = xi.shape
        T_flat = lie_to_se3(xi.reshape(N_obj * H, 6))
        return T_flat.reshape(N_obj, H, 4, 4)
