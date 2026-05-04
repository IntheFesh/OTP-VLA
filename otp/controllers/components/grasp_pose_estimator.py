"""
GraspPoseEstimator: computes top-down grasp poses from object pose.

Interface contract (§REPO_LAYOUT.md Stage 2):
  estimate_grasp(object_pose: (4,4)) -> dict with keys:
    pre_grasp_ee_pose:     (4, 4) EE pose at approach height above object
    grasp_ee_pose:         (4, 4) EE pose at grasp height
    release_ee_pose_offset: (3,) positional offset applied at release site
"""

from __future__ import annotations

import numpy as np


# Top-down end-effector orientation: EE z-axis points into the table (-world z).
# EE x-axis forward in world, y-axis left.
_R_TOPDOWN = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=np.float64)


class GraspPoseEstimator:
    """
    Produces three canonical EE poses for a top-down grasp strategy.

    Args:
        approach_height:     Metres above object centre for the pre-grasp hover.
        grasp_height_offset: Additional z offset applied at the grasp pose
                             (positive = higher than object centre).
    """

    def __init__(
        self,
        approach_height: float = 0.15,
        grasp_height_offset: float = 0.0,
    ) -> None:
        self.approach_height = approach_height
        self.grasp_height_offset = grasp_height_offset

    def estimate_grasp(self, object_pose: np.ndarray) -> dict:
        """
        Args:
            object_pose: (4, 4) SE(3) pose of the target object in world frame.

        Returns:
            dict with:
              pre_grasp_ee_pose     : (4, 4) hover pose above the object.
              grasp_ee_pose         : (4, 4) contact pose at the object.
              release_ee_pose_offset: (3,)   [dx, dy, dz] offset from release xy.
        """
        obj_pos = object_pose[:3, 3]

        pre_grasp_pos = obj_pos.copy()
        pre_grasp_pos[2] += self.approach_height
        pre_grasp_ee_pose = np.eye(4, dtype=np.float64)
        pre_grasp_ee_pose[:3, :3] = _R_TOPDOWN
        pre_grasp_ee_pose[:3, 3] = pre_grasp_pos

        grasp_pos = obj_pos.copy()
        grasp_pos[2] += self.grasp_height_offset
        grasp_ee_pose = np.eye(4, dtype=np.float64)
        grasp_ee_pose[:3, :3] = _R_TOPDOWN
        grasp_ee_pose[:3, 3] = grasp_pos

        release_ee_pose_offset = np.array(
            [0.0, 0.0, self.approach_height], dtype=np.float64
        )

        return {
            "pre_grasp_ee_pose": pre_grasp_ee_pose,
            "grasp_ee_pose": grasp_ee_pose,
            "release_ee_pose_offset": release_ee_pose_offset,
        }
