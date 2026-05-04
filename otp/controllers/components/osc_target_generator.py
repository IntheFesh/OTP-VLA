"""
OSCTargetGenerator: returns a target EE pose for each manipulation phase.

Design rule (M7): one explicit if-elif branch per phase.  No unified formula.
Each branch documents its geometric invariant.

Interface contract (§REPO_LAYOUT.md Stage 2):
  get_target(phase, ee_pose, object_pose, grasp_targets,
             trajectory_waypoints, waypoint_idx, release_target_xy) -> (4, 4)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Top-down orientation reused for approach / release targets.
_R_TOPDOWN = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=np.float64)


class OSCTargetGenerator:
    """
    Produces a 4×4 SE(3) target EE pose for the current manipulation phase.

    The returned pose is used by FixedManipulationController to compute
    6-DOF deltas (position + axis-angle) for the LIBERO action vector.

    Args:
        approach_height_offset: Metres above the object to hover during APPROACH.
        release_lift_offset:    Metres above the release table z for RELEASE
                                target.  Object is dropped from this height
                                onto the receptacle.
        release_table_z:        Absolute world-frame z [m] of the release
                                receptacle's top surface.  Used as the base
                                from which release_lift_offset is added when
                                computing the RELEASE target z.  For LIBERO
                                Spatial the plate top is at ~0.92 m.
    """

    def __init__(
        self,
        approach_height_offset: float = 0.10,
        release_lift_offset: float = 0.05,
        release_table_z: float = 0.92,
    ) -> None:
        self.approach_height_offset = approach_height_offset
        self.release_lift_offset = release_lift_offset
        self.release_table_z = release_table_z

    # ------------------------------------------------------------------
    def get_target(
        self,
        phase: str,
        ee_pose: np.ndarray,
        object_pose: np.ndarray,
        grasp_targets: dict,
        trajectory_waypoints: Optional[np.ndarray],
        waypoint_idx: int,
        release_target_xy: np.ndarray,
    ) -> np.ndarray:
        """
        Return target EE pose (4, 4) for the given phase.

        Args:
            phase:                Current phase string.
            ee_pose:              (4, 4) current EE pose in world frame.
            object_pose:          (4, 4) current object pose in world frame.
            grasp_targets:        Dict from GraspPoseEstimator.estimate_grasp().
            trajectory_waypoints: (N_obj, H, 4, 4) or None; first object's waypoints used.
            waypoint_idx:         Index into the H dimension to select as current target.
            release_target_xy:    (2,) [x, y] target position for object release.

        Returns:
            (4, 4) SE(3) target pose.
        """
        if phase == "APPROACH":
            # Invariant: target is directly above the object XY at approach height.
            target = np.eye(4, dtype=np.float64)
            target[:3, :3] = _R_TOPDOWN
            target[:2, 3] = object_pose[:2, 3]
            target[2, 3] = object_pose[2, 3] + self.approach_height_offset

        elif phase == "PRE_GRASP":
            # Invariant: target == pre_grasp_ee_pose from GraspPoseEstimator.
            target = grasp_targets["pre_grasp_ee_pose"].copy()

        elif phase == "GRASP":
            # Invariant: hold at grasp_ee_pose; EE descends to grasp height
            # while gripper is gated open until EE.z reaches grasp_pose.z.
            target = grasp_targets["grasp_ee_pose"].copy()

        elif phase == "TRANSPORT":
            # Invariant: follow trajectory waypoints for the first tracked object.
            if trajectory_waypoints is not None and len(trajectory_waypoints) > 0:
                idx = min(waypoint_idx, trajectory_waypoints.shape[1] - 1)
                target = trajectory_waypoints[0, idx].copy()
            else:
                # Fallback: stay at current EE pose.
                target = ee_pose.copy()

        elif phase == "RELEASE":
            # Invariant: hover at a FIXED absolute height above the release
            # receptacle.  Using a fixed world-frame z (release_table_z +
            # release_lift_offset) prevents the EE from continuously rising
            # — the previous "ee_pose.z + lift_offset" formula caused the EE
            # to spiral upward indefinitely, never depositing the object.
            target = np.eye(4, dtype=np.float64)
            target[:3, :3] = _R_TOPDOWN
            target[:2, 3] = release_target_xy[:2]
            target[2, 3] = self.release_table_z + self.release_lift_offset

        else:
            # DONE or unknown: hold current pose.
            target = ee_pose.copy()

        return target
