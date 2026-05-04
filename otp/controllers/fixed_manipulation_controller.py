"""
FixedManipulationController: integrates all 6 sub-modules into a 7-DOF action.

Interface contract (§REPO_LAYOUT.md Stage 2):
  set_trajectory(xi, object_pose, release_target_xy)
  step(ee_pose, object_pose, gripper_state) -> (7,) ndarray
  reset()
  phase_trace  — property, proxies PhaseScheduler.phase_trace

Action format (LIBERO): [dx, dy, dz, dax, day, daz, gripper]
  dx/dy/dz   : Cartesian EE delta [m], clipped to ±pos_clip
  dax/day/daz: axis-angle EE delta [rad], clipped to ±ori_clip
  gripper    : exactly ±1.0 (M8 — discrete)

NaN guard (§2.3): NaN trajectory → zero action returned, phase stays APPROACH.

Note on rotation control: when targeting LIBERO's Panda OSC_POSE controller,
setting ori_clip=0.0 (no rotation delta) is recommended.  LIBERO's home
pose already has the gripper pointing down, so attempting to rotate to
_R_TOPDOWN injects a ~180-degree axis-angle delta that severely couples
into the OSC position controller, preventing convergence.

Note on gripper timing: the gripper command for the GRASP phase is gated
on `ee_at_grasp_height` — the gripper does NOT close until the EE has
actually descended to the grasp height (within `grasp_z_tolerance`).
This prevents the controller from closing on empty space above the object.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from otp.controllers.components import (
    FailureRecoveryRule,
    GraspPoseEstimator,
    GripperController,
    OSCTargetGenerator,
    PhaseScheduler,
    TrajectoryParser,
)
from otp.utils.lie_algebra import so3_log

_ZERO_ACTION = np.zeros(7, dtype=np.float64)


class FixedManipulationController:
    """
    Deterministic manipulation controller that combines:
      TrajectoryParser      → Lie algebra → SE(3)
      GraspPoseEstimator    → top-down grasp poses
      PhaseScheduler        → 6-phase state machine
      OSCTargetGenerator    → per-phase EE target
      GripperController     → discrete gripper command
      FailureRecoveryRule   → per-phase timeout

    Args:
        approach_height:        GraspPoseEstimator approach hover height [m].
        grasp_height_offset:    GraspPoseEstimator grasp z-offset [m].
        approach_height_offset: OSCTargetGenerator APPROACH target height [m].
        release_lift_offset:    OSCTargetGenerator RELEASE lift [m].
        xy_approach_threshold:  PhaseScheduler XY distance threshold [m].
        z_pregrasp_threshold:   PhaseScheduler Z distance threshold [m].
        gripper_hold_threshold: PhaseScheduler gripper hold threshold.
        grasp_min_steps:        PhaseScheduler minimum GRASP steps.
        grasp_z_tolerance:      EE-to-grasp z tolerance for GRASP→TRANSPORT
                                AND for closing the gripper [m].
        transport_xy_threshold: PhaseScheduler TRANSPORT→RELEASE XY threshold [m].
        trajectory_done_threshold: PhaseScheduler trajectory completion fraction.
        release_min_steps:      PhaseScheduler minimum RELEASE steps.
        gripper_open_threshold: PhaseScheduler gripper open threshold.
        max_phase_steps:        PhaseScheduler per-phase timeout.
        pos_clip:               Maximum absolute Cartesian delta [m].
        ori_clip:               Maximum absolute axis-angle delta [rad].
                                Set to 0.0 for LIBERO Panda — see module docstring.
    """

    def __init__(
        self,
        approach_height: float = 0.15,
        grasp_height_offset: float = 0.0,
        approach_height_offset: float = 0.10,
        release_lift_offset: float = 0.05,
        xy_approach_threshold: float = 0.05,
        z_pregrasp_threshold: float = 0.03,
        gripper_hold_threshold: float = 0.5,
        grasp_min_steps: int = 5,
        grasp_z_tolerance: float = 0.025,
        transport_xy_threshold: float = 0.05,
        trajectory_done_threshold: float = 0.9,
        release_min_steps: int = 5,
        gripper_open_threshold: float = -0.5,
        max_phase_steps: int = 200,
        pos_clip: float = 0.1,
        ori_clip: float = 0.5,
    ) -> None:
        self._trajectory_parser = TrajectoryParser()
        self._grasp_estimator = GraspPoseEstimator(
            approach_height=approach_height,
            grasp_height_offset=grasp_height_offset,
        )
        self._phase_scheduler = PhaseScheduler(
            xy_approach_threshold=xy_approach_threshold,
            z_pregrasp_threshold=z_pregrasp_threshold,
            gripper_hold_threshold=gripper_hold_threshold,
            grasp_min_steps=grasp_min_steps,
            grasp_z_tolerance=grasp_z_tolerance,
            transport_xy_threshold=transport_xy_threshold,
            trajectory_done_threshold=trajectory_done_threshold,
            release_min_steps=release_min_steps,
            gripper_open_threshold=gripper_open_threshold,
            max_phase_steps=max_phase_steps,
        )
        self._osc_gen = OSCTargetGenerator(
            approach_height_offset=approach_height_offset,
            release_lift_offset=release_lift_offset,
        )
        self._gripper_ctrl = GripperController()
        self._failure_recovery = FailureRecoveryRule()

        self._pos_clip = pos_clip
        self._ori_clip = ori_clip
        self._grasp_z_tolerance = grasp_z_tolerance

        # Runtime state.
        self._trajectory: Optional[np.ndarray] = None   # (N_obj, H, 4, 4)
        self._grasp_targets: Optional[dict] = None
        self._release_target_xy: Optional[np.ndarray] = None
        self._trajectory_step: int = 0
        self._control_step: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_trajectory(
        self,
        xi: torch.Tensor,
        object_pose: np.ndarray,
        release_target_xy: np.ndarray,
    ) -> None:
        """
        Update controller with a new predicted trajectory.

        Args:
            xi:                 (N_obj, H, 6) Lie algebra tensor from OTPHead.
            object_pose:        (4, 4) current object pose in world frame.
            release_target_xy:  (2,) [x, y] release site.
        """
        self.reset()

        if torch.isnan(xi).any() or torch.isinf(xi).any():
            return

        T = self._trajectory_parser.parse(xi)
        self._trajectory = T.detach().cpu().numpy()
        self._grasp_targets = self._grasp_estimator.estimate_grasp(object_pose)
        self._release_target_xy = np.asarray(release_target_xy, dtype=np.float64)

    def reset(self) -> None:
        """Reset all internal state to initial APPROACH configuration."""
        self._phase_scheduler.reset()
        self._failure_recovery.reset()
        self._trajectory = None
        self._grasp_targets = None
        self._release_target_xy = None
        self._trajectory_step = 0
        self._control_step = 0

    def step(
        self,
        ee_pose: np.ndarray,
        object_pose: np.ndarray,
        gripper_state: float,
    ) -> np.ndarray:
        """
        Compute 7-DOF action for one control step.

        Args:
            ee_pose:       (4, 4) current EE pose in world frame.
            object_pose:   (4, 4) current object pose in world frame.
            gripper_state: Scalar ∈ [−1, +1]; +1 = fully closed.

        Returns:
            (7,) ndarray [dx, dy, dz, dax, day, daz, gripper].
        """
        # NaN guard (§2.3).
        if self._trajectory is None or self._grasp_targets is None:
            return _ZERO_ACTION.copy()

        release_xy = (
            self._release_target_xy
            if self._release_target_xy is not None
            else np.zeros(2, dtype=np.float64)
        )

        H = self._trajectory.shape[1]
        trajectory_progress = min(self._trajectory_step / max(H, 1), 1.0)

        # Phase update.
        current_phase = self._phase_scheduler.step(
            self._control_step,
            ee_pose,
            object_pose,
            gripper_state,
            trajectory_progress,
            self._grasp_targets,
            release_xy,
        )

        # Failure recovery override.
        if self._failure_recovery.check(
            current_phase, self._phase_scheduler.phase_step_count
        ):
            current_phase = "DONE"

        # Get target EE pose from OSCTargetGenerator.
        waypoint_idx = min(self._trajectory_step, H - 1)
        target_pose = self._osc_gen.get_target(
            current_phase,
            ee_pose,
            object_pose,
            self._grasp_targets,
            self._trajectory,
            waypoint_idx,
            release_xy,
        )

        # Compute position delta.
        delta_pos = target_pose[:3, 3] - ee_pose[:3, 3]
        delta_pos = np.clip(delta_pos, -self._pos_clip, self._pos_clip)

        # Compute orientation delta via SO(3) log map.
        # Note: when ori_clip=0.0 we skip the expensive log computation.
        if self._ori_clip > 0.0:
            R_target = target_pose[:3, :3]
            R_ee = ee_pose[:3, :3]
            R_delta = R_target @ R_ee.T
            R_delta_t = torch.from_numpy(R_delta).float().unsqueeze(0)
            delta_ori = so3_log(R_delta_t).squeeze(0).numpy()
            delta_ori = np.clip(delta_ori, -self._ori_clip, self._ori_clip)
        else:
            delta_ori = np.zeros(3, dtype=np.float64)

        # Gripper command — gated on EE being at grasp height during GRASP.
        grasp_pos = self._grasp_targets["grasp_ee_pose"][:3, 3]
        ee_at_grasp_height = (
            abs(ee_pose[2, 3] - grasp_pos[2]) < self._grasp_z_tolerance
        )
        gripper_cmd = self._gripper_ctrl.get_command(
            current_phase, ee_at_grasp_height=ee_at_grasp_height,
        )

        # Advance trajectory step during TRANSPORT.
        if current_phase == "TRANSPORT":
            self._trajectory_step += 1
        self._control_step += 1

        action = np.concatenate([delta_pos, delta_ori, [gripper_cmd]])
        return action.astype(np.float64)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def phase_trace(self) -> List[Dict[str, Any]]:
        """Proxy to PhaseScheduler.phase_trace for external inspection."""
        return self._phase_scheduler.phase_trace

    @property
    def current_phase(self) -> str:
        return self._phase_scheduler.current_phase
