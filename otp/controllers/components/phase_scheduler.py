"""
PhaseScheduler: 6-phase grasp-transport-release state machine.

Phases (in order):
  APPROACH → PRE_GRASP → GRASP → TRANSPORT → RELEASE → DONE

CRITICAL INVARIANTS (M7, §2.3):
  1. ALL transition signals are computed and stored in `signals` dict BEFORE
     any conditional branching.  No `return` or early exit before the log append.
  2. `phase_trace` is appended with the complete signals dict every step,
     capturing the state that caused each decision.
  3. Transitions are evaluated using ONLY pre-computed signal values.
  4. GRASP→TRANSPORT requires BOTH gripper stability AND minimum step count
     (multi-condition guard against premature transitions).

Interface contract (§REPO_LAYOUT.md Stage 2):
  step(...) -> str (current phase after transition check)
  phase_trace: List[dict] — grows by exactly 1 per step call.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


VALID_PHASES = ("APPROACH", "PRE_GRASP", "GRASP", "TRANSPORT", "RELEASE", "DONE")

_SIGNAL_KEYS = frozenset({
    "xy_dist_to_object",
    "z_above_object",
    "z_dist_to_pregrasp",
    "z_dist_to_grasp",
    "gripper_state",
    "gripper_recent_std",
    "xy_dist_to_release",
    "trajectory_progress",
})


class PhaseScheduler:
    """
    Deterministic phase state machine for fixed manipulation controller.

    All transition signals (distances, gripper state, etc.) are computed
    unconditionally at the top of step() and stored in a signals dict.
    The phase_trace log is written from that dict before any if-elif chain
    examines it.  This prevents short-circuit evaluation from hiding signal
    computation bugs.

    Defaults are tuned for LIBERO Spatial OSC_POSE controller at 20Hz with
    output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5] — meaning the EE cannot
    move more than 5cm per step in Cartesian space, and PD overshoot makes
    sub-5cm threshold tracking unreliable.

    Args:
        xy_approach_threshold:    XY distance [m] to object that triggers
                                  APPROACH → PRE_GRASP.  0.10 ≈ 2× max EE step.
        approach_min_steps:       Minimum steps in APPROACH before allowing
                                  transition (lets EE settle into hover).
        pregrasp_xy_threshold:    XY distance [m] for PRE_GRASP → GRASP.
                                  Must be tight (EE precisely above object xy).
        pregrasp_min_steps:       Minimum steps in PRE_GRASP before
                                  PRE_GRASP → GRASP (lets EE align in xy then
                                  GRASP phase handles z descent).
        z_pregrasp_threshold:     Legacy threshold (kept for API compat); not
                                  used in current PRE_GRASP transition logic.
        gripper_hold_threshold:   Minimum gripper state value (−1..+1) to
                                  consider the object held.
        grasp_min_steps:          Minimum steps in GRASP before allowing
                                  GRASP → TRANSPORT (gripper settling time).
        transport_xy_threshold:   XY distance [m] to release site that triggers
                                  TRANSPORT → RELEASE.
        trajectory_done_threshold: Minimum trajectory_progress to allow release.
        release_min_steps:        Minimum steps in RELEASE before RELEASE → DONE.
        gripper_open_threshold:   Maximum gripper state to consider gripper open.
        max_phase_steps:          Global per-phase timeout; any phase stuck this
                                  many steps transitions to DONE.
    """

    def __init__(
        self,
        xy_approach_threshold: float = 0.10,
        approach_min_steps: int = 5,
        pregrasp_xy_threshold: float = 0.03,
        pregrasp_min_steps: int = 8,
        z_pregrasp_threshold: float = 0.03,
        gripper_hold_threshold: float = 0.5,
        grasp_min_steps: int = 5,
        grasp_z_tolerance: float = 0.025,
        transport_xy_threshold: float = 0.08,
        trajectory_done_threshold: float = 0.9,
        release_min_steps: int = 5,
        gripper_open_threshold: float = -0.5,
        max_phase_steps: int = 200,
    ) -> None:
        self.xy_approach_threshold = xy_approach_threshold
        self.approach_min_steps = approach_min_steps
        self.pregrasp_xy_threshold = pregrasp_xy_threshold
        self.pregrasp_min_steps = pregrasp_min_steps
        self.z_pregrasp_threshold = z_pregrasp_threshold
        self.gripper_hold_threshold = gripper_hold_threshold
        self.grasp_min_steps = grasp_min_steps
        self.grasp_z_tolerance = grasp_z_tolerance
        self.transport_xy_threshold = transport_xy_threshold
        self.trajectory_done_threshold = trajectory_done_threshold
        self.release_min_steps = release_min_steps
        self.gripper_open_threshold = gripper_open_threshold
        self.max_phase_steps = max_phase_steps

        self.current_phase: str = "APPROACH"
        self.phase_step_count: int = 0
        self.phase_trace: List[Dict[str, Any]] = []
        self._gripper_history: List[float] = []

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset to initial APPROACH phase, clearing all history."""
        self.current_phase = "APPROACH"
        self.phase_step_count = 0
        self.phase_trace = []
        self._gripper_history = []

    # ------------------------------------------------------------------
    def step(
        self,
        control_step: int,
        ee_pose: np.ndarray,
        object_pose: np.ndarray,
        gripper_state: float,
        trajectory_progress: float,
        grasp_targets: dict,
        release_target_xy: np.ndarray,
    ) -> str:
        """
        Advance the state machine by one control step.

        Args:
            control_step:       Global step counter (used only for the trace).
            ee_pose:            (4, 4) current end-effector pose in world frame.
            object_pose:        (4, 4) current object pose in world frame.
            gripper_state:      Scalar in [−1, +1]; +1 = fully closed.
            trajectory_progress: Fraction of trajectory completed ∈ [0, 1].
            grasp_targets:      Dict from GraspPoseEstimator.estimate_grasp().
            release_target_xy:  (2,) [x, y] target position for object release.

        Returns:
            Current phase string after transition check.
        """
        entry_phase = self.current_phase
        entry_step_count = self.phase_step_count

        # ------------------------------------------------------------------ #
        # 1. Compute ALL signals — no early return, no short-circuit.         #
        # ------------------------------------------------------------------ #
        ee_pos = ee_pose[:3, 3]
        obj_pos = object_pose[:3, 3]

        xy_dist_to_object = float(np.linalg.norm(ee_pos[:2] - obj_pos[:2]))
        z_above_object = float(ee_pos[2] - obj_pos[2])

        pre_grasp_pos = grasp_targets["pre_grasp_ee_pose"][:3, 3]
        grasp_pos = grasp_targets["grasp_ee_pose"][:3, 3]

        z_dist_to_pregrasp = float(abs(ee_pos[2] - pre_grasp_pos[2]))
        z_dist_to_grasp = float(abs(ee_pos[2] - grasp_pos[2]))
        # New: xy distance to grasp target (for PRE_GRASP → GRASP).
        xy_dist_to_grasp = float(np.linalg.norm(ee_pos[:2] - grasp_pos[:2]))

        self._gripper_history.append(float(gripper_state))
        if len(self._gripper_history) > 10:
            self._gripper_history.pop(0)
        gripper_recent_std = (
            float(np.std(self._gripper_history))
            if len(self._gripper_history) > 1
            else 1.0
        )

        xy_dist_to_release = float(
            np.linalg.norm(ee_pos[:2] - release_target_xy[:2])
        )

        signals: Dict[str, Any] = {
            "xy_dist_to_object":   xy_dist_to_object,
            "z_above_object":      z_above_object,
            "z_dist_to_pregrasp":  z_dist_to_pregrasp,
            "z_dist_to_grasp":     z_dist_to_grasp,
            "xy_dist_to_grasp":    xy_dist_to_grasp,
            "gripper_state":       float(gripper_state),
            "gripper_recent_std":  gripper_recent_std,
            "xy_dist_to_release":  xy_dist_to_release,
            "trajectory_progress": float(trajectory_progress),
        }

        # ------------------------------------------------------------------ #
        # 2. Log BEFORE any branching (M7 — logging must precede decisions).  #
        # ------------------------------------------------------------------ #
        self.phase_trace.append({
            "t":                  control_step,
            "phase":              entry_phase,
            "phase_step_count":   entry_step_count,
            "transition_signals": signals.copy(),
        })

        # ------------------------------------------------------------------ #
        # 3. Evaluate transitions using ONLY pre-computed signal values.      #
        # ------------------------------------------------------------------ #
        new_phase = entry_phase

        if entry_phase == "APPROACH":
            # Wait until EE is roughly above the object (xy aligned within
            # threshold, z hovering above) AND has spent at least some steps
            # in this phase (so PD-overshoot oscillations have settled).
            if (xy_dist_to_object < self.xy_approach_threshold
                    and z_above_object > 0.05
                    and entry_step_count >= self.approach_min_steps):
                new_phase = "PRE_GRASP"

        elif entry_phase == "PRE_GRASP":
            # PRE_GRASP target is the hover pose ABOVE the grasp pose.
            # Transition to GRASP when xy is precisely aligned with the grasp
            # target AND we've spent enough steps stabilising at hover.
            # (z descent happens in GRASP phase, not here.)
            if (xy_dist_to_grasp < self.pregrasp_xy_threshold
                    and entry_step_count >= self.pregrasp_min_steps):
                new_phase = "GRASP"

        elif entry_phase == "GRASP":
            # Multi-condition guard: EE must have descended to grasp height
            # AND gripper must be stable AND holding for min steps.
            # Without the z-descent check, the gripper closes in mid-air
            # before EE reaches the object, and we transport empty.
            if (z_dist_to_grasp < self.grasp_z_tolerance
                    and gripper_state > self.gripper_hold_threshold
                    and gripper_recent_std < 0.1
                    and entry_step_count >= self.grasp_min_steps):
                new_phase = "TRANSPORT"

        elif entry_phase == "TRANSPORT":
            if (trajectory_progress >= self.trajectory_done_threshold
                    and xy_dist_to_release < self.transport_xy_threshold):
                new_phase = "RELEASE"

        elif entry_phase == "RELEASE":
            if (gripper_state < self.gripper_open_threshold
                    and entry_step_count >= self.release_min_steps):
                new_phase = "DONE"

        # Global per-phase timeout (fires regardless of transition conditions).
        if entry_phase != "DONE" and entry_step_count >= self.max_phase_steps:
            new_phase = "DONE"

        # ------------------------------------------------------------------ #
        # 4. Update phase state.                                              #
        # ------------------------------------------------------------------ #
        if new_phase != entry_phase:
            self.current_phase = new_phase
            self.phase_step_count = 0
        else:
            self.phase_step_count += 1

        return self.current_phase
