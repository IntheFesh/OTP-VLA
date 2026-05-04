"""
Tests for otp/controllers/components/phase_scheduler.py

Covers:
  1. APPROACH → PRE_GRASP transition on correct condition.
  2. All transition signals are recorded in phase_trace before branching.
  3. Partial conditions do NOT trigger GRASP → TRANSPORT.
  4. phase_trace grows by exactly 1 per step call.
  5. max_phase_steps timeout forces DONE.
"""

import math

import numpy as np
import pytest

from otp.controllers.components.phase_scheduler import PhaseScheduler, _SIGNAL_KEYS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ee(x=0.0, y=0.0, z=0.3):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def _make_obj(x=0.0, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def _make_grasp_targets(obj_pose=None, approach_height=0.15):
    if obj_pose is None:
        obj_pose = _make_obj()
    obj_pos = obj_pose[:3, 3]

    pre_grasp = np.eye(4)
    pre_grasp[:3, 3] = obj_pos + [0, 0, approach_height]

    grasp = np.eye(4)
    grasp[:3, 3] = obj_pos.copy()

    return {
        "pre_grasp_ee_pose": pre_grasp,
        "grasp_ee_pose":     grasp,
        "release_ee_pose_offset": np.array([0.0, 0.0, approach_height]),
    }


def _step(scheduler, ee_pose, obj_pose=None, gripper=0.0, progress=0.0,
          release_xy=None, grasp_targets=None, control_step=0):
    if obj_pose is None:
        obj_pose = _make_obj()
    if grasp_targets is None:
        grasp_targets = _make_grasp_targets(obj_pose)
    if release_xy is None:
        release_xy = np.zeros(2)
    return scheduler.step(
        control_step, ee_pose, obj_pose, gripper,
        progress, grasp_targets, release_xy
    )


# ---------------------------------------------------------------------------
# 1. APPROACH → PRE_GRASP transition
# ---------------------------------------------------------------------------

class TestApproachToPreGrasp:

    def test_no_transition_when_too_far(self):
        """EE far from object → stays APPROACH."""
        sched = PhaseScheduler(xy_approach_threshold=0.04)
        # EE at (1, 0, 0.3), object at (0, 0, 0) → xy_dist = 1.0
        ee = _make_ee(x=1.0, z=0.3)
        phase = _step(sched, ee)
        assert phase == "APPROACH"

    def test_transition_when_close_and_above(self):
        """EE close in XY and above object → APPROACH → PRE_GRASP."""
        sched = PhaseScheduler(xy_approach_threshold=0.04)
        obj = _make_obj(x=0.0, y=0.0, z=0.0)
        grasp_targets = _make_grasp_targets(obj)
        # EE at (0.01, 0.01, 0.2) → xy_dist ≈ 0.014 < 0.04, z_above = 0.2
        ee = _make_ee(x=0.01, y=0.01, z=0.2)
        phase = _step(sched, ee, obj_pose=obj, grasp_targets=grasp_targets)
        assert phase == "PRE_GRASP", f"Expected PRE_GRASP but got {phase}"

    def test_no_transition_when_not_above_object(self):
        """EE below object z → z_above_object < 0.05 → no transition."""
        sched = PhaseScheduler(xy_approach_threshold=0.04)
        obj = _make_obj(x=0.0, y=0.0, z=0.5)
        grasp_targets = _make_grasp_targets(obj)
        # EE at same XY as object but BELOW it (z=0.3 < obj_z=0.5, diff=-0.2 < 0.05)
        ee = _make_ee(x=0.0, y=0.0, z=0.3)
        phase = _step(sched, ee, obj_pose=obj, grasp_targets=grasp_targets)
        assert phase == "APPROACH"


# ---------------------------------------------------------------------------
# 2. All signals computed before branching
# ---------------------------------------------------------------------------

class TestAllSignalsComputedBeforeBranching:

    def test_all_signal_keys_present_every_step(self):
        """phase_trace entry must contain exactly the required signal keys each step."""
        sched = PhaseScheduler()
        ee = _make_ee(x=1.0, z=0.3)
        for i in range(5):
            _step(sched, ee, control_step=i)

        assert len(sched.phase_trace) == 5
        for entry in sched.phase_trace:
            assert "transition_signals" in entry
            recorded = set(entry["transition_signals"].keys())
            missing = _SIGNAL_KEYS - recorded
            assert not missing, f"Missing signal keys: {missing}"

    def test_trace_fields_present(self):
        """Each trace entry must have t, phase, phase_step_count, transition_signals."""
        sched = PhaseScheduler()
        _step(sched, _make_ee(), control_step=7)
        entry = sched.phase_trace[0]
        assert entry["t"] == 7
        assert entry["phase"] == "APPROACH"
        assert "phase_step_count" in entry
        assert isinstance(entry["transition_signals"], dict)


# ---------------------------------------------------------------------------
# 3. Partial conditions do NOT trigger GRASP → TRANSPORT
# ---------------------------------------------------------------------------

class TestPartialConditionNoTransition:

    def test_grasp_transport_requires_both_gripper_and_min_steps(self):
        """
        GRASP → TRANSPORT requires:
          gripper_state > gripper_hold_threshold (0.5)
          AND gripper_recent_std < 0.1 (stable)
          AND phase_step_count >= grasp_min_steps.

        When only gripper is high but step count too low → no transition.
        """
        sched = PhaseScheduler(grasp_min_steps=10, gripper_hold_threshold=0.5)
        sched.current_phase = "GRASP"

        ee = _make_ee()
        obj = _make_obj()
        grasp_targets = _make_grasp_targets(obj)

        # Run 5 steps with high gripper (fills history with stable 0.8)
        for i in range(5):
            phase = sched.step(i, ee, obj, 0.8, 0.0, grasp_targets, np.zeros(2))

        # step_count = 4 after 5 steps, still < grasp_min_steps=10
        assert phase == "GRASP", (
            f"Should still be GRASP (step_count < 10), got {phase}"
        )

    def test_grasp_transport_fires_after_enough_steps(self):
        """After grasp_min_steps steps with stable gripper → transitions to TRANSPORT."""
        sched = PhaseScheduler(grasp_min_steps=3, gripper_hold_threshold=0.5)
        sched.current_phase = "GRASP"

        ee = _make_ee()
        obj = _make_obj()
        grasp_targets = _make_grasp_targets(obj)

        phase = "GRASP"
        for i in range(20):
            phase = sched.step(i, ee, obj, 0.9, 0.0, grasp_targets, np.zeros(2))
            if phase != "GRASP":
                break

        assert phase == "TRANSPORT", f"Expected TRANSPORT, got {phase}"


# ---------------------------------------------------------------------------
# 4. phase_trace grows by exactly 1 per step
# ---------------------------------------------------------------------------

class TestPhaseTraceLogging:

    def test_trace_grows_monotonically(self):
        """After N steps, len(phase_trace) == N."""
        sched = PhaseScheduler()
        ee = _make_ee(x=5.0)
        for i in range(20):
            _step(sched, ee, control_step=i)
            assert len(sched.phase_trace) == i + 1

    def test_trace_records_pre_transition_phase(self):
        """
        The entry logged at step T records the phase BEFORE the transition,
        not the phase after.
        """
        sched = PhaseScheduler(xy_approach_threshold=0.04)
        obj = _make_obj()
        grasp_targets = _make_grasp_targets(obj)
        # Close EE → will trigger APPROACH → PRE_GRASP on first step
        ee = _make_ee(x=0.01, y=0.01, z=0.2)
        sched.step(0, ee, obj, 0.0, 0.0, grasp_targets, np.zeros(2))

        entry = sched.phase_trace[0]
        assert entry["phase"] == "APPROACH", (
            "Trace should record 'APPROACH' (pre-transition) not the new phase"
        )

    def test_reset_clears_trace(self):
        """reset() must clear phase_trace."""
        sched = PhaseScheduler()
        _step(sched, _make_ee())
        _step(sched, _make_ee())
        sched.reset()
        assert sched.phase_trace == []
        assert sched.current_phase == "APPROACH"


# ---------------------------------------------------------------------------
# 5. max_phase_steps timeout forces DONE
# ---------------------------------------------------------------------------

class TestTimeoutForcesDone:

    def test_timeout_triggers_done(self):
        """Exceeding max_phase_steps in any phase forces transition to DONE."""
        sched = PhaseScheduler(max_phase_steps=3)
        ee = _make_ee(x=10.0)      # far from object → no organic transition
        phase = "APPROACH"
        for i in range(10):
            phase = _step(sched, ee, control_step=i)
            if phase == "DONE":
                break

        assert phase == "DONE", f"Expected DONE from timeout, got {phase}"

    def test_timeout_does_not_fire_early(self):
        """Before max_phase_steps is reached, phase must NOT be forced to DONE."""
        sched = PhaseScheduler(max_phase_steps=10)
        ee = _make_ee(x=10.0)
        for i in range(9):
            phase = _step(sched, ee, control_step=i)
        assert phase == "APPROACH"
