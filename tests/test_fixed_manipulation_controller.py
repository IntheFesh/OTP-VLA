"""
Tests for otp/controllers/fixed_manipulation_controller.py

Covers:
  1. step() returns action of shape (7,).
  2. Gripper element is exactly ±1 (M8 — discrete).
  3. NaN trajectory → zero action returned (§2.3 guard).
  4. phase_trace is accessible and grows each step.
  5. reset() clears state back to APPROACH and empty trace.
  6. Continuous steps advance trajectory_step in TRANSPORT phase.
"""

import numpy as np
import pytest
import torch

from otp.controllers.fixed_manipulation_controller import FixedManipulationController

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pose(x=0.0, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def _make_valid_trajectory(N_obj=2, H=4) -> torch.Tensor:
    """Return a simple Lie algebra trajectory tensor (N_obj, H, 6) with no NaN."""
    return torch.randn(N_obj, H, 6) * 0.1


def _make_controller(**kwargs) -> FixedManipulationController:
    return FixedManipulationController(
        xy_approach_threshold=0.04,
        z_pregrasp_threshold=0.02,
        grasp_min_steps=3,
        max_phase_steps=50,
        **kwargs,
    )


def _set_and_step(ctrl, ee_pose=None, obj_pose=None, xi=None, gripper=0.0,
                  release_xy=None):
    if xi is None:
        xi = _make_valid_trajectory()
    if obj_pose is None:
        obj_pose = _make_pose(5.0, 5.0, 0.0)   # far from EE → APPROACH stays
    if release_xy is None:
        release_xy = np.array([1.0, 1.0])
    ctrl.set_trajectory(xi, obj_pose, release_xy)

    if ee_pose is None:
        ee_pose = _make_pose(0.0, 0.0, 0.3)
    return ctrl.step(ee_pose, obj_pose, gripper)


# ---------------------------------------------------------------------------
# 1. Action shape
# ---------------------------------------------------------------------------

class TestActionShape:

    def test_action_is_7d(self):
        ctrl = _make_controller()
        action = _set_and_step(ctrl)
        assert action.shape == (7,), f"Expected (7,), got {action.shape}"

    def test_action_dtype_float(self):
        ctrl = _make_controller()
        action = _set_and_step(ctrl)
        assert action.dtype in (np.float32, np.float64), \
            f"Unexpected dtype: {action.dtype}"

    def test_multiple_steps_always_7d(self):
        ctrl = _make_controller()
        xi = _make_valid_trajectory()
        obj = _make_pose(5.0, 5.0, 0.0)
        ctrl.set_trajectory(xi, obj, np.array([1.0, 1.0]))
        ee = _make_pose(0.0, 0.0, 0.3)
        for _ in range(10):
            action = ctrl.step(ee, obj, 0.0)
            assert action.shape == (7,)


# ---------------------------------------------------------------------------
# 2. Gripper discreteness (M8)
# ---------------------------------------------------------------------------

class TestGripperDiscrete:

    def test_gripper_is_exactly_plus_or_minus_one(self):
        ctrl = _make_controller()
        action = _set_and_step(ctrl)
        gripper = action[6]
        assert gripper in (-1.0, 1.0), f"Gripper must be ±1, got {gripper}"

    def test_gripper_open_in_approach(self):
        """APPROACH phase must have gripper open (−1)."""
        ctrl = _make_controller()
        # Object far away → controller stays in APPROACH.
        action = _set_and_step(ctrl, ee_pose=_make_pose(0.0, 0.0, 0.3),
                               obj_pose=_make_pose(5.0, 5.0, 0.0))
        assert action[6] == -1.0, f"APPROACH gripper must be -1, got {action[6]}"

    def test_gripper_stays_discrete_across_phases(self):
        """Across all steps, gripper must remain in {-1.0, +1.0}."""
        ctrl = _make_controller()
        xi = _make_valid_trajectory()
        obj = _make_pose(5.0, 5.0)
        ctrl.set_trajectory(xi, obj, np.array([1.0, 1.0]))
        ee = _make_pose(0.0, 0.0, 0.3)
        for _ in range(30):
            action = ctrl.step(ee, obj, 0.8)
            assert action[6] in (-1.0, 1.0), \
                f"Gripper must be ±1, got {action[6]}"


# ---------------------------------------------------------------------------
# 3. NaN guard (§2.3)
# ---------------------------------------------------------------------------

class TestNaNGuard:

    def test_nan_trajectory_returns_zero_action(self):
        ctrl = _make_controller()
        xi_nan = torch.full((2, 4, 6), float("nan"))
        obj = _make_pose()
        ctrl.set_trajectory(xi_nan, obj, np.zeros(2))
        action = ctrl.step(_make_pose(), obj, 0.0)
        assert np.allclose(action, np.zeros(7)), \
            f"NaN trajectory should give zero action, got {action}"

    def test_inf_trajectory_returns_zero_action(self):
        ctrl = _make_controller()
        xi_inf = torch.full((2, 4, 6), float("inf"))
        obj = _make_pose()
        ctrl.set_trajectory(xi_inf, obj, np.zeros(2))
        action = ctrl.step(_make_pose(), obj, 0.0)
        assert np.allclose(action, np.zeros(7))

    def test_no_trajectory_set_returns_zero_action(self):
        """Calling step() before set_trajectory() must return zeros."""
        ctrl = _make_controller()
        action = ctrl.step(_make_pose(), _make_pose(), 0.0)
        assert np.allclose(action, np.zeros(7))


# ---------------------------------------------------------------------------
# 4. phase_trace accessible and grows
# ---------------------------------------------------------------------------

class TestPhaseTraceAccessible:

    def test_trace_grows_each_step(self):
        ctrl = _make_controller()
        xi = _make_valid_trajectory()
        obj = _make_pose(5.0, 5.0)
        ctrl.set_trajectory(xi, obj, np.array([1.0, 1.0]))
        ee = _make_pose(0.0, 0.0, 0.3)
        for i in range(5):
            ctrl.step(ee, obj, 0.0)
            assert len(ctrl.phase_trace) == i + 1

    def test_trace_contains_required_fields(self):
        ctrl = _make_controller()
        xi = _make_valid_trajectory()
        obj = _make_pose(5.0, 5.0)
        ctrl.set_trajectory(xi, obj, np.array([1.0, 1.0]))
        ctrl.step(_make_pose(), obj, 0.0)
        entry = ctrl.phase_trace[0]
        assert "t" in entry
        assert "phase" in entry
        assert "transition_signals" in entry

    def test_nan_trajectory_does_not_log_trace(self):
        """NaN trajectory → step returns zeros without advancing the trace."""
        ctrl = _make_controller()
        xi_nan = torch.full((2, 4, 6), float("nan"))
        ctrl.set_trajectory(xi_nan, _make_pose(), np.zeros(2))
        ctrl.step(_make_pose(), _make_pose(), 0.0)
        assert len(ctrl.phase_trace) == 0, \
            "NaN trajectory guard should prevent phase scheduler from running"


# ---------------------------------------------------------------------------
# 5. reset() clears state
# ---------------------------------------------------------------------------

class TestResetClearsState:

    def test_reset_clears_phase_trace(self):
        ctrl = _make_controller()
        xi = _make_valid_trajectory()
        obj = _make_pose(5.0, 5.0)
        ctrl.set_trajectory(xi, obj, np.array([1.0, 1.0]))
        for _ in range(5):
            ctrl.step(_make_pose(), obj, 0.0)
        ctrl.reset()
        assert ctrl.phase_trace == []

    def test_reset_restores_approach_phase(self):
        ctrl = _make_controller()
        xi = _make_valid_trajectory()
        obj = _make_pose(5.0, 5.0)
        ctrl.set_trajectory(xi, obj, np.array([1.0, 1.0]))
        for _ in range(5):
            ctrl.step(_make_pose(), obj, 0.0)
        ctrl.reset()
        assert ctrl.current_phase == "APPROACH"

    def test_reset_clears_trajectory(self):
        """After reset, step() with no new trajectory should return zeros."""
        ctrl = _make_controller()
        xi = _make_valid_trajectory()
        obj = _make_pose(5.0, 5.0)
        ctrl.set_trajectory(xi, obj, np.array([1.0, 1.0]))
        ctrl.step(_make_pose(), obj, 0.0)
        ctrl.reset()
        action = ctrl.step(_make_pose(), obj, 0.0)
        assert np.allclose(action, np.zeros(7)), \
            "After reset without new trajectory, action should be zeros"

    def test_set_trajectory_implicitly_resets(self):
        """set_trajectory() calls reset() so trace starts fresh."""
        ctrl = _make_controller()
        xi = _make_valid_trajectory()
        obj = _make_pose(5.0, 5.0)
        ctrl.set_trajectory(xi, obj, np.array([1.0, 1.0]))
        for _ in range(3):
            ctrl.step(_make_pose(), obj, 0.0)
        # Set a new trajectory → should reset trace.
        ctrl.set_trajectory(_make_valid_trajectory(), obj, np.array([1.0, 1.0]))
        assert ctrl.phase_trace == []


# ---------------------------------------------------------------------------
# 6. Position deltas are bounded
# ---------------------------------------------------------------------------

class TestActionBounds:

    def test_pos_deltas_clipped(self):
        """Position deltas must be within ±pos_clip."""
        ctrl = _make_controller(pos_clip=0.1)
        # Place EE very far from target to maximise delta.
        xi = _make_valid_trajectory()
        obj = _make_pose(0.0, 0.0, 0.0)
        ctrl.set_trajectory(xi, obj, np.array([0.0, 0.0]))
        ee = _make_pose(100.0, 100.0, 100.0)  # extreme position
        action = ctrl.step(ee, obj, 0.0)
        assert (np.abs(action[:3]) <= 0.1 + 1e-9).all(), \
            f"Position deltas exceed pos_clip: {action[:3]}"
