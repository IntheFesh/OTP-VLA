"""
Tests for otp/controllers/components/osc_target_generator.py

Covers:
  1. APPROACH target is above object (correct XY, higher Z).
  2. PRE_GRASP target equals pre_grasp_ee_pose from grasp_targets.
  3. GRASP target equals grasp_ee_pose from grasp_targets.
  4. TRANSPORT target comes from the trajectory waypoints.
  5. RELEASE target is above release_target_xy.
  6. DONE target equals current EE pose (stay still).
"""

import numpy as np
import pytest

from otp.controllers.components.osc_target_generator import OSCTargetGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pose(x=0.0, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def _make_grasp_targets(approach_height=0.15):
    pre_grasp = _make_pose(0.1, 0.2, 0.5)
    grasp = _make_pose(0.1, 0.2, 0.0)
    return {
        "pre_grasp_ee_pose":      pre_grasp,
        "grasp_ee_pose":          grasp,
        "release_ee_pose_offset": np.array([0.0, 0.0, approach_height]),
    }


def _make_trajectory(N_obj=1, H=5):
    waypoints = np.stack([np.eye(4)] * H)              # (H, 4, 4)
    for i in range(H):
        waypoints[i, :3, 3] = [float(i), 0.0, 0.5]
    return waypoints[np.newaxis].repeat(N_obj, axis=0)  # (N_obj, H, 4, 4)


# ---------------------------------------------------------------------------
# 1. APPROACH target
# ---------------------------------------------------------------------------

class TestApproachTarget:

    def test_target_above_object(self):
        gen = OSCTargetGenerator(approach_height_offset=0.10)
        object_pose = _make_pose(0.3, 0.4, 0.2)
        target = gen.get_target(
            "APPROACH",
            ee_pose=_make_pose(),
            object_pose=object_pose,
            grasp_targets=_make_grasp_targets(),
            trajectory_waypoints=None,
            waypoint_idx=0,
            release_target_xy=np.zeros(2),
        )
        # Z must be above object z.
        assert target[2, 3] > object_pose[2, 3], "APPROACH target must be above object"
        assert abs(target[2, 3] - (object_pose[2, 3] + 0.10)) < 1e-6

    def test_target_xy_matches_object(self):
        gen = OSCTargetGenerator(approach_height_offset=0.10)
        object_pose = _make_pose(0.5, -0.3, 0.1)
        target = gen.get_target(
            "APPROACH",
            ee_pose=_make_pose(),
            object_pose=object_pose,
            grasp_targets=_make_grasp_targets(),
            trajectory_waypoints=None,
            waypoint_idx=0,
            release_target_xy=np.zeros(2),
        )
        assert np.allclose(target[:2, 3], object_pose[:2, 3], atol=1e-6), \
            f"APPROACH target XY should match object XY: {target[:2, 3]} vs {object_pose[:2, 3]}"


# ---------------------------------------------------------------------------
# 2. PRE_GRASP target
# ---------------------------------------------------------------------------

class TestPreGraspTarget:

    def test_target_equals_pregrasp_pose(self):
        gen = OSCTargetGenerator()
        grasp_targets = _make_grasp_targets()
        target = gen.get_target(
            "PRE_GRASP",
            ee_pose=_make_pose(),
            object_pose=_make_pose(),
            grasp_targets=grasp_targets,
            trajectory_waypoints=None,
            waypoint_idx=0,
            release_target_xy=np.zeros(2),
        )
        assert np.allclose(target, grasp_targets["pre_grasp_ee_pose"], atol=1e-6), \
            "PRE_GRASP target must equal grasp_targets['pre_grasp_ee_pose']"


# ---------------------------------------------------------------------------
# 3. GRASP target
# ---------------------------------------------------------------------------

class TestGraspTarget:

    def test_target_equals_grasp_pose(self):
        gen = OSCTargetGenerator()
        grasp_targets = _make_grasp_targets()
        target = gen.get_target(
            "GRASP",
            ee_pose=_make_pose(),
            object_pose=_make_pose(),
            grasp_targets=grasp_targets,
            trajectory_waypoints=None,
            waypoint_idx=0,
            release_target_xy=np.zeros(2),
        )
        assert np.allclose(target, grasp_targets["grasp_ee_pose"], atol=1e-6), \
            "GRASP target must equal grasp_targets['grasp_ee_pose']"


# ---------------------------------------------------------------------------
# 4. TRANSPORT target
# ---------------------------------------------------------------------------

class TestTransportTarget:

    def test_target_from_trajectory_waypoint(self):
        gen = OSCTargetGenerator()
        traj = _make_trajectory(N_obj=1, H=5)
        # Waypoint at index 3 has position [3, 0, 0.5].
        traj[0, 3, :3, 3] = [1.5, 2.5, 0.8]
        target = gen.get_target(
            "TRANSPORT",
            ee_pose=_make_pose(),
            object_pose=_make_pose(),
            grasp_targets=_make_grasp_targets(),
            trajectory_waypoints=traj,
            waypoint_idx=3,
            release_target_xy=np.zeros(2),
        )
        assert np.allclose(target[:3, 3], [1.5, 2.5, 0.8], atol=1e-6), \
            f"TRANSPORT target position wrong: {target[:3, 3]}"

    def test_waypoint_idx_clamped_to_last(self):
        """waypoint_idx > H-1 should clamp to the last waypoint."""
        gen = OSCTargetGenerator()
        traj = _make_trajectory(N_obj=1, H=4)
        traj[0, 3, :3, 3] = [9.0, 0.0, 0.0]   # last waypoint
        target = gen.get_target(
            "TRANSPORT",
            ee_pose=_make_pose(),
            object_pose=_make_pose(),
            grasp_targets=_make_grasp_targets(),
            trajectory_waypoints=traj,
            waypoint_idx=100,               # out of bounds
            release_target_xy=np.zeros(2),
        )
        assert np.allclose(target[:3, 3], [9.0, 0.0, 0.0], atol=1e-6)

    def test_fallback_to_ee_when_no_trajectory(self):
        """No trajectory → TRANSPORT target == current EE pose."""
        gen = OSCTargetGenerator()
        ee = _make_pose(1.0, 2.0, 3.0)
        target = gen.get_target(
            "TRANSPORT",
            ee_pose=ee,
            object_pose=_make_pose(),
            grasp_targets=_make_grasp_targets(),
            trajectory_waypoints=None,
            waypoint_idx=0,
            release_target_xy=np.zeros(2),
        )
        assert np.allclose(target, ee, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. RELEASE target
# ---------------------------------------------------------------------------

class TestReleaseTarget:

    def test_target_xy_at_release_position(self):
        gen = OSCTargetGenerator(release_lift_offset=0.05)
        release_xy = np.array([0.7, -0.4])
        ee = _make_pose(0.0, 0.0, 0.3)
        target = gen.get_target(
            "RELEASE",
            ee_pose=ee,
            object_pose=_make_pose(),
            grasp_targets=_make_grasp_targets(),
            trajectory_waypoints=None,
            waypoint_idx=0,
            release_target_xy=release_xy,
        )
        assert np.allclose(target[:2, 3], release_xy, atol=1e-6), \
            f"RELEASE target XY should match release_target_xy: {target[:2, 3]}"

    def test_target_z_lifted_above_ee(self):
        """RELEASE target Z must be above current EE Z by release_lift_offset."""
        gen = OSCTargetGenerator(release_lift_offset=0.08)
        ee = _make_pose(0.0, 0.0, 0.25)
        target = gen.get_target(
            "RELEASE",
            ee_pose=ee,
            object_pose=_make_pose(),
            grasp_targets=_make_grasp_targets(),
            trajectory_waypoints=None,
            waypoint_idx=0,
            release_target_xy=np.zeros(2),
        )
        assert abs(target[2, 3] - (ee[2, 3] + 0.08)) < 1e-6, \
            f"RELEASE Z should be ee_z + lift_offset: {target[2, 3]} vs {ee[2, 3] + 0.08}"


# ---------------------------------------------------------------------------
# 6. DONE target
# ---------------------------------------------------------------------------

class TestDoneTarget:

    def test_done_target_equals_ee_pose(self):
        gen = OSCTargetGenerator()
        ee = _make_pose(1.2, -0.5, 0.6)
        target = gen.get_target(
            "DONE",
            ee_pose=ee,
            object_pose=_make_pose(),
            grasp_targets=_make_grasp_targets(),
            trajectory_waypoints=None,
            waypoint_idx=0,
            release_target_xy=np.zeros(2),
        )
        assert np.allclose(target, ee, atol=1e-6), \
            "DONE target must equal current EE pose (stay still)"
