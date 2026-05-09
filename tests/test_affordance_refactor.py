"""
Unit tests for load_affordance_for_objects (free function refactor).

Tests the padding/clipping logic on synthetic npz files:
  - K_npz=4 (< K=8): pad with zeros, mask=False at padded slots
  - K_npz=10 (> K=8): clip to first 8
  - P_npz=100 (< P=256): pad mesh by repeating last point
  - _N suffix stripping: "fake_object_1" -> loads "fake_object.npz"
  - _N collision: "fake_object_1" and "fake_object_2" -> same npz file

Real LIBERO objects all have K_npz=8 P_npz=256, so padding/clipping
branches are never exercised by the production data path. These
synthetic tests guard against silent regression of those code paths
(important for future RoboCasa extension where new objects may have
different grasp counts / mesh densities).
"""

import numpy as np
import pytest
from pathlib import Path

from otp.data.libero_loader import load_affordance_for_objects


# ---------------------------------------------------------------------------
# Fixture: synthetic npz directory
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_grasp_dir(tmp_path):
    """Build a synthetic grasp_affordances/ dir with 3 objects covering
    all padding/clipping branches."""
    np.random.seed(0)  # deterministic synthetic data

    # Object 1: K_npz=4 < K=8, P_npz=100 < P=256 (padding branches)
    np.savez(
        tmp_path / "small_obj.npz",
        grasps=np.random.randn(4, 7).astype(np.float32),
        valid_mask=np.array([True, True, False, True], dtype=bool),
        mesh_vertices=np.arange(300, dtype=np.float32).reshape(100, 3),
    )

    # Object 2: K_npz=10 > K=8, P_npz=300 > P=256 (clipping branches)
    np.savez(
        tmp_path / "large_obj.npz",
        grasps=np.random.randn(10, 7).astype(np.float32),
        valid_mask=np.array([True] * 10, dtype=bool),
        mesh_vertices=np.arange(900, dtype=np.float32).reshape(300, 3),
    )

    # Object 3: K_npz=8, P_npz=256 (exact match — production case)
    np.savez(
        tmp_path / "exact_obj.npz",
        grasps=np.random.randn(8, 7).astype(np.float32),
        valid_mask=np.array([True] * 8, dtype=bool),
        mesh_vertices=np.random.randn(256, 3).astype(np.float32),
    )

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_padding_branch_K_lt(synthetic_grasp_dir):
    """K_npz=4 < K=8: pad grasps[4:] with zeros, mask[4:]=False."""
    aff, mask, pc = load_affordance_for_objects(
        synthetic_grasp_dir, ["small_obj_0"], num_grasps=8, num_points=256,
    )
    assert aff.shape == (1, 8, 7)
    assert mask.shape == (1, 8)
    assert pc.shape == (1, 256, 3)

    # First 4 grasp slots should match raw npz
    raw = np.load(synthetic_grasp_dir / "small_obj.npz")
    np.testing.assert_array_equal(aff[0, :4], raw["grasps"])
    np.testing.assert_array_equal(mask[0, :4], raw["valid_mask"])

    # Padded slots [4:8] must be zeros + False
    np.testing.assert_array_equal(aff[0, 4:], np.zeros((4, 7), dtype=np.float32))
    np.testing.assert_array_equal(mask[0, 4:], np.zeros(4, dtype=bool))


def test_padding_branch_P_lt(synthetic_grasp_dir):
    """P_npz=100 < P=256: pad point_cloud[100:] by repeating last point."""
    aff, mask, pc = load_affordance_for_objects(
        synthetic_grasp_dir, ["small_obj_0"], num_grasps=8, num_points=256,
    )
    raw = np.load(synthetic_grasp_dir / "small_obj.npz")

    # First 100 points exact match
    np.testing.assert_array_equal(pc[0, :100], raw["mesh_vertices"])
    # Padded slots [100:256] are all = last point (100 + 156 repeats)
    expected_pad = np.broadcast_to(raw["mesh_vertices"][-1:], (156, 3))
    np.testing.assert_array_equal(pc[0, 100:], expected_pad)


def test_clipping_branch_K_gt(synthetic_grasp_dir):
    """K_npz=10 > K=8: keep only first 8 grasps."""
    aff, mask, pc = load_affordance_for_objects(
        synthetic_grasp_dir, ["large_obj_0"], num_grasps=8, num_points=256,
    )
    raw = np.load(synthetic_grasp_dir / "large_obj.npz")

    np.testing.assert_array_equal(aff[0], raw["grasps"][:8])
    np.testing.assert_array_equal(mask[0], raw["valid_mask"][:8])


def test_clipping_branch_P_gt(synthetic_grasp_dir):
    """P_npz=300 > P=256: keep only first 256 mesh vertices."""
    aff, mask, pc = load_affordance_for_objects(
        synthetic_grasp_dir, ["large_obj_0"], num_grasps=8, num_points=256,
    )
    raw = np.load(synthetic_grasp_dir / "large_obj.npz")
    np.testing.assert_array_equal(pc[0], raw["mesh_vertices"][:256])


def test_exact_match_no_padding(synthetic_grasp_dir):
    """K_npz=8, P_npz=256: no padding or clipping, byte-identical to npz."""
    aff, mask, pc = load_affordance_for_objects(
        synthetic_grasp_dir, ["exact_obj_0"], num_grasps=8, num_points=256,
    )
    raw = np.load(synthetic_grasp_dir / "exact_obj.npz")

    np.testing.assert_array_equal(aff[0], raw["grasps"])
    np.testing.assert_array_equal(mask[0], raw["valid_mask"])
    np.testing.assert_array_equal(pc[0], raw["mesh_vertices"])


def test_suffix_stripping(synthetic_grasp_dir):
    """Names like 'small_obj_5' should strip _5 and load small_obj.npz."""
    aff_a, mask_a, pc_a = load_affordance_for_objects(
        synthetic_grasp_dir, ["small_obj_0"], num_grasps=8, num_points=256,
    )
    aff_b, mask_b, pc_b = load_affordance_for_objects(
        synthetic_grasp_dir, ["small_obj_42"], num_grasps=8, num_points=256,
    )
    # Both should produce identical output
    np.testing.assert_array_equal(aff_a, aff_b)
    np.testing.assert_array_equal(mask_a, mask_b)
    np.testing.assert_array_equal(pc_a, pc_b)


def test_instance_collision(synthetic_grasp_dir):
    """Two instance names mapping to same base — both produce same data."""
    aff, mask, pc = load_affordance_for_objects(
        synthetic_grasp_dir,
        ["small_obj_1", "small_obj_2"],
        num_grasps=8, num_points=256,
    )
    assert aff.shape == (2, 8, 7)
    np.testing.assert_array_equal(aff[0], aff[1])
    np.testing.assert_array_equal(mask[0], mask[1])
    np.testing.assert_array_equal(pc[0], pc[1])


def test_missing_object_raises(synthetic_grasp_dir):
    """Nonexistent base npz raises FileNotFoundError with informative message."""
    with pytest.raises(FileNotFoundError, match="Grasp affordance file missing"):
        load_affordance_for_objects(
            synthetic_grasp_dir, ["nonexistent_obj_1"],
            num_grasps=8, num_points=256,
        )


def test_batch_consistency(synthetic_grasp_dir):
    """Batched call vs N single calls produce same output."""
    names = ["small_obj_0", "large_obj_0", "exact_obj_0"]
    batched = load_affordance_for_objects(
        synthetic_grasp_dir, names, num_grasps=8, num_points=256,
    )

    individual = []
    for n in names:
        individual.append(load_affordance_for_objects(
            synthetic_grasp_dir, [n], num_grasps=8, num_points=256,
        ))

    for i in range(3):
        np.testing.assert_array_equal(batched[0][i], individual[i][0][0])
        np.testing.assert_array_equal(batched[1][i], individual[i][1][0])
        np.testing.assert_array_equal(batched[2][i], individual[i][2][0])
