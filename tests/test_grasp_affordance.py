"""
Tests for otp/data/grasp_affordance.py

Coverage:
  1. Antipodal grasps respect gripper_width.
  2. Antipodal grasps have anti-aligned contact normals (within friction cone).
  3. Flat-plane mesh yields zero valid grasps.
  4. Sampling is deterministic for a fixed seed.
  5. Padding to num_grasps with all-zero rows.
  6. Bad input shapes raise ValueError.
"""

import numpy as np
import pytest

from otp.data.grasp_affordance import (
    _sample_antipodal_grasps_internal,
    sample_antipodal_grasps,
)


# ---------------------------------------------------------------------------
# Synthetic mesh fixtures
# ---------------------------------------------------------------------------

def _box_mesh(extent: float = 0.04, n_per_face: int = 80, seed: int = 0):
    """Sample n_per_face random points on each face of a centered box."""
    rng = np.random.default_rng(seed)
    e = extent / 2.0
    verts: list = []
    norms: list = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            n = np.zeros(3, dtype=np.float32)
            n[axis] = sign
            for _ in range(n_per_face):
                p = rng.uniform(-e, e, size=3).astype(np.float32)
                p[axis] = sign * e
                verts.append(p.copy())
                norms.append(n.copy())
    return np.stack(verts, 0), np.stack(norms, 0)


def _plane_mesh(n: int = 200, seed: int = 0):
    """All vertices on z=0 plane, all normals = +z (no antipodal pair)."""
    rng = np.random.default_rng(seed)
    verts = np.zeros((n, 3), dtype=np.float32)
    verts[:, :2] = rng.uniform(-0.05, 0.05, size=(n, 2))
    norms = np.tile(np.array([0, 0, 1], dtype=np.float32), (n, 1))
    return verts, norms


# ---------------------------------------------------------------------------
# 1. gripper_width invariant
# ---------------------------------------------------------------------------

class TestAntipodalWithinGripperWidth:

    def test_all_valid_contacts_within_gripper_width(self):
        v, n = _box_mesh(extent=0.04)
        out = _sample_antipodal_grasps_internal(
            v, n, num_grasps=8, gripper_width=0.08, friction_coef=0.5,
            seed=42,
        )
        valid = out["valid_mask"]
        assert valid.any(), "Box mesh should yield at least one valid grasp"
        for i in np.where(valid)[0]:
            d = float(np.linalg.norm(out["contact_a"][i] - out["contact_b"][i]))
            assert d < 0.08 + 1e-6, (
                f"Grasp {i} contact distance {d:.4f} ≥ gripper_width 0.08"
            )

    def test_grasp_position_is_midpoint_of_contacts(self):
        v, n = _box_mesh(extent=0.04)
        out = _sample_antipodal_grasps_internal(
            v, n, num_grasps=8, gripper_width=0.08, seed=42,
        )
        for i in np.where(out["valid_mask"])[0]:
            pos = out["grasps"][i, :3]
            mid = 0.5 * (out["contact_a"][i] + out["contact_b"][i])
            assert np.allclose(pos, mid, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. normals anti-aligned within friction cone
# ---------------------------------------------------------------------------

class TestAntipodalNormalsOpposite:

    def test_all_valid_normals_within_friction_cone(self):
        friction = 0.5
        v, n = _box_mesh(extent=0.04)
        out = _sample_antipodal_grasps_internal(
            v, n, num_grasps=8, gripper_width=0.08, friction_coef=friction,
            seed=42,
        )
        cos_friction = float(np.cos(np.arctan(friction)))
        for i in np.where(out["valid_mask"])[0]:
            na = out["normal_a"][i]
            nb = out["normal_b"][i]
            dot = float(np.dot(na, nb))
            assert dot <= -cos_friction + 1e-6, (
                f"Grasp {i}: normals not anti-aligned (n_a·n_b={dot:.3f}, "
                f"required ≤ {-cos_friction:.3f})"
            )


# ---------------------------------------------------------------------------
# 3. Flat plane → all invalid
# ---------------------------------------------------------------------------

class TestPaddingWhenInsufficientGrasps:

    def test_flat_plane_produces_all_invalid(self):
        v, n = _plane_mesh()
        grasps = sample_antipodal_grasps(
            v, n, num_grasps=8, gripper_width=0.08, friction_coef=0.5,
            seed=42,
        )
        valid_mask = ~np.all(grasps == 0, axis=1)
        assert not valid_mask.any(), (
            f"Flat plane should produce zero valid grasps; got "
            f"{int(valid_mask.sum())}."
        )
        assert grasps.shape == (8, 7)

    def test_partial_fill_pads_with_zero_rows(self):
        """Force a small mesh + tight gripper so only a few grasps are found."""
        v, n = _box_mesh(extent=0.005, n_per_face=10, seed=1)   # 1cm cube
        grasps = sample_antipodal_grasps(
            v, n, num_grasps=8, gripper_width=0.02, seed=7,
        )
        valid_mask = ~np.all(grasps == 0, axis=1)
        invalid = grasps[~valid_mask]
        # Every invalid row must be exactly all-zero (padding).
        for row in invalid:
            assert np.array_equal(row, np.zeros(7, dtype=row.dtype))


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------

class TestDeterministicWithSeed:

    def test_same_seed_same_grasps(self):
        v, n = _box_mesh(extent=0.04, seed=2)
        g1 = sample_antipodal_grasps(v, n, num_grasps=8, seed=123)
        g2 = sample_antipodal_grasps(v, n, num_grasps=8, seed=123)
        assert np.array_equal(g1, g2), (
            "Sampler is non-deterministic for fixed seed."
        )

    def test_different_seed_different_grasps(self):
        v, n = _box_mesh(extent=0.04, seed=2)
        g1 = sample_antipodal_grasps(v, n, num_grasps=8, seed=1)
        g2 = sample_antipodal_grasps(v, n, num_grasps=8, seed=2)
        assert not np.array_equal(g1, g2)


# ---------------------------------------------------------------------------
# 5. Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:

    def test_grasps_dtype_and_shape(self):
        v, n = _box_mesh(extent=0.04)
        g = sample_antipodal_grasps(v, n, num_grasps=8, seed=42)
        assert g.dtype == np.float32
        assert g.shape == (8, 7)

    def test_quaternion_is_unit(self):
        v, n = _box_mesh(extent=0.04)
        g = sample_antipodal_grasps(v, n, num_grasps=8, seed=42)
        valid_mask = ~np.all(g == 0, axis=1)
        for row in g[valid_mask]:
            q = row[3:]
            norm = float(np.linalg.norm(q))
            assert abs(norm - 1.0) < 1e-3, (
                f"Quaternion not unit: ||q|| = {norm:.4f}"
            )


# ---------------------------------------------------------------------------
# 6. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_bad_vertex_shape_raises(self):
        with pytest.raises(ValueError):
            sample_antipodal_grasps(
                np.zeros((10, 4), dtype=np.float32),
                np.zeros((10, 3), dtype=np.float32),
                num_grasps=4,
            )

    def test_normal_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            sample_antipodal_grasps(
                np.zeros((10, 3), dtype=np.float32),
                np.zeros((9, 3), dtype=np.float32),
                num_grasps=4,
            )
