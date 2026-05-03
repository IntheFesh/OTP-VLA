"""
Tests for otp/utils/lie_algebra.py

Covers:
  • so3_exp / so3_log round-trip
  • se3_to_lie / lie_to_se3 round-trip
  • Small-angle numerical stability
  • Quaternion consistency (q ↔ -q equivalence)
  • Batch dimension support
  • float16 input tolerance
  • Interface contract (M1): shape and dtype assertions

All tests run on CPU (no GPU required).
"""

import math

import pytest
import torch

from otp.utils.lie_algebra import (
    lie_to_se3,
    quat_to_so3,
    se3_to_lie,
    so3_exp,
    so3_log,
    so3_to_quat,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)
_ATOL = 1e-5


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _random_rotation(batch: int = 1, max_angle: float = math.pi - 0.15) -> torch.Tensor:
    """
    Generate rotation matrices with angle uniformly in (0, max_angle].

    Default max_angle avoids the near-π regime where testing the vector-level
    round-trip would require additional sign-disambiguation logic.
    Use max_angle=math.pi-0.01 to test near-π at the matrix level.
    """
    angles = torch.rand(batch) * max_angle + 0.01
    axes = torch.randn(batch, 3)
    axes = axes / axes.norm(dim=-1, keepdim=True)
    omega = axes * angles.unsqueeze(-1)
    return so3_exp(omega)


def _random_se3(batch: int = 1) -> torch.Tensor:
    """SE(3) matrices with random rotation and random translation in [-2, 2]."""
    T = torch.eye(4).unsqueeze(0).expand(batch, 4, 4).clone()
    T[:, :3, :3] = _random_rotation(batch)
    T[:, :3, 3] = torch.rand(batch, 3) * 4 - 2
    return T


# ---------------------------------------------------------------------------
# 1. so3_log ∘ so3_exp round-trip
# ---------------------------------------------------------------------------

class TestSO3LogExpInverse:

    def test_single_random_axis(self):
        """Random axis-angle → R → axis-angle should recover original."""
        angle = math.pi / 3
        axis = torch.tensor([[1.0, 1.0, 1.0]]) / math.sqrt(3)
        omega = axis * angle                          # (1, 3)
        R = so3_exp(omega)
        omega_rec = so3_log(R)
        assert torch.allclose(omega, omega_rec, atol=_ATOL), \
            f"Round-trip failed: {omega} vs {omega_rec}"

    def test_batch_random(self):
        """
        Batch omega round-trip for angles in [0, π).

        so3_log always returns the principal value ‖ω‖ ∈ [0, π], so the
        vector round-trip  omega == so3_log(so3_exp(omega))  is only valid
        when ‖omega‖ ∈ [0, π - ε].  Angles outside that range are tested
        at the rotation-matrix level in test_rotation_matrix_roundtrip_all_angles.
        """
        B = 64
        # Restrict to (0, π - 0.1) — the principal domain of so3_log
        angles = torch.rand(B) * (math.pi - 0.15) + 0.01
        axes = torch.randn(B, 3)
        axes = axes / axes.norm(dim=-1, keepdim=True)
        omega = axes * angles.unsqueeze(-1)
        R = so3_exp(omega)
        omega_rec = so3_log(R)
        assert torch.allclose(omega, omega_rec, atol=_ATOL), \
            f"Batch round-trip: max err = {(omega - omega_rec).abs().max().item():.3e}"

    def test_rotation_matrix_roundtrip_all_angles(self):
        """
        R = so3_exp(so3_log(R)) must hold for ALL rotations, including near π.
        This tests the rotation-matrix level, not the omega-vector level.
        """
        B = 128
        # Generate rotations at all angles including near π
        angles = torch.rand(B) * (math.pi - 0.01) + 0.01
        axes = torch.randn(B, 3)
        axes = axes / axes.norm(dim=-1, keepdim=True)
        omega = axes * angles.unsqueeze(-1)
        R = so3_exp(omega)

        omega_log = so3_log(R)
        R_rec = so3_exp(omega_log)

        max_err = (R - R_rec).abs().max().item()
        assert torch.allclose(R, R_rec, atol=_ATOL), \
            f"R-level round-trip failed (max err={max_err:.3e})"

    def test_near_pi_rotation_matrix_roundtrip(self):
        """Specifically test rotations very close to angle π."""
        B = 32
        angles = torch.rand(B) * 0.05 + (math.pi - 0.05)  # in [π-0.05, π]
        axes = torch.randn(B, 3)
        axes = axes / axes.norm(dim=-1, keepdim=True)
        omega = axes * angles.unsqueeze(-1)
        R = so3_exp(omega)

        omega_log = so3_log(R)
        R_rec = so3_exp(omega_log)

        max_err = (R - R_rec).abs().max().item()
        assert max_err < 1e-4, \
            f"Near-pi R-level round-trip failed (max err={max_err:.3e})"

    def test_identity(self):
        """Zero rotation → identity matrix."""
        omega = torch.zeros(1, 3)
        R = so3_exp(omega)
        assert torch.allclose(R, torch.eye(3).unsqueeze(0), atol=_ATOL)
        omega_rec = so3_log(R)
        assert torch.allclose(omega_rec, torch.zeros(1, 3), atol=_ATOL)

    def test_output_shape(self):
        omega = torch.randn(8, 3)
        R = so3_exp(omega)
        assert R.shape == (8, 3, 3)
        omega_back = so3_log(R)
        assert omega_back.shape == (8, 3)

    def test_result_is_rotation_matrix(self):
        """R^T R = I and det(R) = 1."""
        omega = torch.randn(16, 3) * 0.5
        R = so3_exp(omega)
        I_approx = R @ R.transpose(-1, -2)
        I = torch.eye(3, dtype=R.dtype).unsqueeze(0).expand(16, 3, 3)
        assert torch.allclose(I_approx, I, atol=_ATOL), \
            f"R not orthogonal: max err = {(I_approx - I).abs().max().item():.3e}"
        dets = torch.det(R)
        assert torch.allclose(dets, torch.ones(16), atol=_ATOL), \
            f"det(R) ≠ 1: {dets}"


# ---------------------------------------------------------------------------
# 2. se3_to_lie ∘ lie_to_se3 round-trip
# ---------------------------------------------------------------------------

class TestSE3LieInverse:

    def test_single(self):
        T = _random_se3(1)
        xi = se3_to_lie(T)
        T_rec = lie_to_se3(xi)
        assert torch.allclose(T, T_rec, atol=_ATOL), \
            f"SE3 round-trip failed:\n{T}\nvs\n{T_rec}"

    def test_batch(self):
        B = 32
        T = _random_se3(B)
        xi = se3_to_lie(T)
        T_rec = lie_to_se3(xi)
        assert torch.allclose(T, T_rec, atol=_ATOL), \
            f"Batch SE3 round-trip: max err = {(T - T_rec).abs().max().item():.3e}"

    def test_output_shapes(self):
        T = _random_se3(4)
        xi = se3_to_lie(T)
        assert xi.shape == (4, 6)
        T_back = lie_to_se3(xi)
        assert T_back.shape == (4, 4, 4)

    def test_translation_preserved(self):
        """Translation component ξ[3:6] == T[:3, 3]."""
        T = _random_se3(8)
        xi = se3_to_lie(T)
        assert torch.allclose(xi[:, 3:], T[:, :3, 3], atol=_ATOL), \
            "Translation not preserved in ξ"

    def test_bottom_row_correct(self):
        """Bottom row of reconstructed T must be [0, 0, 0, 1]."""
        T = _random_se3(8)
        T_rec = lie_to_se3(se3_to_lie(T))
        expected = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(8, 1, 4)
        assert torch.allclose(T_rec[:, 3:, :], expected, atol=_ATOL)


# ---------------------------------------------------------------------------
# 3. Small-angle numerical stability
# ---------------------------------------------------------------------------

class TestSmallAngleStability:

    @pytest.mark.parametrize("eps", [1e-4, 1e-6, 1e-7, 1e-10, 0.0])
    def test_no_nan_inf_exp(self, eps):
        """so3_exp at near-zero omega must not produce NaN/Inf."""
        omega = torch.full((4, 3), eps / math.sqrt(3))
        R = so3_exp(omega)
        assert not torch.isnan(R).any(), f"NaN in so3_exp at eps={eps}"
        assert not torch.isinf(R).any(), f"Inf in so3_exp at eps={eps}"

    @pytest.mark.parametrize("eps", [1e-4, 1e-6, 1e-7, 0.0])
    def test_no_nan_inf_log(self, eps):
        """so3_log on nearly-identity R must not produce NaN/Inf."""
        omega = torch.full((4, 3), eps / math.sqrt(3))
        R = so3_exp(omega)
        omega_rec = so3_log(R)
        assert not torch.isnan(omega_rec).any(), f"NaN in so3_log at eps={eps}"
        assert not torch.isinf(omega_rec).any(), f"Inf in so3_log at eps={eps}"

    def test_zero_rotation_roundtrip(self):
        """Exact zero rotation: exp → log → exp chain must be stable."""
        omega = torch.zeros(1, 3)
        R = so3_exp(omega)
        omega2 = so3_log(R)
        R2 = so3_exp(omega2)
        assert torch.allclose(R, R2, atol=_ATOL)

    def test_small_angle_value(self):
        """At eps = 1e-7, exp should give ≈ I + hat(omega)."""
        eps = 1e-7
        omega = torch.tensor([[eps, 0.0, 0.0]])
        R = so3_exp(omega)
        # R ≈ I + hat(omega) to first order
        expected = torch.eye(3).unsqueeze(0)
        expected[0, 1, 2] = -eps
        expected[0, 2, 1] = eps
        assert torch.allclose(R, expected, atol=1e-6), \
            f"Small-angle approximation failed:\n{R}\nvs\n{expected}"


# ---------------------------------------------------------------------------
# 4. Quaternion consistency
# ---------------------------------------------------------------------------

class TestQuatConsistency:

    def test_quat_to_so3_to_quat_random(self):
        """q → R → q' should be equivalent (q' == q or q' == -q)."""
        B = 32
        q = torch.randn(B, 4)
        q = q / q.norm(dim=-1, keepdim=True)

        R = quat_to_so3(q)
        q_rec = so3_to_quat(R)

        # q and -q represent the same rotation
        # Canonical form: w ≥ 0, so flip input q if needed
        q_canonical = torch.where(q[:, :1] < 0, -q, q)

        diff = (q_canonical - q_rec).abs()
        assert diff.max().item() < _ATOL, \
            f"quat round-trip max err = {diff.max().item():.3e}"

    def test_so3_to_quat_to_so3(self):
        """R → q → R' should be identity."""
        R = _random_rotation(16)
        q = so3_to_quat(R)
        R_rec = quat_to_so3(q)
        assert torch.allclose(R, R_rec, atol=_ATOL), \
            f"R → q → R round-trip max err = {(R - R_rec).abs().max().item():.3e}"

    def test_unit_quaternion_output(self):
        """so3_to_quat must always return unit quaternions."""
        R = _random_rotation(64)
        q = so3_to_quat(R)
        norms = q.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(64), atol=_ATOL), \
            f"Quaternion not unit: min={norms.min():.4f}, max={norms.max():.4f}"

    def test_canonical_positive_w(self):
        """so3_to_quat output must always have w ≥ 0."""
        R = _random_rotation(64)
        q = so3_to_quat(R)
        assert (q[:, 0] >= 0).all(), \
            f"Negative w in quaternion: {q[:, 0].min():.4f}"

    def test_identity_rotation(self):
        """Identity matrix → quaternion should be [1, 0, 0, 0]."""
        R = torch.eye(3).unsqueeze(0)
        q = so3_to_quat(R)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(q, expected, atol=_ATOL), f"Got: {q}"


# ---------------------------------------------------------------------------
# 5. Batch dimension support
# ---------------------------------------------------------------------------

class TestBatchDim:

    def test_so3_exp_single(self):
        omega = torch.randn(1, 3) * 0.5
        R = so3_exp(omega)
        assert R.shape == (1, 3, 3)

    def test_so3_exp_large_batch(self):
        omega = torch.randn(128, 3) * 0.5
        R = so3_exp(omega)
        assert R.shape == (128, 3, 3)

    def test_so3_log_single(self):
        R = _random_rotation(1)
        omega = so3_log(R)
        assert omega.shape == (1, 3)

    def test_se3_operations_batch(self):
        for B in [1, 4, 32]:
            T = _random_se3(B)
            xi = se3_to_lie(T)
            T_rec = lie_to_se3(xi)
            assert T_rec.shape == (B, 4, 4), f"Shape mismatch at B={B}"

    def test_quat_batch(self):
        for B in [1, 8, 64]:
            q = torch.randn(B, 4)
            q = q / q.norm(dim=-1, keepdim=True)
            R = quat_to_so3(q)
            assert R.shape == (B, 3, 3), f"Shape mismatch at B={B}"
            q_back = so3_to_quat(R)
            assert q_back.shape == (B, 4)


# ---------------------------------------------------------------------------
# 6. float16 input
# ---------------------------------------------------------------------------

class TestFloat16Input:

    def test_so3_exp_float16(self):
        """float16 input should produce float16 output without NaN."""
        omega = (torch.randn(4, 3) * 0.5).half()
        R = so3_exp(omega)
        assert R.dtype == torch.float16
        assert not torch.isnan(R.float()).any()

    def test_so3_log_float16(self):
        R = _random_rotation(4).half()
        omega = so3_log(R)
        assert omega.dtype == torch.float16
        assert not torch.isnan(omega.float()).any()

    def test_se3_float16(self):
        T = _random_se3(4).half()
        xi = se3_to_lie(T)
        assert xi.dtype == torch.float16
        T_rec = lie_to_se3(xi)
        assert T_rec.dtype == torch.float16

    def test_quat_float16(self):
        q = (torch.randn(4, 4)).half()
        q = q / q.norm(dim=-1, keepdim=True)
        R = quat_to_so3(q)
        assert R.dtype == torch.float16
        q_back = so3_to_quat(R)
        assert q_back.dtype == torch.float16
