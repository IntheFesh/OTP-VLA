"""
SO(3) / SE(3) Lie algebra utilities.

Parameterisation:  ξ ∈ R^6  =  [ω (3), t (3)]
  ω  is the so(3) element (axis-angle, ‖ω‖ = rotation angle in radians)
  t  is the translation

All functions:
  • Accept float16 input (cast to float32 internally, cast result back).
  • Support arbitrary batch prefix shapes — flatten to (B, ...) then reshape.
  • Handle the small-angle singularity (‖ω‖ < _EPS) via Taylor expansion.

Interface contract (§REPO_LAYOUT §otp/utils/lie_algebra.py):
  so3_exp(so3_log(R))    ≈ R     (atol 1e-5)
  lie_to_se3(se3_to_lie(T)) ≈ T  (atol 1e-5)
"""

import math

import torch
import torch.nn.functional as F

# Threshold below which Taylor expansion is used instead of the exact formula.
# At ‖ω‖ = 1e-6 the cos/sin branch gives ~1 ULP error on float32.
_EPS = 1e-6

# Angle range within which the skew-symmetric formula becomes numerically
# unreliable (sin θ → 0). Use the symmetric-part extraction instead.
_NEAR_PI = math.pi - 0.1  # ≈ 3.04 rad


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _promote(x: torch.Tensor) -> tuple[torch.Tensor, torch.dtype]:
    """Cast float16 → float32 for computation; return (promoted, original_dtype)."""
    orig = x.dtype
    if orig == torch.float16:
        return x.float(), orig
    return x, orig


def _hat(omega: torch.Tensor) -> torch.Tensor:
    """
    Build skew-symmetric matrices from axis-angle vectors.

    Args:
        omega: (..., 3)
    Returns:
        (..., 3, 3)  skew-symmetric
    """
    *batch, _ = omega.shape
    w1, w2, w3 = omega[..., 0], omega[..., 1], omega[..., 2]
    z = torch.zeros_like(w1)
    rows = [
        torch.stack([ z,  -w3,  w2], dim=-1),
        torch.stack([ w3,   z, -w1], dim=-1),
        torch.stack([-w2,  w1,   z], dim=-1),
    ]
    return torch.stack(rows, dim=-2)  # (..., 3, 3)


def _vee(Omega: torch.Tensor) -> torch.Tensor:
    """
    Extract axis-angle vector from skew-symmetric matrix.

    Args:
        Omega: (..., 3, 3)
    Returns:
        (..., 3)
    """
    return torch.stack([
        Omega[..., 2, 1],
        Omega[..., 0, 2],
        Omega[..., 1, 0],
    ], dim=-1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues formula: axis-angle → rotation matrix.

    Args:
        omega: (..., 3)  axis-angle vector (direction = axis, magnitude = angle in rad)
    Returns:
        (..., 3, 3)  rotation matrix in SO(3)
    """
    omega, orig_dtype = _promote(omega)
    *batch, _ = omega.shape

    theta = omega.norm(dim=-1, keepdim=True)          # (..., 1)
    theta2 = (theta * theta).squeeze(-1)               # (...,)

    # A = sin(θ)/θ,  B = (1 - cos(θ))/θ²
    # Taylor:  A ≈ 1 - θ²/6,  B ≈ 1/2 - θ²/24
    safe_theta = theta.squeeze(-1).clamp(min=_EPS)
    A_exact = torch.sin(safe_theta) / safe_theta
    B_exact = (1.0 - torch.cos(safe_theta)) / (safe_theta * safe_theta)

    A_taylor = 1.0 - theta2 / 6.0
    B_taylor = 0.5 - theta2 / 24.0

    small = (theta.squeeze(-1) < _EPS)
    A = torch.where(small, A_taylor, A_exact)          # (...,)
    B = torch.where(small, B_taylor, B_exact)          # (...,)

    Omega = _hat(omega)                                # (..., 3, 3)
    I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(*batch, 3, 3)
    # R = I + A * Ω + B * Ω²
    R = I + A[..., None, None] * Omega + B[..., None, None] * (Omega @ Omega)

    return R.to(orig_dtype)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """
    Matrix logarithm of a rotation matrix → axis-angle vector.

    Key design decision — use atan2 for θ, not acos:
      sin(θ) = ‖vee(R − R^T)‖ / 2   (skew part, accurate for all θ because
               the dominant (1−cos θ) n⊗n terms cancel in R − R^T)
      cos(θ) = (tr(R) − 1) / 2
      θ = atan2(sin_θ, cos_θ)        (no cancellation near θ = 0 or π)

    Axis extraction:
      • θ < _EPS    (near identity) : Taylor  C ≈ ½ + θ²/12
      • θ > _NEAR_PI (near π)       : symmetric-part outer product n⊗n
      • otherwise                   : standard  ω = C · vee(R − R^T)

    Args:
        R: (..., 3, 3)  rotation matrix
    Returns:
        (..., 3)  axis-angle vector, ‖ω‖ ∈ [0, π]
    """
    R, orig_dtype = _promote(R)
    *batch, _, _ = R.shape

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    skew  = R - R.transpose(-1, -2)                        # (..., 3, 3)
    vee_skew = _vee(skew)                                  # (..., 3) = 2 sin(θ) n

    # sin(θ) from the skew part — accurate even near θ = π because the
    # large (1−cos θ) n⊗n terms cancel exactly in R − R^T.
    sin_theta = vee_skew.norm(dim=-1) / 2.0                # (...,)

    # cos(θ) from the trace; combine with sin via atan2 to avoid acos
    # catastrophic cancellation near θ = π.
    cos_theta_raw = (trace - 1.0) / 2.0
    cos_theta = cos_theta_raw.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.atan2(sin_theta, cos_theta)              # (...,) ∈ [0, π], accurate

    # ------------------------------------------------------------------
    # Branch 1 — standard axis extraction (θ not near 0 or π)
    # C = θ / (2 sin θ);  Taylor fallback for θ ≈ 0
    # ------------------------------------------------------------------
    safe_sin = sin_theta.clamp(min=_EPS)
    C_exact = theta / (2.0 * safe_sin)
    C_taylor = 0.5 + (theta * theta) / 12.0

    small = theta < _EPS
    C = torch.where(small, C_taylor, C_exact)

    omega_standard = C[..., None] * vee_skew               # (..., 3)

    # ------------------------------------------------------------------
    # Branch 2 — near π: recover axis via the unbiased outer product
    #
    #   (R + R^T)/2 = cos(θ) I + (1−cos θ) n⊗n
    #   → n⊗n = [(R+R^T)/2 − cos(θ) I] / (1−cos θ)
    #
    # At θ = π: (1−cos θ) = 2  (well-conditioned denominator).
    # Pick the column of n⊗n with largest diagonal n_j²,
    # normalise → ±n.  Resolve sign from vee_skew.
    # ------------------------------------------------------------------
    I = torch.eye(3, dtype=R.dtype, device=R.device).expand(*batch, 3, 3)
    sym_half = (R + R.transpose(-1, -2)) * 0.5
    one_minus_cos = (1.0 - cos_theta).clamp(min=_EPS)      # ≈ 2 at θ = π
    nn_T = (sym_half - cos_theta[..., None, None] * I) / one_minus_cos[..., None, None]

    diag_sq = torch.diagonal(nn_T, dim1=-2, dim2=-1).clamp(min=0.0)
    best_j   = diag_sq.argmax(dim=-1)
    idx_exp  = best_j[..., None, None].expand(*batch, 3, 1)
    col      = nn_T.gather(-1, idx_exp).squeeze(-1)         # n_j · n
    axis_pi  = F.normalize(col, dim=-1)                     # ±n

    # Sign from skew part: vee_skew = 2 sin(θ) n, non-zero unless θ = π exactly.
    dot_pi   = (axis_pi * vee_skew).sum(dim=-1, keepdim=True)
    axis_pi  = torch.where(dot_pi < 0, -axis_pi, axis_pi)

    omega_near_pi = theta[..., None] * axis_pi              # (..., 3)

    # ------------------------------------------------------------------
    # Select branch per element
    # sin_theta < threshold covers both θ ≈ 0 and θ ≈ π;
    # the theta > 1.0 guard keeps only the near-π case here.
    # ------------------------------------------------------------------
    near_pi = (sin_theta < 0.1) & (theta > 1.0)
    omega = torch.where(near_pi[..., None], omega_near_pi, omega_standard)

    return omega.to(orig_dtype)


def se3_to_lie(T: torch.Tensor) -> torch.Tensor:
    """
    SE(3) homogeneous matrix → Lie algebra element.

    Args:
        T: (..., 4, 4)  SE(3) transformation matrix
    Returns:
        (..., 6)  [ω (3) | t (3)]

    Note: uses the simple decoupled approximation ξ = [log(R) | t],
    which is exact for the translational part when the rotation is extracted
    separately (sufficient for OTP's trajectory parameterisation).
    """
    T, orig_dtype = _promote(T)
    R = T[..., :3, :3]                                # (..., 3, 3)
    t = T[..., :3, 3]                                  # (..., 3)
    omega = so3_log(R)                                 # (..., 3)
    xi = torch.cat([omega, t], dim=-1)                 # (..., 6)
    return xi.to(orig_dtype)


def lie_to_se3(xi: torch.Tensor) -> torch.Tensor:
    """
    Lie algebra element → SE(3) homogeneous matrix.

    Args:
        xi: (..., 6)  [ω (3) | t (3)]
    Returns:
        (..., 4, 4)  SE(3) transformation matrix
    """
    xi, orig_dtype = _promote(xi)
    *batch, _ = xi.shape

    omega = xi[..., :3]                                # (..., 3)
    t = xi[..., 3:]                                    # (..., 3)
    R = so3_exp(omega)                                 # (..., 3, 3)

    # Build 4×4 matrix
    bottom = torch.zeros(*batch, 1, 4, dtype=xi.dtype, device=xi.device)
    bottom[..., 0, 3] = 1.0

    Rt = torch.cat([R, t.unsqueeze(-1)], dim=-1)       # (..., 3, 4)
    T = torch.cat([Rt, bottom], dim=-2)                # (..., 4, 4)

    return T.to(orig_dtype)


def quat_to_so3(q: torch.Tensor) -> torch.Tensor:
    """
    Unit quaternion (wxyz convention) → rotation matrix.

    Args:
        q: (..., 4)  [w, x, y, z]  (need not be unit; normalised internally)
    Returns:
        (..., 3, 3)
    """
    q, orig_dtype = _promote(q)
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    R = torch.stack([
        1 - 2*(yy+zz),   2*(xy-wz),    2*(xz+wy),
          2*(xy+wz),    1-2*(xx+zz),   2*(yz-wx),
          2*(xz-wy),      2*(yz+wx), 1-2*(xx+yy),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)

    return R.to(orig_dtype)


def so3_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    Rotation matrix → unit quaternion (wxyz, canonical w ≥ 0).

    Uses the numerically stable Shepperd method (largest component first).

    Args:
        R: (..., 3, 3)
    Returns:
        (..., 4)  [w, x, y, z]
    """
    R, orig_dtype = _promote(R)
    *batch, _, _ = R.shape

    # Four candidate squares: 4w², 4x², 4y², 4z²
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    q_sq = torch.stack([
        (1.0 + trace) / 4.0,
        (1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]) / 4.0,
        (1.0 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2]) / 4.0,
        (1.0 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2]) / 4.0,
    ], dim=-1).clamp(min=0.0)                          # (..., 4)

    # Index of the largest component
    idx = q_sq.argmax(dim=-1)                          # (...,)

    def _q_from_w_large(R):
        w = torch.sqrt(q_sq[..., 0])
        denom = (4.0 * w).clamp(min=_EPS)
        x = (R[..., 2, 1] - R[..., 1, 2]) / denom
        y = (R[..., 0, 2] - R[..., 2, 0]) / denom
        z = (R[..., 1, 0] - R[..., 0, 1]) / denom
        return torch.stack([w, x, y, z], dim=-1)

    def _q_from_x_large(R):
        x = torch.sqrt(q_sq[..., 1])
        denom = (4.0 * x).clamp(min=_EPS)
        w = (R[..., 2, 1] - R[..., 1, 2]) / denom
        y = (R[..., 1, 0] + R[..., 0, 1]) / denom
        z = (R[..., 0, 2] + R[..., 2, 0]) / denom
        return torch.stack([w, x, y, z], dim=-1)

    def _q_from_y_large(R):
        y = torch.sqrt(q_sq[..., 2])
        denom = (4.0 * y).clamp(min=_EPS)
        w = (R[..., 0, 2] - R[..., 2, 0]) / denom
        x = (R[..., 1, 0] + R[..., 0, 1]) / denom
        z = (R[..., 2, 1] + R[..., 1, 2]) / denom
        return torch.stack([w, x, y, z], dim=-1)

    def _q_from_z_large(R):
        z = torch.sqrt(q_sq[..., 3])
        denom = (4.0 * z).clamp(min=_EPS)
        w = (R[..., 1, 0] - R[..., 0, 1]) / denom
        x = (R[..., 0, 2] + R[..., 2, 0]) / denom
        y = (R[..., 2, 1] + R[..., 1, 2]) / denom
        return torch.stack([w, x, y, z], dim=-1)

    # Compute all four candidates and select by idx
    q0 = _q_from_w_large(R)
    q1 = _q_from_x_large(R)
    q2 = _q_from_y_large(R)
    q3 = _q_from_z_large(R)

    # Stack: (..., 4, 4); select along dim -2 by idx
    candidates = torch.stack([q0, q1, q2, q3], dim=-2)  # (..., 4, 4)
    idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(*batch, 1, 4)
    q = candidates.gather(-2, idx_exp).squeeze(-2)       # (..., 4)

    # Canonical form: w ≥ 0
    q = torch.where(q[..., :1] < 0, -q, q)
    q = F.normalize(q, dim=-1)

    return q.to(orig_dtype)


# ---------------------------------------------------------------------------
# Self-check: interface contract assertion (M1)
# Called once on module import in debug mode; skipped in production.
# ---------------------------------------------------------------------------

def _assert_contracts_smoke() -> None:
    """Quick sanity check — runs in <1 ms on CPU."""
    import math
    # so3_exp ∘ so3_log round-trip
    angle = math.pi / 4
    omega_ref = torch.tensor([[angle / math.sqrt(3)] * 3])
    R = so3_exp(omega_ref)
    omega_rec = so3_log(R)
    assert torch.allclose(omega_ref, omega_rec, atol=1e-5), \
        f"so3 round-trip failed: {omega_ref} vs {omega_rec}"

    # lie_to_se3 ∘ se3_to_lie round-trip
    T = torch.eye(4).unsqueeze(0)
    T[0, :3, :3] = R[0]
    T[0, :3, 3] = torch.tensor([0.1, 0.2, 0.3])
    xi = se3_to_lie(T)
    T_rec = lie_to_se3(xi)
    assert torch.allclose(T, T_rec, atol=1e-5), \
        f"se3 round-trip failed:\n{T}\nvs\n{T_rec}"
