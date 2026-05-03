"""
Conditional Flow Matching (CFM) and Shortcut Flow Matching for OTP.

Theory:
  Given source x_0 ~ N(0, I) and target x_1 (training data):
    x_t = (1 - t) * x_0 + t * x_1,   t ~ U[0, 1]
    v_target(x_t, t) = x_1 - x_0      (optimal transport path)
    loss_CFM = E[‖v_θ(x_t, t, c) - v_target‖²]

  Shortcut extension (Frans et al. 2024):
    v_θ(x_t, t, c, d) predicts the velocity to step d units forward.
    Self-consistency: v(x_t, t, d) = ½ [v(x_t, t, d/2) + v(x_{t+d/2}, t+d/2, d/2)]
    At d → 0 this is identical to standard CFM.

Model calling convention:
  FlowMatching        : model(x_t, t, condition)          -> (B, D) velocity
  ShortcutFlowMatching: model(x_t, t, condition, d)       -> (B, D) velocity
    where  d is a (B,) or (B, 1) float tensor ∈ [0, 1]

Returns:
  LossOutput (§2.2) — training loop reads via to_csv_row(), never raw key access.

Interface contracts (M1):
  component keys: {"flow"}                        for FlowMatching
  component keys: {"flow", "consistency"}         for ShortcutFlowMatching
  All asserted via LossOutput.assert_keys() immediately after construction.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from otp.utils.loss_output import LossOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Condition = Dict[str, torch.Tensor]
VelocityModel = Callable[..., torch.Tensor]


# ---------------------------------------------------------------------------
# FlowMatching
# ---------------------------------------------------------------------------

class FlowMatching(nn.Module):
    """
    Conditional Flow Matching loss + Euler-step sampler.

    Args:
        model: callable(x_t, t, condition) -> (B, D) velocity prediction.
               Must be an nn.Module or any callable.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    # ------------------------------------------------------------------
    # Forward: compute CFM loss
    # ------------------------------------------------------------------
    def forward(
        self,
        x_1: torch.Tensor,
        condition: Condition,
    ) -> LossOutput:
        """
        Compute conditional flow matching loss.

        Args:
            x_1:       (B, D) target samples from training data.
            condition: dict of conditioning tensors passed to self.model.

        Returns:
            LossOutput with component "flow" and diagnostics "t_mean", "v_norm".

        NaN guard (§2.3): if x_1 contains NaN/Inf, returns LossOutput(total=None).
        """
        if torch.isnan(x_1).any() or torch.isinf(x_1).any():
            logger.error(
                "[FlowMatching] NaN/Inf detected in x_1 — skipping batch (§2.3). "
                "shape=%s, nan_count=%d",
                x_1.shape,
                torch.isnan(x_1).sum().item(),
            )
            out = LossOutput(
                total=None,
                components={},
                diagnostics={"nan_batch": 1.0},
            )
            return out

        B, D = x_1.shape
        device, dtype = x_1.device, x_1.dtype

        # Sample noise and time
        x_0 = torch.randn_like(x_1)
        t = torch.rand(B, 1, device=device, dtype=dtype)

        # Linear interpolation
        x_t = (1.0 - t) * x_0 + t * x_1
        v_target = x_1 - x_0                           # (B, D)

        # Velocity prediction
        v_pred = self.model(x_t, t, condition)          # (B, D)

        # M1: verify model output shape matches target
        assert v_pred.shape == v_target.shape, (
            f"[M1] velocity shape mismatch: v_pred={v_pred.shape}, "
            f"v_target={v_target.shape}"
        )

        flow_loss = F.mse_loss(v_pred, v_target)

        out = LossOutput(
            total=flow_loss,
            components={"flow": flow_loss},
            diagnostics={
                "t_mean":  t.mean(),
                "v_norm":  v_pred.detach().norm(dim=-1).mean(),
            },
        )
        out.assert_keys(["flow"])                       # M1 contract check
        return out

    # ------------------------------------------------------------------
    # Sample: Euler ODE integration
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        condition: Condition,
        num_steps: int = 16,
        x_0: Optional[torch.Tensor] = None,
        shape: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Generate samples by Euler integration of the learned vector field.

        Args:
            condition:  conditioning dict.
            num_steps:  number of Euler steps (higher = more accurate).
            x_0:        optional starting noise; if None, sampled from N(0, I).
            shape:      (B, D) shape for x_0 when x_0 is None.

        Returns:
            (B, D) tensor of generated samples.

        Raises:
            ValueError: if neither x_0 nor shape is provided.
            RuntimeError (§2.3 / M2): if NaN detected mid-integration.
        """
        if x_0 is None:
            if shape is None:
                raise ValueError("FlowMatching.sample: provide x_0 or shape")
            # Infer device from condition tensors if possible
            dev = _infer_device(condition)
            x_0 = torch.randn(*shape, device=dev)

        x = x_0.clone()
        dt = 1.0 / num_steps
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((B, 1), t_val, device=device, dtype=dtype)
            v = self.model(x, t, condition)
            x = x + dt * v

            if torch.isnan(x).any() or torch.isinf(x).any():
                nan_count = torch.isnan(x).sum().item()
                logger.error(
                    "[FlowMatching.sample] NaN/Inf at step %d/%d — "
                    "nan_count=%d (§2.3 / M2)",
                    i + 1, num_steps, nan_count,
                )
                raise RuntimeError(
                    f"FlowMatching.sample: NaN/Inf at integration step {i + 1}/{num_steps}. "
                    f"Check model numerics."
                )

        return x


# ---------------------------------------------------------------------------
# ShortcutFlowMatching
# ---------------------------------------------------------------------------

class ShortcutFlowMatching(FlowMatching):
    """
    Shortcut Flow Matching (Frans et al. 2024).

    Extends CFM with a self-consistency loss that enables variable-step
    inference: at test time, a single forward pass with d=1 can approximate
    the full trajectory.

    Model calling convention:
        model(x_t, t, condition, d) -> (B, D)
    where d is a (B, 1) float tensor.

    Args:
        model:               callable(x_t, t, condition, d) -> velocity.
        num_shortcut_levels: K; shortcut sizes are {1/2^k : k=0,...,K}.
                             k=K corresponds to d→0 (standard FM).
        consistency_weight:  λ; weight on self-consistency loss.
                             Total = flow_loss + λ * consistency_loss.
    """

    def __init__(
        self,
        model: nn.Module,
        num_shortcut_levels: int = 4,
        consistency_weight: float = 1.0,
    ) -> None:
        super().__init__(model)
        self.num_shortcut_levels = num_shortcut_levels
        self.consistency_weight = consistency_weight

    def forward(
        self,
        x_1: torch.Tensor,
        condition: Condition,
    ) -> LossOutput:
        """
        Compute CFM loss + self-consistency loss.

        For each sample in the batch:
          1. Sample d from {1, 1/2, 1/4, ..., 1/2^K}.
          2. Compute self-consistency target via two d/2 sub-steps
             (stop gradient on the "bootstrap" path per Frans et al. 2024).
          3. Also compute standard FM loss at d → 0 (k = K level).

        Returns LossOutput with components {"flow", "consistency"}.

        NaN guard (§2.3): returns LossOutput(total=None) on NaN/Inf input.
        """
        if torch.isnan(x_1).any() or torch.isinf(x_1).any():
            logger.error(
                "[ShortcutFlowMatching] NaN/Inf in x_1 — skipping batch (§2.3)."
            )
            return LossOutput(total=None, components={}, diagnostics={"nan_batch": 1.0})

        B, D = x_1.shape
        device, dtype = x_1.device, x_1.dtype

        # ------------------------------------------------------------------
        # 1. Standard FM loss (use the model with d ≈ 0 via smallest level)
        # ------------------------------------------------------------------
        x_0 = torch.randn_like(x_1)
        t_fm = torch.rand(B, 1, device=device, dtype=dtype)
        x_t_fm = (1.0 - t_fm) * x_0 + t_fm * x_1
        v_target_fm = x_1 - x_0

        d_zero = torch.zeros(B, 1, device=device, dtype=dtype)
        v_pred_fm = self.model(x_t_fm, t_fm, condition, d_zero)

        assert v_pred_fm.shape == v_target_fm.shape, (
            f"[M1] FM branch shape mismatch: {v_pred_fm.shape} vs {v_target_fm.shape}"
        )
        flow_loss = F.mse_loss(v_pred_fm, v_target_fm)

        # ------------------------------------------------------------------
        # 2. Self-consistency loss
        #    Sample shortcut level k ~ U{0,...,K-1} (d = 1/2^k, never k=K)
        # ------------------------------------------------------------------
        K = self.num_shortcut_levels
        # k=0 → d=1, k=1 → d=0.5, ..., k=K-1 → d=1/2^(K-1)
        k = torch.randint(0, K, (B,), device=device)
        d = (0.5 ** k.float()).unsqueeze(-1)            # (B, 1)
        d_half = d / 2.0                                # (B, 1)

        # Sample t such that t + d ≤ 1
        t_max = (1.0 - d).clamp(min=0.0)
        t_sc = torch.rand(B, 1, device=device, dtype=dtype) * t_max

        x_0_sc = torch.randn_like(x_1)
        x_t_sc = (1.0 - t_sc) * x_0_sc + t_sc * x_1

        # Bootstrap path (stop gradient to avoid shortcut collapse)
        with torch.no_grad():
            v_first = self.model(x_t_sc, t_sc, condition, d_half)
            x_mid = x_t_sc + d_half * v_first
            t_mid = (t_sc + d_half).clamp(max=1.0)
            v_second = self.model(x_mid, t_mid, condition, d_half)

        # Self-consistency target: average of two half-steps (M1 — verified below)
        v_consist_target = 0.5 * (v_first + v_second)  # (B, D)

        # Prediction for full d
        v_pred_sc = self.model(x_t_sc, t_sc, condition, d)

        assert v_pred_sc.shape == v_consist_target.shape, (
            f"[M1] SC branch shape mismatch: {v_pred_sc.shape} vs {v_consist_target.shape}"
        )
        consist_loss = F.mse_loss(v_pred_sc, v_consist_target)

        total = flow_loss + self.consistency_weight * consist_loss

        out = LossOutput(
            total=total,
            components={"flow": flow_loss, "consistency": consist_loss},
            diagnostics={
                "d_mean":           d.mean(),
                "t_sc_mean":        t_sc.mean(),
                "v_consist_norm":   v_pred_sc.detach().norm(dim=-1).mean(),
            },
        )
        out.assert_keys(["flow", "consistency"])        # M1 contract check
        return out

    @torch.no_grad()
    def sample(
        self,
        condition: Condition,
        num_steps: int = 1,
        x_0: Optional[torch.Tensor] = None,
        shape: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Generate samples using shortcut steps.

        With num_steps=1 and a well-trained model, a single forward pass
        with d=1 gives a direct estimate of x_1.

        Args:
            condition:  conditioning dict.
            num_steps:  number of shortcut steps (1 = one big step with d=1).
            x_0:        optional starting noise.
            shape:      (B, D) for x_0 when x_0 is None.
        """
        if x_0 is None:
            if shape is None:
                raise ValueError("ShortcutFlowMatching.sample: provide x_0 or shape")
            dev = _infer_device(condition)
            x_0 = torch.randn(*shape, device=dev)

        x = x_0.clone()
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        d_step = 1.0 / num_steps
        d = torch.full((B, 1), d_step, device=device, dtype=dtype)

        for i in range(num_steps):
            t_val = i * d_step
            t = torch.full((B, 1), t_val, device=device, dtype=dtype)
            v = self.model(x, t, condition, d)
            x = x + d_step * v

            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.error(
                    "[ShortcutFlowMatching.sample] NaN/Inf at step %d/%d (§2.3 / M2)",
                    i + 1, num_steps,
                )
                raise RuntimeError(
                    f"ShortcutFlowMatching.sample: NaN at step {i + 1}/{num_steps}"
                )

        return x


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _infer_device(condition: Condition) -> torch.device:
    """Extract device from first tensor in condition dict, fallback CPU."""
    for v in condition.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")
