"""
OTPHead: cross-attention decoder + Shortcut Flow Matching trajectory head.

Predicts (B, N_obj, H, 6) Lie algebra trajectories for each object over a
planning horizon H.

Architecture:
  1. OTPCrossAttentionLayer  — per-layer cross-attention from object queries
                               to backbone hidden states + positionwise FFN.
  2. OTPVelocityNet          — MLP velocity predictor for flow matching;
                               signature: forward(x_t, t, condition, d=None).
  3. OTPHead                 — wraps layers 1–2 with ShortcutFlowMatching
                               (or plain FlowMatching when use_shortcut=False).

Interface contract (§REPO_LAYOUT.md Stage 2):
  forward(backbone_hidden, backbone_attention_mask, object_indices,
          gt_trajectory=None) -> dict
    training  (gt_trajectory provided): {'loss_output': LossOutput}
    inference (gt_trajectory=None):     {'trajectories': (B,N_obj,H,6)}

  sample(backbone_hidden, backbone_attention_mask, object_indices,
         num_steps=16) -> (B, N_obj, H, 6)

NaN guard (§2.3): NaN backbone_hidden → LossOutput(total=None) / zero trajectories.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from otp.utils.flow_matching import FlowMatching, ShortcutFlowMatching
from otp.utils.loss_output import LossOutput


# ---------------------------------------------------------------------------
# Cross-attention layer
# ---------------------------------------------------------------------------

class OTPCrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer: object queries attend to backbone tokens.

    Uses pre-norm (LayerNorm before attention and FFN) for training stability.
    The FFN has a 4× expansion ratio.

    Args:
        query_dim:  Dimension of object query vectors.
        memory_dim: Dimension of backbone hidden states (e.g. 4096 for LLaMA-7B).
        num_heads:  Number of attention heads.
        dropout:    Dropout probability on attention weights.
    """

    def __init__(
        self,
        query_dim: int,
        memory_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_m = nn.LayerNorm(memory_dim)
        self.kv_proj = nn.Linear(memory_dim, query_dim * 2, bias=False)
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.attn = nn.MultiheadAttention(
            query_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm_ff = nn.LayerNorm(query_dim)
        self.ff = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Linear(query_dim * 4, query_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries:                  (B, N_obj, query_dim)
            memory:                   (B, S, memory_dim) backbone hidden states.
            memory_key_padding_mask:  (B, S) bool; True where tokens are padding
                                      (i.e. attention_mask == 0).

        Returns:
            Updated queries: (B, N_obj, query_dim).
        """
        # Cross-attention (pre-norm on queries and memory separately).
        q = self.q_proj(self.norm_q(queries))
        kv = self.kv_proj(self.norm_m(memory))
        k, v = kv.chunk(2, dim=-1)

        attn_out, _ = self.attn(
            q, k, v, key_padding_mask=memory_key_padding_mask
        )
        queries = queries + attn_out

        # Positionwise FFN (pre-norm).
        queries = queries + self.ff(self.norm_ff(queries))
        return queries


# ---------------------------------------------------------------------------
# Velocity network (flow matching model)
# ---------------------------------------------------------------------------

class OTPVelocityNet(nn.Module):
    """
    MLP velocity predictor for (Shortcut) Flow Matching over trajectories.

    Calling conventions:
      FlowMatching        : forward(x_t, t, condition)       — d is None
      ShortcutFlowMatching: forward(x_t, t, condition, d)    — d is (B, 1)

    Args:
        traj_dim:      Flat trajectory dimension = N_obj * H * 6.
        condition_dim: Flat object-feature dimension = N_obj * hidden_dim.
        hidden_dim:    Width of each MLP hidden layer.
        num_layers:    Number of hidden layers.
    """

    def __init__(
        self,
        traj_dim: int,
        condition_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.traj_dim = traj_dim
        # Input: x_t(D) + t(1) + condition(condition_dim) [+ d(1) if shortcut]
        in_dim = traj_dim + 1 + condition_dim + 1     # +1 reserved for d (0 if unused)
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, traj_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: dict,
        d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t:        (B, D) noisy trajectory at time t.
            t:          (B, 1) diffusion time ∈ [0, 1].
            condition:  dict with 'obj_features': (B, N_obj, C).
            d:          (B, 1) shortcut step size; zeros when not using shortcut.

        Returns:
            (B, D) predicted velocity.
        """
        obj_feat = condition["obj_features"].flatten(1)   # (B, N_obj*C)

        if d is None:
            d_inp = torch.zeros(x_t.shape[0], 1, device=x_t.device, dtype=x_t.dtype)
        else:
            d_inp = d.to(dtype=x_t.dtype)

        inp = torch.cat([x_t, t, obj_feat, d_inp], dim=-1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# OTP head
# ---------------------------------------------------------------------------

class OTPHead(nn.Module):
    """
    Full OTP trajectory prediction head.

    Extracts per-object features from LLM backbone tokens via cross-attention,
    then uses (Shortcut) Flow Matching to learn a distribution over SE(3)
    Lie algebra trajectories.

    Args:
        backbone_dim:   Hidden dimension of the backbone (e.g. 4096).
        hidden_dim:     Projection / cross-attention dimension.
        num_objects:    Number of tracked objects N_obj.
        horizon:        Planning horizon H (number of SE(3) waypoints).
        num_heads:      Attention heads in cross-attention layers.
        num_layers:     Number of OTPCrossAttentionLayer stacks.
        vel_hidden_dim: Hidden dim for OTPVelocityNet (defaults to hidden_dim).
        vel_num_layers: Depth of OTPVelocityNet.
        use_shortcut:   Use ShortcutFlowMatching if True, FlowMatching if False.
        num_shortcut_levels: Passed to ShortcutFlowMatching.
        consistency_weight:  Passed to ShortcutFlowMatching.
        num_sample_steps:    Default Euler / shortcut steps for sample().
    """

    def __init__(
        self,
        backbone_dim: int = 4096,
        hidden_dim: int = 512,
        num_objects: int = 5,
        horizon: int = 8,
        num_heads: int = 8,
        num_layers: int = 4,
        vel_hidden_dim: Optional[int] = None,
        vel_num_layers: int = 3,
        use_shortcut: bool = True,
        num_shortcut_levels: int = 4,
        consistency_weight: float = 1.0,
        num_sample_steps: int = 16,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        self.num_objects = num_objects
        self.horizon = horizon
        self.num_sample_steps = num_sample_steps
        self.use_shortcut = use_shortcut
        self.deterministic = deterministic

        # Learnable object query embeddings.
        self.obj_queries = nn.Parameter(torch.randn(1, num_objects, hidden_dim) * 0.02)

        # Project backbone tokens to hidden_dim for key/value computation.
        self.backbone_proj = nn.Linear(backbone_dim, hidden_dim, bias=False)

        # Cross-attention decoder layers.
        self.cross_attn_layers = nn.ModuleList([
            OTPCrossAttentionLayer(
                query_dim=hidden_dim,
                memory_dim=hidden_dim,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        traj_dim = num_objects * horizon * 6
        condition_dim = num_objects * hidden_dim

        if deterministic:
            # PATH B: direct trajectory regression. Bypasses flow matching.
            self.traj_head = nn.Sequential(
                nn.Linear(condition_dim, condition_dim),
                nn.GELU(),
                nn.Linear(condition_dim, condition_dim),
                nn.GELU(),
                nn.Linear(condition_dim, traj_dim),
            )
            self.velocity_net = None
            self.flow_matcher = None
        else:
            v_hidden = vel_hidden_dim if vel_hidden_dim is not None else hidden_dim
            self.velocity_net = OTPVelocityNet(
                traj_dim=traj_dim,
                condition_dim=condition_dim,
                hidden_dim=v_hidden,
                num_layers=vel_num_layers,
            )
            if use_shortcut:
                self.flow_matcher: FlowMatching = ShortcutFlowMatching(
                    self.velocity_net,
                    num_shortcut_levels=num_shortcut_levels,
                    consistency_weight=consistency_weight,
                )
            else:
                self.flow_matcher = FlowMatching(self.velocity_net)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_condition(
        self,
        backbone_hidden: torch.Tensor,
        backbone_attention_mask: Optional[torch.Tensor],
        object_indices: torch.Tensor,
    ) -> dict:
        """
        Extract per-object feature vectors via cross-attention.

        Args:
            backbone_hidden:         (B, S, backbone_dim)
            backbone_attention_mask: (B, S) int/bool; 1=valid, 0=padding (or None).
            object_indices:          (B, N_obj) int indices into the S dimension.

        Returns:
            dict with 'obj_features': (B, N_obj, hidden_dim).
        """
        B = backbone_hidden.shape[0]

        # Project backbone to hidden_dim (keys/values for cross-attention).
        memory = self.backbone_proj(backbone_hidden)     # (B, S, hidden_dim)

        # Build padding mask: True where tokens should be IGNORED.
        if backbone_attention_mask is not None:
            key_padding_mask = (backbone_attention_mask == 0)   # (B, S)
        else:
            key_padding_mask = None

        # Initialise object queries from learnable embeddings.
        # If object_indices are provided, also add the backbone token embedding
        # at each object position as a positional hint.
        queries = self.obj_queries.expand(B, -1, -1).clone()  # (B, N_obj, D)
        idx = object_indices.unsqueeze(-1).expand(-1, -1, memory.shape[-1])
        obj_tokens = torch.gather(memory, 1, idx)              # (B, N_obj, D)
        queries = queries + obj_tokens

        # Cross-attention layers.
        for layer in self.cross_attn_layers:
            queries = layer(queries, memory, key_padding_mask)

        return {"obj_features": queries}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        backbone_hidden: torch.Tensor,
        backbone_attention_mask: Optional[torch.Tensor],
        object_indices: torch.Tensor,
        gt_trajectory: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            backbone_hidden:         (B, S, backbone_dim)
            backbone_attention_mask: (B, S) or None
            object_indices:          (B, N_obj) int
            gt_trajectory:           (B, N_obj, H, 6) or None

        Returns:
            Training   (gt_trajectory provided): {'loss_output': LossOutput}
            Inference  (gt_trajectory=None):     {'trajectories': (B, N_obj, H, 6)}
        """
        B = backbone_hidden.shape[0]

        # NaN guard on backbone input (§2.3).
        if torch.isnan(backbone_hidden).any() or torch.isinf(backbone_hidden).any():
            if gt_trajectory is not None:
                return {
                    "loss_output": LossOutput(
                        total=None,
                        components={},
                        diagnostics={"nan_backbone": 1.0},
                    )
                }
            traj_dim = self.num_objects * self.horizon * 6
            return {
                "trajectories": torch.zeros(
                    B, self.num_objects, self.horizon, 6,
                    device=backbone_hidden.device,
                    dtype=backbone_hidden.dtype,
                )
            }

        condition = self._get_condition(
            backbone_hidden, backbone_attention_mask, object_indices
        )

        if self.deterministic:
            # PATH B: direct regression.
            obj_feats = condition["obj_features"]
            feat_flat = obj_feats.reshape(B, -1)
            traj_flat = self.traj_head(feat_flat)
            trajectories = traj_flat.reshape(
                B, self.num_objects, self.horizon, 6
            )
            if gt_trajectory is not None:
                loss = F.l1_loss(trajectories, gt_trajectory)
                return {
                    "loss_output": LossOutput(
                        total=loss,
                        components={"l1": loss.detach()},
                        diagnostics={
                            "traj_norm": trajectories.norm(dim=-1).mean().detach(),
                            "gt_norm": gt_trajectory.norm(dim=-1).mean().detach(),
                        },
                    )
                }
            return {"trajectories": trajectories}

        if gt_trajectory is not None:
            # Training path.
            x_1 = gt_trajectory.reshape(B, -1)           # (B, N_obj*H*6)
            loss_output = self.flow_matcher(x_1, condition)
            return {"loss_output": loss_output}

        # Inference path.
        traj_dim = self.num_objects * self.horizon * 6
        samples = self.flow_matcher.sample(
            condition,
            num_steps=self.num_sample_steps,
            shape=(B, traj_dim),
        )
        trajectories = samples.reshape(B, self.num_objects, self.horizon, 6)
        return {"trajectories": trajectories}

    # ------------------------------------------------------------------
    # Convenience sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        backbone_hidden: torch.Tensor,
        backbone_attention_mask: Optional[torch.Tensor],
        object_indices: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample trajectories without computing loss.

        Returns:
            (B, N_obj, H, 6) Lie algebra trajectory tensor.
        """
        B = backbone_hidden.shape[0]
        condition = self._get_condition(
            backbone_hidden, backbone_attention_mask, object_indices
        )

        if self.deterministic:
            obj_feats = condition["obj_features"]
            feat_flat = obj_feats.reshape(B, -1)
            traj_flat = self.traj_head(feat_flat)
            return traj_flat.reshape(B, self.num_objects, self.horizon, 6)

        steps = num_steps if num_steps is not None else self.num_sample_steps
        traj_dim = self.num_objects * self.horizon * 6
        samples = self.flow_matcher.sample(
            condition, num_steps=steps, shape=(B, traj_dim)
        )
        return samples.reshape(B, self.num_objects, self.horizon, 6)
