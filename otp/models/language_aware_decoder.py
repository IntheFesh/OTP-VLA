"""
LanguageAwareDecoder — OCLDStrictVariant decoder (ablation baseline).

DESIGN NOTE: This module INTENTIONALLY violates the C3 (Causal Counterfactual
Compatibility) constraint.  It is used as an ablation baseline to measure how
much performance degrades when language is injected into the decoder.

Unlike LanguageAgnosticDecoder, this decoder receives backbone_hidden states
(which encode the language instruction ℓ) through an extra cross-attention
block inserted between the multimodal encoding and the flow-matching head.

Do NOT use this module in the OTP-Soft policy.  Use LanguageAgnosticDecoder
(otp/models/language_agnostic_decoder.py) for the C3-compliant variant.

Architecture delta vs LanguageAgnosticDecoder:
  + backbone_proj:      Linear(backbone_dim → hidden_dim)  — maps backbone
                        tokens to decoder hidden dim.
  + lang_cross_attn:    TransformerDecoder — queries are the pooled multimodal
                        feature; keys/values come from backbone_hidden.
  Everything else (trajectory_encoder, affordance_encoder, geometry_encoder,
  proprio_encoder, positional embeddings, flow_matcher) is identical.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from otp.utils.flow_matching import FlowMatching, ShortcutFlowMatching
from otp.utils.loss_output import LossOutput


# ---------------------------------------------------------------------------
# Velocity net (shared with LanguageAgnosticDecoder structure)
# ---------------------------------------------------------------------------

class _LangAwareVelocityNet(nn.Module):
    """MLP velocity model for ShortcutFlowMatching over action space."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        condition_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.flat_dim = action_dim * horizon
        in_dim = self.flat_dim + 1 + condition_dim + 1
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, self.flat_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: dict,
        d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = condition["feat"]
        if d is None:
            d = torch.zeros(x_t.shape[0], 1, device=x_t.device, dtype=x_t.dtype)
        else:
            d = d.to(dtype=x_t.dtype)
        return self.net(torch.cat([x_t, t, feat, d], dim=-1))


# ---------------------------------------------------------------------------
# LanguageAwareDecoder
# ---------------------------------------------------------------------------

class LanguageAwareDecoder(nn.Module):
    """
    OCLDStrictVariant decoder: same as LanguageAgnosticDecoder plus one
    cross-attention block that ingests backbone hidden states (language).

    Args:
        backbone_dim:          Hidden dim of the VLA backbone (e.g. 4096).
        trajectory_dim:        6 (SE(3) Lie algebra).
        affordance_dim:        7 (pos(3) + quat(4)).
        geometry_dim:          Output dim of GeometryEncoder.
        proprio_dim:           8 (EE pose(7) + grip(1)).
        hidden_dim:            Common hidden dim for all modality projections.
        num_objects:           N_obj.
        horizon:               H (planning horizon).
        num_grasps_per_object: K.
        num_decoder_layers:    Depth of multimodal cross-attention transformer.
        num_lang_layers:       Depth of language cross-attention block.
        num_heads:             Attention heads.
        action_dim:            7 (LIBERO).
        use_flow_matching:     Shortcut if True, plain Flow if False.
        num_sample_steps:      Default inference steps.
        consistency_weight:    Shortcut self-consistency weight.
    """

    def __init__(
        self,
        backbone_dim: int = 4096,
        trajectory_dim: int = 6,
        affordance_dim: int = 7,
        geometry_dim: int = 64,
        proprio_dim: int = 8,
        hidden_dim: int = 512,
        num_objects: int = 5,
        horizon: int = 8,
        num_grasps_per_object: int = 8,
        num_decoder_layers: int = 4,
        num_lang_layers: int = 2,
        num_heads: int = 8,
        action_dim: int = 7,
        use_flow_matching: bool = True,
        num_sample_steps: int = 8,
        consistency_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.trajectory_dim = trajectory_dim
        self.affordance_dim = affordance_dim
        self.geometry_dim = geometry_dim
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.horizon = horizon
        self.num_grasps_per_object = num_grasps_per_object
        self.action_dim = action_dim
        self.num_sample_steps = num_sample_steps

        # ---- Modality encoders (identical to LanguageAgnosticDecoder) ---- #
        self.trajectory_encoder = nn.Linear(trajectory_dim, hidden_dim)
        self.affordance_encoder = nn.Linear(affordance_dim, hidden_dim)
        self.geometry_encoder_proj = nn.Linear(geometry_dim, hidden_dim)
        self.proprio_encoder = nn.Linear(proprio_dim, hidden_dim)

        # ---- Positional embeddings ---- #
        self.traj_pos_embed = nn.Parameter(
            torch.randn(1, 1, horizon, hidden_dim) * 0.02
        )
        self.aff_pos_embed = nn.Parameter(
            torch.randn(1, 1, num_grasps_per_object, hidden_dim) * 0.02
        )
        self.obj_pos_embed = nn.Parameter(
            torch.randn(1, num_objects, hidden_dim) * 0.02
        )

        # ---- Multimodal cross-attention (same as base decoder) ---- #
        mm_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.cross_attn = nn.TransformerDecoder(mm_layer, num_layers=num_decoder_layers)

        # ---- Language cross-attention (C3 violation — ablation only) ---- #
        # Maps backbone tokens to decoder hidden_dim for k/v.
        self.backbone_proj = nn.Linear(backbone_dim, hidden_dim)
        lang_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.lang_cross_attn = nn.TransformerDecoder(lang_layer, num_layers=num_lang_layers)

        # ---- Conditioning projection ---- #
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)

        # ---- Flow matching head ---- #
        velocity_net = _LangAwareVelocityNet(
            action_dim=action_dim,
            horizon=horizon,
            condition_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
        )
        if use_flow_matching:
            self.flow_matcher: FlowMatching = ShortcutFlowMatching(
                velocity_net,
                num_shortcut_levels=4,
                consistency_weight=consistency_weight,
            )
        else:
            self.flow_matcher = FlowMatching(velocity_net)

    # ------------------------------------------------------------------
    def forward(
        self,
        trajectory: torch.Tensor,
        proprioception: torch.Tensor,
        grasp_affordance: torch.Tensor,
        object_geometry: torch.Tensor,
        backbone_hidden: torch.Tensor,
        backbone_attention_mask: Optional[torch.Tensor] = None,
        gt_action: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            trajectory:              (B, N_obj, H, 6)
            proprioception:          (B, 8)
            grasp_affordance:        (B, N_obj, K, 7)
            object_geometry:         (B, N_obj, D_geom)
            backbone_hidden:         (B, S, backbone_dim)   [language signal]
            backbone_attention_mask: (B, S) or None
            gt_action:               (B, H, 7) for training, else None.

        Returns (training):
            {'loss_output': LossOutput, 'loss': scalar or None,
             'pred_action': None}
        Returns (inference):
            {'pred_action': (B, H, 7)}
        """
        B = trajectory.shape[0]
        N_obj = trajectory.shape[1]
        H = trajectory.shape[2]
        K = grasp_affordance.shape[2]

        # ---- Encode modalities ---- #
        traj_tok = self.trajectory_encoder(trajectory) + self.traj_pos_embed    # (B, N, H, D)
        aff_tok = self.affordance_encoder(grasp_affordance) + self.aff_pos_embed
        geom_tok = self.geometry_encoder_proj(object_geometry) + self.obj_pos_embed
        proprio_tok = self.proprio_encoder(proprioception)                       # (B, D)

        memory = torch.cat([
            aff_tok.reshape(B, N_obj * K, self.hidden_dim),
            geom_tok.reshape(B, N_obj, self.hidden_dim),
            proprio_tok.unsqueeze(1),
        ], dim=1)                                                                # (B, M, D)

        queries = traj_tok.reshape(B, N_obj * H, self.hidden_dim)               # (B, N*H, D)

        # ---- Multimodal cross-attention ---- #
        decoded = self.cross_attn(queries, memory)                              # (B, N*H, D)

        # ---- Language cross-attention (ablation: inject backbone) ---- #
        lang_memory = self.backbone_proj(backbone_hidden)                       # (B, S, D)
        if backbone_attention_mask is not None:
            key_padding_mask = (backbone_attention_mask == 0)                   # (B, S) bool
        else:
            key_padding_mask = None
        decoded = self.lang_cross_attn(
            decoded, lang_memory, memory_key_padding_mask=key_padding_mask
        )                                                                        # (B, N*H, D)

        # ---- Pool → condition ---- #
        cond_feat = self.cond_proj(decoded.mean(dim=1))                         # (B, D)
        condition = {"feat": cond_feat}

        # ---- Flow matching ---- #
        flat_dim = self.action_dim * H
        if gt_action is not None:
            assert gt_action.shape == (B, H, self.action_dim)
            x_1 = gt_action.reshape(B, flat_dim)
            loss_output: LossOutput = self.flow_matcher(x_1, condition)
            return {
                "loss_output": loss_output,
                "loss": loss_output.total,
                "pred_action": None,
            }

        samples = self.flow_matcher.sample(
            condition, num_steps=self.num_sample_steps, shape=(B, flat_dim)
        )
        return {"pred_action": samples.reshape(B, H, self.action_dim)}

    @torch.no_grad()
    def sample(
        self,
        trajectory: torch.Tensor,
        proprioception: torch.Tensor,
        grasp_affordance: torch.Tensor,
        object_geometry: torch.Tensor,
        backbone_hidden: torch.Tensor,
        backbone_attention_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Inference-only sampler; returns (B, H, action_dim)."""
        out = self.forward(
            trajectory=trajectory,
            proprioception=proprioception,
            grasp_affordance=grasp_affordance,
            object_geometry=object_geometry,
            backbone_hidden=backbone_hidden,
            backbone_attention_mask=backbone_attention_mask,
            gt_action=None,
        )
        return out["pred_action"]
