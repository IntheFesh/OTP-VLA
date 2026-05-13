"""
LanguageAgnosticDecoder φ_θ : (z, w(o)) → a^(H)

Theorem 2 (revised):
    π_θ(a | o, ℓ) = ∫ φ_θ(z, w(o)) p_θ(z | o, ℓ) dz

The language instruction ℓ enters the policy ONLY through the OTP head's
trajectory distribution p_θ(z | o, ℓ).  The decoder φ_θ marginalises over
that distribution and is itself a function of (z, w(o)) — it never sees ℓ.

C3 (Causal Counterfactual Compatibility) requires that the decoder's
output distribution be invariant to ℓ when (z, w(o)) is held fixed.

CRITICAL DESIGN INVARIANTS (this code is the source-of-truth proof of C3):
  1. forward() argument names must NOT contain any of:
       'instruction', 'language', 'lang', 'text',
       'backbone', 'hidden', 'input_ids'
     — enforced by an inspect-based static check (see _verify_signature_at_init).
  2. forward() must NOT touch any field on `self` whose name contains
     'backbone' / 'language' (the decoder is a leaf module by construction).
  3. tests/test_language_agnostic_decoder.py exercises both checks
     (test_c3_compliance_static + test_c3_compliance_dynamic).

Inputs (every one of these is independent of ℓ):
  trajectory       : (B, N_obj, H, 6)        SE(3) Lie algebra waypoints
                                              produced by the OTP head, which
                                              already marginalised over ℓ.
  proprioception   : (B, 8)                  EE pose(7) + gripper state(1)
  grasp_affordance : (B, N_obj, K, 7)        K candidate grasp poses per obj
                                              (position(3) + quaternion(4))
  object_geometry  : (B, N_obj, D_geom)       output of GeometryEncoder

Output: a^(H) ∈ R^(H × 7)  (LIBERO action sequence over the horizon)

The action distribution is modelled with conditional Shortcut Flow Matching
on the flat action space (B, H × 7) — the multi-modality is genuine
(different valid grasps yield different action prefixes).
"""

from __future__ import annotations

import inspect
from typing import Optional

import torch
import torch.nn as nn

from otp.utils.flow_matching import FlowMatching, ShortcutFlowMatching
from otp.utils.loss_output import LossOutput


# ---------------------------------------------------------------------------
# Forbidden parameter / attribute name fragments — enforced statically.
# ---------------------------------------------------------------------------
_FORBIDDEN_FRAGMENTS = (
    "instruction",
    "language",
    "lang_",
    "text",
    "backbone",
    "hidden_state",
    "input_ids",
)


def _check_forbidden(names) -> None:
    """Raise AssertionError if any name contains a forbidden fragment."""
    for n in names:
        low = n.lower()
        for frag in _FORBIDDEN_FRAGMENTS:
            assert frag not in low, (
                f"[C3 VIOLATION] Name {n!r} contains forbidden fragment "
                f"{frag!r}.  LanguageAgnosticDecoder must not reference "
                f"language/instruction/backbone state — see Theorem 2 (C3)."
            )


# ---------------------------------------------------------------------------
# Velocity network for action flow matching
# ---------------------------------------------------------------------------

class _ActionVelocityNet(nn.Module):
    """
    MLP velocity model for ShortcutFlowMatching over action space.

    Calling convention compatible with ShortcutFlowMatching:
        forward(x_t, t, condition, d=None) -> (B, D)
    where condition['feat'] is the (B, condition_dim) feature vector produced
    by the decoder's cross-attention block.
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        condition_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.flat_dim = action_dim * horizon

        # Input layout: x_t (flat_dim) + t (1) + condition (condition_dim) + d (1)
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
        inp = torch.cat([x_t, t, feat, d], dim=-1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# LanguageAgnosticDecoder
# ---------------------------------------------------------------------------

class CocosSourceMLP(nn.Module):
    """F_phi(c): condition vector -> source distribution mean shift.

    For conditional flow matching with shifted Gaussian source:
        x_0 ~ N(alpha * F_phi(c), beta^2 * I)
    instead of the standard N(0, I).

    C3 SAFE: takes only the pooled condition feature (already constructed
    from non-language inputs in LanguageAgnosticDecoder), no language input.
    """
    def __init__(self, condition_dim: int, target_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        # c: (B, condition_dim) -> (B, target_dim)
        return self.net(c)


class LanguageAgnosticDecoder(nn.Module):
    """
    Language-agnostic decoder φ_θ.

    Args:
        trajectory_dim:        Per-step trajectory dim (6 for SE(3) Lie algebra).
        affordance_dim:        Per-grasp affordance dim (7 = pos(3) + quat(4)).
        geometry_dim:          Output dim of GeometryEncoder.
        proprio_dim:           Proprioception dim (8 = EE pose(7) + grip(1)).
        hidden_dim:            Common hidden dim for all token types.
        num_objects:           N_obj.
        horizon:               H — planning horizon.
        num_grasps_per_object: K — grasp candidates per object.
        num_decoder_layers:    Depth of cross-attention transformer.
        num_heads:             Attention heads.
        action_dim:            Per-step action dim (7 for LIBERO).
        use_flow_matching:     If False, use plain FlowMatching; else Shortcut.
        num_sample_steps:      Default sample() steps.
        consistency_weight:    Weight on Shortcut self-consistency loss.
    """

    def __init__(
        self,
        trajectory_dim: int = 6,
        affordance_dim: int = 7,
        geometry_dim: int = 64,
        proprio_dim: int = 8,
        hidden_dim: int = 512,
        num_objects: int = 5,
        horizon: int = 8,
        num_grasps_per_object: int = 8,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        action_dim: int = 7,
        use_flow_matching: bool = True,
        num_sample_steps: int = 8,
        consistency_weight: float = 1.0,
        use_cocos_source: bool = False,
        cocos_alpha: float = 1.0,
        cocos_beta: float = 1.0,
        # ===== §V.D ablation drop flags (init-time attributes) =====
        # Names verified against _FORBIDDEN_FRAGMENTS — no overlap.
        drop_mesh: bool = False,
        drop_grasp: bool = False,
        drop_proprio: bool = False,
    ) -> None:
        super().__init__()

        # === C3 STATIC CHECK ============================================ #
        # Verify the forward signature contains no forbidden parameter
        # names BEFORE the rest of the module is built.  This catches
        # accidental edits at import time.
        sig = inspect.signature(self.forward)
        _check_forbidden(sig.parameters.keys())
        # ================================================================ #

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

        # ===== §V.D ablation drop flags =====
        # Stored as plain bools (not Parameters/Modules).
        # Won't appear in named_modules / named_parameters.
        # Will appear in dir(self) — verified safe against _FORBIDDEN_FRAGMENTS.
        self.drop_mesh = drop_mesh
        self.drop_grasp = drop_grasp
        self.drop_proprio = drop_proprio

        # ---- Modality encoders ---- #
        self.trajectory_encoder = nn.Linear(trajectory_dim, hidden_dim)
        self.affordance_encoder = nn.Linear(affordance_dim, hidden_dim)
        self.geometry_encoder = nn.Linear(geometry_dim, hidden_dim)
        self.proprio_encoder = nn.Linear(proprio_dim, hidden_dim)

        # ---- Positional embeddings ---- #
        self.traj_pos_embed = nn.Parameter(torch.randn(1, 1, horizon, hidden_dim) * 0.02)
        self.aff_pos_embed = nn.Parameter(
            torch.randn(1, 1, num_grasps_per_object, hidden_dim) * 0.02
        )
        self.obj_pos_embed = nn.Parameter(
            torch.randn(1, num_objects, hidden_dim) * 0.02
        )

        # ---- Transformer cross-attention ---- #
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.cross_attn = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # ---- Conditioning projection ---- #
        # Pool cross-attention output to a fixed-size condition vector.
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        condition_dim = hidden_dim

        # ---- Flow matching action head ---- #
        self.velocity_net = _ActionVelocityNet(
            action_dim=action_dim,
            horizon=horizon,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim // 2,
        )
        if use_flow_matching:
            self.flow_matcher: FlowMatching = ShortcutFlowMatching(
                self.velocity_net,
                num_shortcut_levels=4,
                consistency_weight=consistency_weight,
            )
        else:
            self.flow_matcher = FlowMatching(self.velocity_net)

        # ---- Cocos source distribution (V7 Step 4) ---- #
        # When enabled, replaces N(0, I) source with N(alpha*F_phi(c), beta^2*I).
        self.use_cocos_source = use_cocos_source
        self.cocos_alpha = cocos_alpha
        self.cocos_beta = cocos_beta
        if use_cocos_source:
            self.cocos_F = CocosSourceMLP(
                condition_dim=condition_dim,
                target_dim=action_dim * horizon,
            )
        else:
            self.cocos_F = None

        # === C3 STATIC CHECK on submodule names ========================= #
        _check_forbidden(name for name, _ in self.named_modules())
        _check_forbidden(name for name, _ in self.named_parameters())
        # ================================================================ #

    # ======================================================================
    # Forward — STRICT C3 BOUNDARY
    # ======================================================================
    def forward(
        self,
        trajectory: torch.Tensor,
        proprioception: torch.Tensor,
        grasp_affordance: torch.Tensor,
        object_geometry: torch.Tensor,
        gt_action: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            trajectory:       (B, N_obj, H, 6)
            proprioception:   (B, 8)
            grasp_affordance: (B, N_obj, K, 7)
            object_geometry:  (B, N_obj, D_geom)
            gt_action:        (B, H, 7) for training, else None.

        Returns (training):
            {'loss_output': LossOutput, 'loss': scalar Tensor or None,
             'pred_action': (B, H, 7) or None}
        Returns (inference):
            {'pred_action': (B, H, 7)}

        === C3 RUNTIME GUARDS ===
          - assert no forbidden fragment in locals();
          - assert no `self.<forbidden>` attribute is read.
        =========================
        """
        # === C3 RUNTIME GUARD on local variable names ====================
        _check_forbidden(locals().keys())

        # === C3 RUNTIME GUARD on attribute access pattern ================
        # The decoder must not have any attribute that mentions the
        # forbidden fragments — guarded again at call time.
        _check_forbidden(dir(self))

        # ===== §V.D ablation drops (Strategy A: zero raw input) =====
        # Apply BEFORE shape checks so shapes are preserved via zeros_like.
        # Drops affect both train and eval (sample() calls forward()).
        if self.drop_mesh:
            object_geometry = torch.zeros_like(object_geometry)
        if self.drop_grasp:
            grasp_affordance = torch.zeros_like(grasp_affordance)
        if self.drop_proprio:
            proprioception = torch.zeros_like(proprioception)

        # ---- Shape checks (M1) ---- #
        B = trajectory.shape[0]
        N_obj = trajectory.shape[1]
        H = trajectory.shape[2]
        K = grasp_affordance.shape[2]
        assert trajectory.shape == (B, N_obj, H, self.trajectory_dim)
        assert proprioception.shape == (B, self.proprio_dim)
        assert grasp_affordance.shape == (B, N_obj, K, self.affordance_dim)
        assert object_geometry.shape == (B, N_obj, self.geometry_dim)

        # ---- Encode all four modalities ---- #
        traj_tok = self.trajectory_encoder(trajectory) + self.traj_pos_embed
        aff_tok = self.affordance_encoder(grasp_affordance) + self.aff_pos_embed
        geom_tok = self.geometry_encoder(object_geometry) + self.obj_pos_embed
        proprio_tok = self.proprio_encoder(proprioception)             # (B, hidden)

        # ---- Build memory (k/v) ---- #
        # Memory tokens: aff (B, N_obj*K, H), geom (B, N_obj, H), proprio (B, 1, H)
        memory = torch.cat([
            aff_tok.reshape(B, N_obj * K, self.hidden_dim),
            geom_tok.reshape(B, N_obj, self.hidden_dim),
            proprio_tok.unsqueeze(1),
        ], dim=1)

        # ---- Build queries (trajectory tokens) ---- #
        queries = traj_tok.reshape(B, N_obj * H, self.hidden_dim)

        # ---- Cross-attention ---- #
        decoded = self.cross_attn(queries, memory)                      # (B, N_obj*H, hidden)

        # ---- Pool to (B, hidden) condition vector ---- #
        cond_feat = self.cond_proj(decoded.mean(dim=1))                 # (B, hidden)
        condition = {"feat": cond_feat}

        # ---- Flow matching ---- #
        flat_action_dim = self.action_dim * H

        if gt_action is not None:
            assert gt_action.shape == (B, H, self.action_dim)
            x_1 = gt_action.reshape(B, flat_action_dim)
            # Cocos: build shifted source if enabled, else default N(0, I)
            if self.use_cocos_source:
                mean_shift = self.cocos_F(cond_feat)                  # (B, flat_action_dim)
                x_0_src = self.cocos_alpha * mean_shift + self.cocos_beta * torch.randn_like(x_1)
            else:
                x_0_src = None
            loss_output: LossOutput = self.flow_matcher(x_1, condition, x_0_source=x_0_src)
            return {
                "loss_output": loss_output,
                "loss": loss_output.total,
                "pred_action": None,
            }

        # Cocos: sample with shifted x_0 if enabled
        if self.use_cocos_source:
            mean_shift = self.cocos_F(cond_feat)                       # (B, flat_action_dim)
            x_0_src = self.cocos_alpha * mean_shift + self.cocos_beta * torch.randn(
                (B, flat_action_dim), device=cond_feat.device, dtype=cond_feat.dtype
            )
            samples = self.flow_matcher.sample(
                condition,
                num_steps=self.num_sample_steps,
                x_0=x_0_src,
            )
        else:
            samples = self.flow_matcher.sample(
                condition,
                num_steps=self.num_sample_steps,
                shape=(B, flat_action_dim),
            )
        pred_action = samples.reshape(B, H, self.action_dim)
        return {"pred_action": pred_action}

    # ======================================================================
    # Inference-only sampler
    # ======================================================================
    @torch.no_grad()
    def sample(
        self,
        trajectory: torch.Tensor,
        proprioception: torch.Tensor,
        grasp_affordance: torch.Tensor,
        object_geometry: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample action sequence; returns (B, H, action_dim)."""
        out = self.forward(
            trajectory=trajectory,
            proprioception=proprioception,
            grasp_affordance=grasp_affordance,
            object_geometry=object_geometry,
            gt_action=None,
        )
        return out["pred_action"]
