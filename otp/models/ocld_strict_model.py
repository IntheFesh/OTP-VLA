"""
OCLDStrictModel: OCLD-Strict ablation policy.

Composition (identical to OTPSoftModel except for the decoder):
  backbone           — VLA encoder (same as OTPSoftModel).
  otp_head           — OTPHead (same as OTPSoftModel).
  geometry_encoder   — GeometryEncoder (same as OTPSoftModel).
  decoder            — LanguageAwareDecoder: φ_θ(z, w(o), h_ℓ).
                       VIOLATES C3 — language enters the decoder via
                       backbone_hidden (ablation baseline).

Usage:
  python -m otp.models.ocld_strict_model --sanity-check
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from otp.models.grasp_affordance_encoder import GeometryEncoder
from otp.models.language_aware_decoder import LanguageAwareDecoder
from otp.models.otp_head import OTPHead
from otp.models.otp_soft_model import _TinyBackboneStub

logger = logging.getLogger(__name__)


class OCLDStrictModel(nn.Module):
    """
    OCLD-Strict ablation model.

    Structurally identical to OTPSoftModel but the decoder also receives
    backbone_hidden states (the language signal), intentionally crossing the
    C3 boundary for the purposes of the ablation study.

    Args:
        config:    Same schema as OTPSoftModel.config; additionally accepts:
                     decoder.num_lang_layers  (int, default 2)
        backbone:  Optional pre-built backbone.  Defaults to _TinyBackboneStub.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = dict(config)

        backbone_dim: int = config.get("backbone_dim", 4096)
        if backbone is None:
            backbone = _TinyBackboneStub(hidden_dim=backbone_dim)
        self.backbone = backbone

        # ---- OTP head (shared with OTPSoftModel) ---- #
        head_cfg = dict(config.get("otp_head", {}))
        self.otp_head = OTPHead(
            backbone_dim=backbone_dim,
            hidden_dim=head_cfg.get("hidden_dim", 256),
            num_objects=head_cfg.get("num_objects", 5),
            horizon=head_cfg.get("horizon", 8),
            num_heads=head_cfg.get("num_heads", 4),
            num_layers=head_cfg.get("num_layers", 2),
            vel_hidden_dim=head_cfg.get("vel_hidden_dim", None),
            vel_num_layers=head_cfg.get("vel_num_layers", 2),
            use_shortcut=head_cfg.get("use_shortcut", True),
            num_sample_steps=head_cfg.get("num_sample_steps", 4),
        )

        # ---- Geometry encoder (shared with OTPSoftModel) ---- #
        geom_cfg = dict(config.get("geometry_encoder", {}))
        self.geometry_encoder = GeometryEncoder(
            num_points=geom_cfg.get("num_points", 256),
            output_dim=geom_cfg.get("output_dim", 64),
            hidden_dim=geom_cfg.get("hidden_dim", 128),
        )

        # ---- Language-AWARE decoder (OCLDStrictVariant) ---- #
        dec_cfg = dict(config.get("decoder", {}))
        self.decoder = LanguageAwareDecoder(
            backbone_dim=backbone_dim,
            trajectory_dim=dec_cfg.get("trajectory_dim", 6),
            affordance_dim=dec_cfg.get("affordance_dim", 7),
            geometry_dim=geom_cfg.get("output_dim", 64),
            proprio_dim=dec_cfg.get("proprio_dim", 8),
            hidden_dim=dec_cfg.get("hidden_dim", 256),
            num_objects=head_cfg.get("num_objects", 5),
            horizon=head_cfg.get("horizon", 8),
            num_grasps_per_object=dec_cfg.get("num_grasps_per_object", 8),
            num_decoder_layers=dec_cfg.get("num_decoder_layers", 2),
            num_lang_layers=dec_cfg.get("num_lang_layers", 2),
            num_heads=dec_cfg.get("num_heads", 4),
            action_dim=dec_cfg.get("action_dim", 7),
            use_flow_matching=dec_cfg.get("use_flow_matching", True),
            num_sample_steps=dec_cfg.get("num_sample_steps", 4),
            consistency_weight=dec_cfg.get("consistency_weight", 1.0),
        )

        self.otp_head_loss_weight = float(config.get("otp_head_loss_weight", 1.0))
        self.decoder_loss_weight = float(config.get("decoder_loss_weight", 1.0))

    # ------------------------------------------------------------------
    def _run_backbone(self, batch: Dict[str, Any]) -> tuple:
        out = self.backbone(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        if isinstance(out, dict):
            hidden = out.get("hidden_states", None)
            attn_mask = out.get("attention_mask", None)
            if hidden is None and "last_hidden_state" in out:
                hidden = out["last_hidden_state"]
        else:
            hidden = getattr(out, "last_hidden_state", None) or out.hidden_states[-1]
            attn_mask = batch.get("attention_mask")
        return hidden, attn_mask

    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Same batch contract as OTPSoftModel.forward, with total_loss and
        component losses returned in the same structure.
        """
        # ---- 1. Backbone (uses ℓ) ---- #
        hidden, attn_mask = self._run_backbone(batch)

        # ---- 2. OTP head ---- #
        head_out = self.otp_head(
            backbone_hidden=hidden,
            backbone_attention_mask=attn_mask,
            object_indices=batch["object_indices"],
            gt_trajectory=batch.get("gt_trajectory"),
        )

        if "loss_output" in head_out:
            head_loss = head_out["loss_output"].total
            if head_loss is None:
                B = hidden.shape[0]
                trajectory = torch.zeros(
                    B, self.otp_head.num_objects, self.otp_head.horizon, 6,
                    device=hidden.device, dtype=hidden.dtype,
                )
                return {
                    "total_loss":    None,
                    "otp_head_loss": None,
                    "decoder_loss":  None,
                    "trajectory":    trajectory,
                    "pred_action":   None,
                }
            with torch.no_grad():
                trajectory = self.otp_head.sample(
                    backbone_hidden=hidden,
                    backbone_attention_mask=attn_mask,
                    object_indices=batch["object_indices"],
                )
        else:
            head_loss = None
            trajectory = head_out["trajectories"]

        # ---- 3. Geometry encoder ---- #
        object_geometry = self.geometry_encoder(batch["object_point_clouds"])

        # ---- 4. Decoder (language-aware: receives backbone hidden) ---- #
        decoder_out = self.decoder(
            trajectory=trajectory,
            proprioception=batch["proprioception"],
            grasp_affordance=batch["grasp_affordance"],
            object_geometry=object_geometry,
            backbone_hidden=hidden,
            backbone_attention_mask=attn_mask,
            gt_action=batch.get("gt_action"),
        )
        decoder_loss = decoder_out.get("loss")
        pred_action = decoder_out.get("pred_action")

        # ---- 5. Combine ---- #
        total_loss: Optional[torch.Tensor] = None
        contributions = []
        if head_loss is not None:
            contributions.append(self.otp_head_loss_weight * head_loss)
        if decoder_loss is not None:
            contributions.append(self.decoder_loss_weight * decoder_loss)
        if contributions:
            total_loss = sum(contributions[1:], contributions[0])

        return {
            "total_loss":    total_loss,
            "otp_head_loss": head_loss,
            "decoder_loss":  decoder_loss,
            "trajectory":    trajectory,
            "pred_action":   pred_action,
        }


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

def _sanity_check() -> None:
    torch.manual_seed(0)
    print("[OCLDStrictModel sanity check]")

    config = {
        "backbone_dim": 64,
        "otp_head": {
            "hidden_dim": 64, "num_objects": 2, "horizon": 4,
            "num_heads": 4, "num_layers": 1,
        },
        "decoder": {
            "hidden_dim": 64, "num_decoder_layers": 1, "num_lang_layers": 1,
            "num_heads": 4, "num_grasps_per_object": 4,
        },
        "geometry_encoder": {"num_points": 16, "output_dim": 32},
    }
    model = OCLDStrictModel(config).eval()
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):_}")

    B, S, N_obj, K, N_pts, H = 2, 8, 2, 4, 16, 4
    batch = {
        "pixel_values":        torch.randn(B, 3, 16, 16),
        "input_ids":           torch.randint(0, 1000, (B, S)),
        "attention_mask":      torch.ones(B, S, dtype=torch.long),
        "object_indices":      torch.randint(0, S, (B, N_obj)),
        "object_point_clouds": torch.randn(B, N_obj, N_pts, 3),
        "proprioception":      torch.randn(B, 8),
        "grasp_affordance":    torch.randn(B, N_obj, K, 7),
        "gt_trajectory":       torch.randn(B, N_obj, H, 6),
        "gt_action":           torch.randn(B, H, 7),
    }
    out = model(batch)
    print(f"  total_loss    = {out['total_loss']}")
    print(f"  otp_head_loss = {out['otp_head_loss']}")
    print(f"  decoder_loss  = {out['decoder_loss']}")

    inf_batch = {k: v for k, v in batch.items()
                 if k not in ("gt_trajectory", "gt_action")}
    out_inf = model(inf_batch)
    print(f"  pred_action.shape = {tuple(out_inf['pred_action'].shape)}")
    print("[OK] OCLDStrictModel sanity check passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity-check", action="store_true")
    args = parser.parse_args()
    if args.sanity_check:
        _sanity_check()
    else:
        print("Use --sanity-check to run a 1-batch forward pass.", file=sys.stderr)
        sys.exit(1)
