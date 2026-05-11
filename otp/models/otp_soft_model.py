"""
OTPSoftModel: end-to-end OTP-Soft policy.

Composition:
  backbone           — VLA encoder (OpenVLA-OFT in production; tiny stub here
                       for sanity checks).  Receives (image, instruction) and
                       returns (B, S, backbone_dim) hidden states.
  otp_head           — Predicts SE(3) trajectory distribution p_θ(z | o, ℓ)
                       from backbone hidden states.  Implementation:
                       otp.models.otp_head.OTPHead.
  geometry_encoder   — PointNet-style encoder mapping object meshes to
                       D_geom features.  Independent of ℓ.
  decoder            — LanguageAgnosticDecoder φ_θ(z, w(o)).  Strictly
                       independent of ℓ (Theorem 2 (C3)).

Forward:
  (1) backbone produces hidden states from (image, instruction)
  (2) otp_head predicts trajectory (uses ℓ implicitly via hidden)
  (3) geometry_encoder produces object features (no ℓ)
  (4) decoder predicts action (no ℓ — C3 boundary)

Total loss:
  L = α · L_otp_head + β · L_decoder

Both α and β default to 1.0.  Either contribution is set to None when the
respective sub-module returns LossOutput(total=None) (NaN guard, §2.3).

Usage:
  python -m otp.models.otp_soft_model --sanity-check     # 1-batch forward,
                                                          # no training, no
                                                          # real backbone.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from otp.models.grasp_affordance_encoder import GeometryEncoder
from otp.models.language_agnostic_decoder import LanguageAgnosticDecoder
from otp.models.otp_head import OTPHead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiny backbone stub for the sanity check / unit tests
# ---------------------------------------------------------------------------

class _TinyBackboneStub(nn.Module):
    """
    Minimal stand-in for an OpenVLA-OFT backbone.

    Accepts the same interface as OpenVLABackboneWrapper:
      forward(image, instruction) → dict{'hidden_states', 'attention_mask'}

    Parameter budget (< 5 M):
      pixel_proj  Linear(3, 4096)       ≈  16 K
      token_embed Embedding(1024, 4096) ≈ 4.2 M
      total                             ≈ 4.2 M  ✓
    """

    _VOCAB_SIZE = 1024  # small table keeps param count < 5 M
    _SEQ_LEN = 32       # fixed text sequence length

    def __init__(self, hidden_dim: int = 4096) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pixel_proj = nn.Linear(3, hidden_dim)
        self.token_embed = nn.Embedding(self._VOCAB_SIZE, hidden_dim)

    @staticmethod
    def _hash_instruction(instruction: str) -> torch.Tensor:
        """Hash each word to [0, _VOCAB_SIZE) and pack into (_SEQ_LEN,)."""
        tokens = torch.zeros(_TinyBackboneStub._SEQ_LEN, dtype=torch.long)
        for j, word in enumerate(instruction.split()[: _TinyBackboneStub._SEQ_LEN]):
            tokens[j] = int(hashlib.md5(word.encode()).hexdigest(), 16) % _TinyBackboneStub._VOCAB_SIZE
        return tokens

    def forward(
        self,
        image: torch.Tensor,          # (B, 3, H, W) uint8 or float
        instruction: List[str],       # B task description strings
        **kwargs,
    ) -> dict:
        B = image.shape[0]
        device = image.device

        # Image: normalise → global-avg-pool → project → 1 image token.
        img_f = image.float()
        if img_f.max() > 1.5:        # uint8 [0, 255]
            img_f = img_f / 255.0
        img_pooled = img_f.mean(dim=(-2, -1))                 # (B, 3)
        img_tok = self.pixel_proj(img_pooled).unsqueeze(1)    # (B, 1, D)

        # Text: hash words into vocab range, then embed.
        input_ids = torch.stack(
            [self._hash_instruction(inst) for inst in instruction]
        ).to(device)                                           # (B, S)
        txt_tok = self.token_embed(input_ids)                  # (B, S, D)

        hidden = torch.cat([img_tok, txt_tok], dim=1)          # (B, S+1, D)
        mask = torch.ones(B, hidden.shape[1], dtype=torch.long, device=device)
        return {"hidden_states": hidden, "attention_mask": mask}


# ---------------------------------------------------------------------------
# OTPSoftModel
# ---------------------------------------------------------------------------

class OTPSoftModel(nn.Module):
    """
    End-to-end OTP-Soft policy.

    Args:
        config:    dict with keys (all optional unless noted):
                     backbone_dim          (int, default 256)
                     otp_head              (dict)
                     decoder               (dict)
                     geometry_encoder      (dict)
                     otp_head_loss_weight  (float, default 1.0)
                     decoder_loss_weight   (float, default 1.0)
        backbone:  Optional pre-built backbone module.  If None, a
                   _TinyBackboneStub of width `backbone_dim` is used.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = dict(config)

        backbone_dim: int = config.get("backbone_dim", 4096)
        backbone_mode: str = config.get("backbone_mode", "stub")
        if backbone is None:
            if backbone_mode in ("frozen", "lora", "full"):
                from otp.models.openvla_wrapper import OpenVLABackboneWrapper
                _ckpt = config.get(
                    "backbone_checkpoint",
                    "moojink/openvla-7b-oft-finetuned-libero-spatial",
                )
                backbone = OpenVLABackboneWrapper(checkpoint=_ckpt, mode=backbone_mode)
            else:  # "stub" or any unrecognised value
                backbone = _TinyBackboneStub(hidden_dim=backbone_dim)
        self.backbone = backbone

        # ---- OTP head ---- #
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

        # ---- Geometry encoder ---- #
        geom_cfg = dict(config.get("geometry_encoder", {}))
        self.geometry_encoder = GeometryEncoder(
            num_points=geom_cfg.get("num_points", 256),
            output_dim=geom_cfg.get("output_dim", 64),
            hidden_dim=geom_cfg.get("hidden_dim", 128),
        )

        # ---- Language-agnostic decoder ---- #
        dec_cfg = dict(config.get("decoder", {}))
        self.decoder = LanguageAgnosticDecoder(
            trajectory_dim=dec_cfg.get("trajectory_dim", 6),
            affordance_dim=dec_cfg.get("affordance_dim", 7),
            geometry_dim=geom_cfg.get("output_dim", 64),
            proprio_dim=dec_cfg.get("proprio_dim", 8),
            hidden_dim=dec_cfg.get("hidden_dim", 256),
            num_objects=head_cfg.get("num_objects", 5),
            horizon=head_cfg.get("horizon", 8),
            num_grasps_per_object=dec_cfg.get("num_grasps_per_object", 8),
            num_decoder_layers=dec_cfg.get("num_decoder_layers", 2),
            num_heads=dec_cfg.get("num_heads", 4),
            action_dim=dec_cfg.get("action_dim", 7),
            use_flow_matching=dec_cfg.get("use_flow_matching", True),
            num_sample_steps=dec_cfg.get("num_sample_steps", 4),
            consistency_weight=dec_cfg.get("consistency_weight", 1.0),
            use_cocos_source=dec_cfg.get("use_cocos_source", False),
            cocos_alpha=dec_cfg.get("cocos_alpha", 1.0),
            cocos_beta=dec_cfg.get("cocos_beta", 1.0),
        )

        self.otp_head_loss_weight = float(config.get("otp_head_loss_weight", 1.0))
        self.decoder_loss_weight = float(config.get("decoder_loss_weight", 1.0))
        # Oracle decoder diagnostic (Stage 4d): when True, the decoder
        # receives the GT trajectory instead of the CFM-sampled one. Used
        # to test whether decoder can learn given a clean trajectory signal.
        # Production runs leave this False.
        self.use_oracle_trajectory = bool(config.get("use_oracle_trajectory", False))

    # ------------------------------------------------------------------
    def _run_backbone(self, batch: Dict[str, Any]) -> tuple:
        """Run backbone → (hidden_states, attention_mask)."""
        out = self.backbone(
            image=batch["image"],
            instruction=batch["instruction"],
        )
        hidden = out["hidden_states"]
        attn_mask = out.get("attention_mask")
        return hidden, attn_mask

    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            batch: dict with keys:
              image                (B, 3, H, W) uint8 or float
              instruction          List[str]  B task descriptions
              object_indices       (B, N_obj)  positions of object tokens
              object_point_clouds  (B, N_obj, N_pts, 3)
              proprioception       (B, 8)
              grasp_affordance     (B, N_obj, K, 7)
              gt_trajectory        (B, N_obj, H, 6)  optional (training)
              gt_action            (B, H, 7)         optional (training)

        Returns:
            dict with:
              total_loss      scalar Tensor or None
              otp_head_loss   scalar Tensor or None
              decoder_loss    scalar Tensor or None
              trajectory      (B, N_obj, H, 6)  sampled from OTP head
              pred_action     (B, H, 7) or None
        """
        # ---- 1. Backbone (uses ℓ) ---- #
        hidden, attn_mask = self._run_backbone(batch)

        # ---- 2. OTP head (uses ℓ implicitly via hidden) ---- #
        head_out = self.otp_head(
            backbone_hidden=hidden,
            backbone_attention_mask=attn_mask,
            object_indices=batch["object_indices"],
            gt_trajectory=batch.get("gt_trajectory"),
        )

        if "loss_output" in head_out:
            head_loss = head_out["loss_output"].total
            if head_loss is None:
                # OTP head NaN guard fired (§2.3): skip downstream sampling,
                # return zero trajectory so the decoder's downstream call site
                # (if reached at all) doesn't propagate NaN.  The decoder loss
                # is intentionally not computed because total_loss must remain
                # None to signal "skip backward" upstream.
                B = hidden.shape[0]
                trajectory = torch.zeros(
                    B, self.otp_head.num_objects, self.otp_head.horizon, 6,
                    device=hidden.device, dtype=hidden.dtype,
                )
                # Skip the decoder branch entirely — downstream loss is None.
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
        # ---- 2.5 Oracle override (diagnostic only) ---- #
        if self.use_oracle_trajectory and batch.get("gt_trajectory") is not None:
            # Replace head-derived trajectory with GT trajectory for diagnostic
            # runs. The CFM head is still trained against gt_trajectory via
            # head_loss; only the decoder input is swapped.
            trajectory = batch["gt_trajectory"]

        # ---- 3. Geometry encoder (no ℓ) ---- #
        object_geometry = self.geometry_encoder(batch["object_point_clouds"])

        # ---- 4. Decoder (no ℓ — C3 BOUNDARY) ---- #
        decoder_out = self.decoder(
            trajectory=trajectory,
            proprioception=batch["proprioception"],
            grasp_affordance=batch["grasp_affordance"],
            object_geometry=object_geometry,
            gt_action=batch.get("gt_action"),
        )
        decoder_loss = decoder_out.get("loss")
        pred_action = decoder_out.get("pred_action")

        # ---- 5. Combine losses ---- #
        total_loss: Optional[torch.Tensor] = None
        contributions = []
        if head_loss is not None:
            contributions.append(self.otp_head_loss_weight * head_loss)
        if decoder_loss is not None:
            contributions.append(self.decoder_loss_weight * decoder_loss)
        if contributions:
            total_loss = sum(contributions[1:], contributions[0])

        return {
            "total_loss":      total_loss,
            "otp_head_loss":   head_loss,
            "decoder_loss":    decoder_loss,
            "trajectory":      trajectory,
            "pred_action":     pred_action,
        }


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

def _sanity_check() -> None:
    """1-batch forward through OTPSoftModel with random data; no training."""
    torch.manual_seed(0)
    print("[OTPSoftModel sanity check]")

    config = {
        "backbone_dim": 64,
        "otp_head": {
            "hidden_dim": 64, "num_objects": 2, "horizon": 4,
            "num_heads": 4, "num_layers": 1,
        },
        "decoder": {
            "hidden_dim": 64, "num_decoder_layers": 1,
            "num_heads": 4, "num_grasps_per_object": 4,
        },
        "geometry_encoder": {"num_points": 16, "output_dim": 32},
    }
    model = OTPSoftModel(config).eval()
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):_}")

    B, N_obj, K, N_pts, H = 2, 2, 4, 16, 4
    batch = {
        "image":               torch.rand(B, 3, 16, 16),
        "instruction":         ["pick up the cube and place it on the plate"] * B,
        "object_indices":      torch.arange(N_obj).unsqueeze(0).expand(B, -1),
        "object_point_clouds": torch.randn(B, N_obj, N_pts, 3),
        "proprioception":      torch.randn(B, 8),
        "grasp_affordance":    torch.randn(B, N_obj, K, 7),
        "gt_trajectory":       torch.randn(B, N_obj, H, 6),
        "gt_action":           torch.randn(B, H, 7),
    }
    out = model(batch)

    print(f"  total_loss     = {out['total_loss']}")
    print(f"  otp_head_loss  = {out['otp_head_loss']}")
    print(f"  decoder_loss   = {out['decoder_loss']}")
    print(f"  trajectory.shape = {tuple(out['trajectory'].shape)}")
    print("  inference path:")
    inf_batch = {k: v for k, v in batch.items() if k not in ("gt_trajectory", "gt_action")}
    out_inf = model(inf_batch)
    print(f"  pred_action.shape = {tuple(out_inf['pred_action'].shape)}")
    print("[OK] OTPSoftModel sanity check passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run a 1-batch forward pass without training")
    args = parser.parse_args()
    if args.sanity_check:
        _sanity_check()
    else:
        print("Use --sanity-check to run a 1-batch forward pass.", file=sys.stderr)
        sys.exit(1)
