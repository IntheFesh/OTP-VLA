"""
Tests for otp/models/otp_soft_model.py

Coverage:
  1. Sanity-check forward (no training): shapes correct, no crash.
  2. Training forward: total_loss is a scalar, backward works.
  3. NaN guard via OTP head: NaN backbone hidden → otp_head_loss is None.
  4. C3 boundary: a monkey-patched backbone that raises on forward must NOT
     be invoked by `model.decoder(...)`.  Decoder must run cleanly without
     touching backbone state.
  5. Geometry encoder: PointNet-style permutation invariance.
"""

import pytest
import torch
import torch.nn as nn

from otp.models.grasp_affordance_encoder import GeometryEncoder
from otp.models.language_agnostic_decoder import LanguageAgnosticDecoder
from otp.models.otp_soft_model import OTPSoftModel, _TinyBackboneStub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_CFG = {
    "backbone_dim": 64,
    "otp_head": {
        "hidden_dim": 64, "num_objects": 2, "horizon": 4,
        "num_heads": 4, "num_layers": 1,
    },
    "decoder": {
        "hidden_dim": 64, "num_decoder_layers": 1, "num_heads": 4,
        "num_grasps_per_object": 4,
    },
    "geometry_encoder": {"num_points": 16, "output_dim": 32},
}


def _make_batch(B: int = 2, training: bool = True):
    S, N_obj, K, N_pts, H = 8, 2, 4, 16, 4
    batch = {
        "pixel_values":        torch.randn(B, 3, 16, 16),
        "input_ids":           torch.randint(0, 1000, (B, S)),
        "attention_mask":      torch.ones(B, S, dtype=torch.long),
        "object_indices":      torch.randint(0, S, (B, N_obj)),
        "object_point_clouds": torch.randn(B, N_obj, N_pts, 3),
        "proprioception":      torch.randn(B, 8),
        "grasp_affordance":    torch.randn(B, N_obj, K, 7),
    }
    if training:
        batch["gt_trajectory"] = torch.randn(B, N_obj, H, 6)
        batch["gt_action"] = torch.randn(B, H, 7)
    return batch


# ---------------------------------------------------------------------------
# 1. Sanity-check forward
# ---------------------------------------------------------------------------

class TestSanityForward:

    def test_inference_pred_action_shape(self):
        torch.manual_seed(0)
        model = OTPSoftModel(DEFAULT_CFG).eval()
        out = model(_make_batch(B=2, training=False))
        assert out["pred_action"].shape == (2, 4, 7)
        assert out["trajectory"].shape == (2, 2, 4, 6)
        assert out["total_loss"] is None

    def test_no_nan_in_inference(self):
        torch.manual_seed(0)
        model = OTPSoftModel(DEFAULT_CFG).eval()
        out = model(_make_batch(B=2, training=False))
        assert not torch.isnan(out["pred_action"]).any()


# ---------------------------------------------------------------------------
# 2. Training forward
# ---------------------------------------------------------------------------

class TestTrainingForward:

    def test_total_loss_is_scalar(self):
        torch.manual_seed(0)
        model = OTPSoftModel(DEFAULT_CFG)
        out = model(_make_batch(B=2, training=True))
        assert out["total_loss"] is not None
        assert out["total_loss"].shape == torch.Size([])

    def test_backward_works(self):
        torch.manual_seed(0)
        model = OTPSoftModel(DEFAULT_CFG)
        out = model(_make_batch(B=2, training=True))
        out["total_loss"].backward()         # must not raise

    def test_both_components_present(self):
        torch.manual_seed(0)
        model = OTPSoftModel(DEFAULT_CFG)
        out = model(_make_batch(B=2, training=True))
        assert out["otp_head_loss"] is not None
        assert out["decoder_loss"] is not None


# ---------------------------------------------------------------------------
# 3. NaN guard via OTP head
# ---------------------------------------------------------------------------

class TestNaNGuard:

    def test_nan_pixel_propagates_to_head_loss_none(self):
        """NaN pixel values cascade into NaN backbone hidden, OTP head guard fires."""
        torch.manual_seed(0)
        model = OTPSoftModel(DEFAULT_CFG)
        batch = _make_batch(B=2, training=True)
        batch["pixel_values"] = torch.full_like(batch["pixel_values"], float("nan"))
        out = model(batch)
        assert out["otp_head_loss"] is None


# ---------------------------------------------------------------------------
# 4. C3 boundary: decoder must not access backbone
# ---------------------------------------------------------------------------

class TestC3DecoderBoundary:

    def test_decoder_runs_without_backbone(self):
        """
        Replace model.backbone with a stub that raises on any call.
        Calling model.decoder(...) directly must succeed — decoder is a
        leaf module that depends only on (trajectory, proprio, affordance,
        geometry), never on backbone state.
        """
        class _RaisingBackbone(nn.Module):
            def forward(self, *a, **kw):
                raise AssertionError(
                    "[C3 VIOLATION] backbone was called from the decoder path"
                )

        torch.manual_seed(0)
        model = OTPSoftModel(DEFAULT_CFG).eval()
        model.backbone = _RaisingBackbone()

        # Build inputs matching the decoder's expected shapes.
        B, N_obj, H, K = 2, 2, 4, 4
        traj = torch.randn(B, N_obj, H, 6)
        propr = torch.randn(B, 8)
        aff = torch.randn(B, N_obj, K, 7)
        # Geometry encoder takes point clouds; decoder expects D_geom features.
        pcs = torch.randn(B, N_obj, 16, 3)
        geom = model.geometry_encoder(pcs)

        # This must NOT call the backbone.
        out = model.decoder(traj, propr, aff, geom)
        assert out["pred_action"].shape == (B, H, 7)

    def test_decoder_attribute_path_no_backbone_reference(self):
        """Module dict of `decoder` must not contain a backbone attribute."""
        model = OTPSoftModel(DEFAULT_CFG)
        decoder_attrs = set(dir(model.decoder))
        for attr in decoder_attrs:
            low = attr.lower()
            assert "backbone" not in low, (
                f"[C3 VIOLATION] decoder has attribute {attr!r} that references "
                f"the backbone path."
            )


# ---------------------------------------------------------------------------
# 5. GeometryEncoder
# ---------------------------------------------------------------------------

class TestGeometryEncoder:

    def test_output_shape(self):
        enc = GeometryEncoder(num_points=64, output_dim=32)
        pc = torch.randn(2, 3, 64, 3)
        out = enc(pc)
        assert out.shape == (2, 3, 32)

    def test_permutation_invariance(self):
        """Max-pool aggregation makes the encoder permutation-invariant."""
        enc = GeometryEncoder(num_points=64, output_dim=32).eval()
        pc = torch.randn(1, 1, 64, 3)
        idx = torch.randperm(64)
        pc_perm = pc[:, :, idx]
        with torch.no_grad():
            a = enc(pc)
            b = enc(pc_perm)
        assert torch.allclose(a, b, atol=1e-6)

    def test_invalid_input_shape(self):
        enc = GeometryEncoder()
        with pytest.raises(ValueError):
            enc(torch.randn(2, 3, 4))     # missing point dim


# ---------------------------------------------------------------------------
# 6. _TinyBackboneStub
# ---------------------------------------------------------------------------

class TestTinyBackboneStub:

    def test_output_shape(self):
        bb = _TinyBackboneStub(hidden_dim=64)
        out = bb(
            pixel_values=torch.randn(2, 3, 16, 16),
            input_ids=torch.randint(0, 1000, (2, 8)),
            attention_mask=torch.ones(2, 8, dtype=torch.long),
        )
        assert out["hidden_states"].shape == (2, 9, 64)         # +1 image token
