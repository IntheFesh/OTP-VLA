"""
Tests for otp/models/language_agnostic_decoder.py

Theorem 2 (C3) compliance is the central concern of this test module.

Coverage:
  1. Output shape correctness (training + inference paths).
  2. C3 STATIC: forward signature contains no language-related parameter
     names; submodule names contain no language-related fragments.
  3. C3 DYNAMIC: identical (trajectory, proprio, affordance, geometry) →
     identical output, even if surrounding 'instruction' fields differ.
  4. Loss decreases on synthetic toy data.
  5. NaN guard: NaN input propagates to LossOutput(total=None).
  6. Batch dimension support.
"""

import inspect
from typing import Tuple

import pytest
import torch
import torch.optim as optim

from otp.models.language_agnostic_decoder import (
    LanguageAgnosticDecoder,
    _FORBIDDEN_FRAGMENTS,
    _check_forbidden,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_decoder(
    B: int = 2,
    N_obj: int = 3,
    H: int = 4,
    K: int = 4,
    geom_dim: int = 32,
) -> LanguageAgnosticDecoder:
    return LanguageAgnosticDecoder(
        trajectory_dim=6,
        affordance_dim=7,
        geometry_dim=geom_dim,
        proprio_dim=8,
        hidden_dim=64,
        num_objects=N_obj,
        horizon=H,
        num_grasps_per_object=K,
        num_decoder_layers=2,
        num_heads=4,
        action_dim=7,
        use_flow_matching=True,
        num_sample_steps=4,
    )


def _make_inputs(
    B: int = 2,
    N_obj: int = 3,
    H: int = 4,
    K: int = 4,
    geom_dim: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    trajectory = torch.randn(B, N_obj, H, 6)
    proprioception = torch.randn(B, 8)
    grasp_affordance = torch.randn(B, N_obj, K, 7)
    object_geometry = torch.randn(B, N_obj, geom_dim)
    return trajectory, proprioception, grasp_affordance, object_geometry


# ---------------------------------------------------------------------------
# 1. Forward shape
# ---------------------------------------------------------------------------

class TestForwardShape:

    def test_inference_pred_action_shape(self):
        B, N_obj, H, K = 2, 3, 4, 4
        decoder = _make_decoder(B=B, N_obj=N_obj, H=H, K=K)
        traj, propr, aff, geom = _make_inputs(B=B, N_obj=N_obj, H=H, K=K)
        out = decoder(traj, propr, aff, geom)
        assert "pred_action" in out
        assert out["pred_action"].shape == (B, H, 7)

    def test_training_returns_loss(self):
        decoder = _make_decoder()
        traj, propr, aff, geom = _make_inputs()
        gt_action = torch.randn(2, 4, 7)
        out = decoder(traj, propr, aff, geom, gt_action=gt_action)
        assert "loss" in out
        assert out["loss"] is not None
        assert out["loss"].shape == torch.Size([])

    def test_training_loss_output_components(self):
        decoder = _make_decoder()
        traj, propr, aff, geom = _make_inputs()
        gt_action = torch.randn(2, 4, 7)
        out = decoder(traj, propr, aff, geom, gt_action=gt_action)
        keys = set(out["loss_output"].components.keys())
        assert keys == {"flow", "consistency"}

    def test_training_backward_succeeds(self):
        decoder = _make_decoder()
        traj, propr, aff, geom = _make_inputs()
        gt_action = torch.randn(2, 4, 7)
        out = decoder(traj, propr, aff, geom, gt_action=gt_action)
        out["loss"].backward()        # must not raise


# ---------------------------------------------------------------------------
# 2. C3 STATIC checks
# ---------------------------------------------------------------------------

class TestC3ComplianceStatic:

    def test_forward_signature_has_no_forbidden_param_names(self):
        sig = inspect.signature(LanguageAgnosticDecoder.forward)
        for param_name in sig.parameters.keys():
            low = param_name.lower()
            for frag in _FORBIDDEN_FRAGMENTS:
                assert frag not in low, (
                    f"[C3 VIOLATION] forward param {param_name!r} contains "
                    f"forbidden fragment {frag!r}"
                )

    def test_submodule_names_have_no_forbidden_fragments(self):
        decoder = _make_decoder()
        for name, _ in decoder.named_modules():
            low = name.lower()
            for frag in _FORBIDDEN_FRAGMENTS:
                assert frag not in low, (
                    f"[C3 VIOLATION] submodule {name!r} contains forbidden "
                    f"fragment {frag!r}"
                )

    def test_parameter_names_have_no_forbidden_fragments(self):
        decoder = _make_decoder()
        for name, _ in decoder.named_parameters():
            low = name.lower()
            for frag in _FORBIDDEN_FRAGMENTS:
                assert frag not in low, (
                    f"[C3 VIOLATION] parameter {name!r} contains forbidden "
                    f"fragment {frag!r}"
                )

    def test_check_forbidden_raises(self):
        """The static-check helper itself must fail loudly on a violation."""
        with pytest.raises(AssertionError):
            _check_forbidden(["instruction_tokens"])
        with pytest.raises(AssertionError):
            _check_forbidden(["language_embedding"])
        with pytest.raises(AssertionError):
            _check_forbidden(["backbone_hidden_states"])

    def test_forward_required_params_only(self):
        """forward must accept exactly: trajectory, proprioception,
        grasp_affordance, object_geometry, gt_action (no others)."""
        sig = inspect.signature(LanguageAgnosticDecoder.forward)
        # Drop 'self'.
        param_names = [p for p in sig.parameters.keys() if p != "self"]
        expected = {
            "trajectory",
            "proprioception",
            "grasp_affordance",
            "object_geometry",
            "gt_action",
        }
        assert set(param_names) == expected, (
            f"Unexpected forward signature: {param_names!r}"
        )


# ---------------------------------------------------------------------------
# 3. C3 DYNAMIC check
# ---------------------------------------------------------------------------

class TestC3ComplianceDynamic:

    def test_identical_inputs_give_identical_outputs(self):
        """
        Two 'batches' carry the same (trajectory, proprio, affordance, geometry)
        but different 'instruction' fields in the wrapping dict.  The decoder
        does not even read the instruction field — so its output for the same
        seed is required to match exactly.
        """
        decoder = _make_decoder().eval()
        traj, propr, aff, geom = _make_inputs()

        # Two surrounding dicts with different 'instruction' fields.
        batch_a = {
            "trajectory": traj, "proprioception": propr,
            "grasp_affordance": aff, "object_geometry": geom,
            "instruction": "pick up the red bowl",      # ignored by decoder
        }
        batch_b = {
            "trajectory": traj, "proprioception": propr,
            "grasp_affordance": aff, "object_geometry": geom,
            "instruction": "place the green cup",       # ignored by decoder
        }

        # The decoder.forward signature does not accept 'instruction':
        # passing it as kwarg would TypeError.  This is itself part of C3.
        sig = inspect.signature(decoder.forward)
        assert "instruction" not in sig.parameters

        torch.manual_seed(123)
        out_a = decoder(
            trajectory=batch_a["trajectory"],
            proprioception=batch_a["proprioception"],
            grasp_affordance=batch_a["grasp_affordance"],
            object_geometry=batch_a["object_geometry"],
        )["pred_action"]

        torch.manual_seed(123)
        out_b = decoder(
            trajectory=batch_b["trajectory"],
            proprioception=batch_b["proprioception"],
            grasp_affordance=batch_b["grasp_affordance"],
            object_geometry=batch_b["object_geometry"],
        )["pred_action"]

        assert torch.equal(out_a, out_b), (
            "[C3 VIOLATION] decoder output differed between batches with "
            "identical (z, w(o)) but different 'instruction' fields."
        )

    def test_different_inputs_give_different_outputs(self):
        """Sanity: when (z, w(o)) actually differs, the output should change."""
        decoder = _make_decoder().eval()
        traj_a, propr, aff, geom = _make_inputs()
        traj_b = traj_a + torch.randn_like(traj_a)        # genuinely different z

        torch.manual_seed(7)
        out_a = decoder(traj_a, propr, aff, geom)["pred_action"]
        torch.manual_seed(7)
        out_b = decoder(traj_b, propr, aff, geom)["pred_action"]

        assert not torch.equal(out_a, out_b), (
            "Decoder produced identical output for different trajectories "
            "— it may be ignoring its inputs."
        )


# ---------------------------------------------------------------------------
# 4. Loss decreases on toy data
# ---------------------------------------------------------------------------

class TestLossDecreases:

    def test_loss_decreases_on_synthetic(self):
        torch.manual_seed(11)
        decoder = LanguageAgnosticDecoder(
            trajectory_dim=6, affordance_dim=7, geometry_dim=16, proprio_dim=8,
            hidden_dim=64, num_objects=2, horizon=3, num_grasps_per_object=2,
            num_decoder_layers=1, num_heads=2, action_dim=7,
            use_flow_matching=False, num_sample_steps=2,
        )
        opt = optim.Adam(decoder.parameters(), lr=2e-3)

        # Fixed conditioning + fixed target.
        B, N_obj, H, K = 4, 2, 3, 2
        trajectory = torch.randn(B, N_obj, H, 6)
        proprio = torch.randn(B, 8)
        affordance = torch.randn(B, N_obj, K, 7)
        geometry = torch.randn(B, N_obj, 16)
        gt_action = torch.randn(B, H, 7) * 0.5

        losses = []
        for _ in range(120):
            out = decoder(trajectory, proprio, affordance, geometry,
                          gt_action=gt_action)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            losses.append(out["loss"].item())

        loss_start = sum(losses[:10]) / 10
        loss_end = sum(losses[-10:]) / 10
        assert loss_end < loss_start * 0.85, (
            f"Loss did not decrease: start={loss_start:.4f}, end={loss_end:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. NaN guard
# ---------------------------------------------------------------------------

class TestNaNGuard:

    def test_nan_gt_action_returns_none_loss(self):
        decoder = _make_decoder()
        traj, propr, aff, geom = _make_inputs()
        gt_action = torch.full((2, 4, 7), float("nan"))
        out = decoder(traj, propr, aff, geom, gt_action=gt_action)
        assert out["loss"] is None
        assert out["loss_output"].total is None


# ---------------------------------------------------------------------------
# 6. Batch dimension
# ---------------------------------------------------------------------------

class TestBatchDim:

    @pytest.mark.parametrize("B", [1, 4, 8])
    def test_inference_various_batch(self, B):
        decoder = _make_decoder()
        traj, propr, aff, geom = _make_inputs(B=B)
        out = decoder(traj, propr, aff, geom)
        assert out["pred_action"].shape == (B, 4, 7)

    @pytest.mark.parametrize("B", [1, 4, 8])
    def test_training_various_batch(self, B):
        decoder = _make_decoder()
        traj, propr, aff, geom = _make_inputs(B=B)
        gt_action = torch.randn(B, 4, 7)
        out = decoder(traj, propr, aff, geom, gt_action=gt_action)
        assert out["loss"] is not None


# ---------------------------------------------------------------------------
# 7. sample() helper
# ---------------------------------------------------------------------------

class TestSampleMethod:

    def test_sample_shape(self):
        decoder = _make_decoder().eval()
        traj, propr, aff, geom = _make_inputs()
        out = decoder.sample(traj, propr, aff, geom)
        assert out.shape == (2, 4, 7)
        assert not torch.isnan(out).any()
