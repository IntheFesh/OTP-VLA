"""
Tests for otp/models/otp_head.py

Covers:
  • Output shape in training and inference modes.
  • Loss decreases on fixed target trajectories.
  • sample() returns valid-shaped tensors.
  • NaN guard: NaN backbone_hidden → LossOutput(total=None).
  • Batch dimension support.
  • LossOutput keys satisfy M1 contract (assert_keys enforced inside flow_matcher).
"""

import math

import pytest
import torch
import torch.optim as optim

from otp.models.otp_head import OTPHead

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_head(
    B=2, N_obj=3, H=4,
    backbone_dim=64, hidden_dim=32,
    num_heads=4, num_layers=2,
    use_shortcut=True,
):
    return OTPHead(
        backbone_dim=backbone_dim,
        hidden_dim=hidden_dim,
        num_objects=N_obj,
        horizon=H,
        num_heads=num_heads,
        num_layers=num_layers,
        vel_hidden_dim=64,
        vel_num_layers=2,
        use_shortcut=use_shortcut,
        num_sample_steps=4,
    )


def _make_inputs(B=2, S=16, N_obj=3, backbone_dim=64):
    backbone_hidden = torch.randn(B, S, backbone_dim)
    backbone_attention_mask = torch.ones(B, S, dtype=torch.long)
    object_indices = torch.randint(0, S, (B, N_obj))
    return backbone_hidden, backbone_attention_mask, object_indices


# ---------------------------------------------------------------------------
# 1. Output shapes — training mode
# ---------------------------------------------------------------------------

class TestForwardTraining:

    def test_returns_loss_output(self):
        head = _make_head()
        bh, mask, obj_idx = _make_inputs()
        gt_traj = torch.randn(2, 3, 4, 6)
        out = head(bh, mask, obj_idx, gt_trajectory=gt_traj)
        assert "loss_output" in out, "Expected 'loss_output' key in training output"

    def test_total_is_scalar(self):
        head = _make_head()
        bh, mask, obj_idx = _make_inputs()
        gt_traj = torch.randn(2, 3, 4, 6)
        out = head(bh, mask, obj_idx, gt_trajectory=gt_traj)
        assert out["loss_output"].total is not None
        assert out["loss_output"].total.shape == torch.Size([])

    def test_shortcut_component_keys(self):
        """ShortcutFlowMatching must produce {'flow', 'consistency'} keys."""
        head = _make_head(use_shortcut=True)
        bh, mask, obj_idx = _make_inputs()
        gt_traj = torch.randn(2, 3, 4, 6)
        out = head(bh, mask, obj_idx, gt_trajectory=gt_traj)
        keys = set(out["loss_output"].components.keys())
        assert keys == {"flow", "consistency"}, f"Unexpected component keys: {keys}"

    def test_plain_fm_component_keys(self):
        """Plain FlowMatching must produce exactly {'flow'} keys."""
        head = _make_head(use_shortcut=False)
        bh, mask, obj_idx = _make_inputs()
        gt_traj = torch.randn(2, 3, 4, 6)
        out = head(bh, mask, obj_idx, gt_trajectory=gt_traj)
        keys = set(out["loss_output"].components.keys())
        assert keys == {"flow"}, f"Unexpected component keys: {keys}"

    def test_backward_succeeds(self):
        """total.backward() must not raise."""
        head = _make_head()
        bh, mask, obj_idx = _make_inputs()
        gt_traj = torch.randn(2, 3, 4, 6)
        out = head(bh, mask, obj_idx, gt_trajectory=gt_traj)
        out["loss_output"].total.backward()   # must not raise


# ---------------------------------------------------------------------------
# 2. Output shapes — inference mode
# ---------------------------------------------------------------------------

class TestForwardInference:

    def test_returns_trajectories_key(self):
        head = _make_head()
        bh, mask, obj_idx = _make_inputs()
        out = head(bh, mask, obj_idx)
        assert "trajectories" in out, "Expected 'trajectories' key in inference output"

    def test_trajectory_shape(self):
        B, N_obj, H = 2, 3, 4
        head = _make_head(B=B, N_obj=N_obj, H=H)
        bh, mask, obj_idx = _make_inputs(B=B, N_obj=N_obj)
        out = head(bh, mask, obj_idx)
        assert out["trajectories"].shape == (B, N_obj, H, 6), \
            f"Expected ({B},{N_obj},{H},6), got {out['trajectories'].shape}"


# ---------------------------------------------------------------------------
# 3. sample() method
# ---------------------------------------------------------------------------

class TestSample:

    def test_sample_shape(self):
        B, N_obj, H = 3, 2, 5
        head = _make_head(B=B, N_obj=N_obj, H=H, backbone_dim=64, hidden_dim=32)
        bh, mask, obj_idx = _make_inputs(B=B, S=10, N_obj=N_obj, backbone_dim=64)
        traj = head.sample(bh, mask, obj_idx, num_steps=4)
        assert traj.shape == (B, N_obj, H, 6), \
            f"Expected ({B},{N_obj},{H},6), got {traj.shape}"

    def test_sample_no_nan(self):
        head = _make_head()
        bh, mask, obj_idx = _make_inputs()
        traj = head.sample(bh, mask, obj_idx, num_steps=4)
        assert not torch.isnan(traj).any(), "sample() produced NaN"
        assert not torch.isinf(traj).any(), "sample() produced Inf"


# ---------------------------------------------------------------------------
# 4. NaN guard (§2.3)
# ---------------------------------------------------------------------------

class TestNaNGuard:

    def test_nan_backbone_training_returns_none_total(self):
        head = _make_head()
        bh = torch.full((2, 16, 64), float("nan"))
        mask = torch.ones(2, 16, dtype=torch.long)
        obj_idx = torch.randint(0, 16, (2, 3))
        gt_traj = torch.randn(2, 3, 4, 6)
        out = head(bh, mask, obj_idx, gt_trajectory=gt_traj)
        assert out["loss_output"].total is None, \
            "NaN backbone should produce LossOutput(total=None)"

    def test_nan_backbone_inference_returns_zeros(self):
        head = _make_head()
        bh = torch.full((2, 16, 64), float("nan"))
        mask = torch.ones(2, 16, dtype=torch.long)
        obj_idx = torch.randint(0, 16, (2, 3))
        out = head(bh, mask, obj_idx)
        assert "trajectories" in out
        assert out["trajectories"].shape == (2, 3, 4, 6)
        assert (out["trajectories"] == 0).all()

    def test_nan_backbone_diagnostics(self):
        head = _make_head()
        bh = torch.full((2, 16, 64), float("nan"))
        mask = torch.ones(2, 16, dtype=torch.long)
        obj_idx = torch.randint(0, 16, (2, 3))
        gt_traj = torch.randn(2, 3, 4, 6)
        out = head(bh, mask, obj_idx, gt_trajectory=gt_traj)
        assert "nan_backbone" in out["loss_output"].diagnostics


# ---------------------------------------------------------------------------
# 5. Loss decreases
# ---------------------------------------------------------------------------

class TestLossDecreases:

    def test_loss_decreases_after_training(self):
        """
        Train on a fixed Gaussian target trajectory — loss must drop by ≥10%
        over 100 gradient steps (relaxed: accounts for flow matching noise).
        """
        torch.manual_seed(10)
        N_obj, H = 2, 3
        head = OTPHead(
            backbone_dim=32, hidden_dim=32, num_objects=N_obj, horizon=H,
            num_heads=2, num_layers=1, vel_hidden_dim=64, vel_num_layers=2,
            use_shortcut=False, num_sample_steps=4,
        )
        opt = optim.Adam(head.parameters(), lr=2e-3)

        B, S = 4, 8
        gt_traj = torch.randn(B, N_obj, H, 6) * 0.5

        losses = []
        for step in range(100):
            torch.manual_seed(step)
            bh = torch.randn(B, S, 32)
            mask = torch.ones(B, S, dtype=torch.long)
            obj_idx = torch.randint(0, S, (B, N_obj))
            out = head(bh, mask, obj_idx, gt_trajectory=gt_traj)
            loss = out["loss_output"].total
            assert loss is not None
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        loss_start = sum(losses[:10]) / 10
        loss_end = sum(losses[-10:]) / 10
        assert loss_end < loss_start * 0.9, (
            f"Loss did not decrease: start={loss_start:.4f}, end={loss_end:.4f}"
        )


# ---------------------------------------------------------------------------
# 6. Batch dimension support
# ---------------------------------------------------------------------------

class TestBatchDim:

    @pytest.mark.parametrize("B", [1, 4, 8])
    def test_training_various_batch(self, B):
        head = _make_head(N_obj=2, H=3)
        bh, mask, obj_idx = _make_inputs(B=B, S=12, N_obj=2)
        gt_traj = torch.randn(B, 2, 3, 6)
        out = head(bh, mask, obj_idx, gt_trajectory=gt_traj)
        assert out["loss_output"].total is not None

    @pytest.mark.parametrize("B", [1, 4, 8])
    def test_inference_various_batch(self, B):
        N_obj, H = 2, 3
        head = _make_head(N_obj=N_obj, H=H)
        bh, mask, obj_idx = _make_inputs(B=B, S=12, N_obj=N_obj)
        out = head(bh, mask, obj_idx)
        assert out["trajectories"].shape == (B, N_obj, H, 6)

    def test_none_attention_mask(self):
        """backbone_attention_mask=None must not raise."""
        head = _make_head()
        bh = torch.randn(2, 16, 64)
        obj_idx = torch.randint(0, 16, (2, 3))
        gt_traj = torch.randn(2, 3, 4, 6)
        out = head(bh, None, obj_idx, gt_trajectory=gt_traj)
        assert out["loss_output"].total is not None
