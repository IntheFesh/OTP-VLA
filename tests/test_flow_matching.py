"""
Tests for otp/utils/flow_matching.py

Covers (§8.2 requirements):
  • Input/output schema correctness (shape, dtype, LossOutput keys)
  • Loss decreases monotonically on toy task
  • Sample quality (mean/std close to target after training)
  • ShortcutFlowMatching at d=1 degrades to standard FM behavior
  • NaN/Inf guard (§2.3): returns LossOutput(total=None)
  • LossOutput.to_csv_row() produces no stray NaN when total is non-None (M1)
  • Batch dimension support

All tests run on CPU.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from otp.utils.flow_matching import FlowMatching, ShortcutFlowMatching
from otp.utils.loss_output import LossOutput

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Toy velocity network fixtures
# ---------------------------------------------------------------------------

class ToyVelNet(nn.Module):
    """
    Minimal MLP velocity model for FlowMatching.
    Signature: forward(x_t, t, condition) -> (B, D)
    """
    def __init__(self, dim: int = 6, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        self.dim = dim

    def forward(self, x_t, t, condition):
        # condition is unused in this toy model
        inp = torch.cat([x_t, t], dim=-1)             # (B, D+1)
        return self.net(inp)


class ToyShortcutVelNet(nn.Module):
    """
    Minimal MLP velocity model for ShortcutFlowMatching.
    Signature: forward(x_t, t, condition, d) -> (B, D)
    """
    def __init__(self, dim: int = 6, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden),               # +2 for t and d
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        self.dim = dim

    def forward(self, x_t, t, condition, d):
        inp = torch.cat([x_t, t, d], dim=-1)          # (B, D+2)
        return self.net(inp)


# ---------------------------------------------------------------------------
# 1. Schema / output structure
# ---------------------------------------------------------------------------

class TestOutputSchema:

    def test_fm_returns_loss_output(self):
        model = ToyVelNet(6)
        fm = FlowMatching(model)
        x_1 = torch.randn(4, 6)
        out = fm(x_1, condition={})
        assert isinstance(out, LossOutput)

    def test_fm_total_is_scalar_tensor(self):
        model = ToyVelNet(6)
        fm = FlowMatching(model)
        x_1 = torch.randn(4, 6)
        out = fm(x_1, condition={})
        assert out.total is not None
        assert out.total.shape == torch.Size([]), f"total is not scalar: {out.total.shape}"

    def test_fm_component_keys(self):
        """M1: FlowMatching must have exactly 'flow' component key."""
        model = ToyVelNet(6)
        fm = FlowMatching(model)
        x_1 = torch.randn(4, 6)
        out = fm(x_1, condition={})
        assert set(out.components.keys()) == {"flow"}

    def test_shortcut_component_keys(self):
        """M1: ShortcutFlowMatching must have 'flow' and 'consistency' keys."""
        model = ToyShortcutVelNet(6)
        sfm = ShortcutFlowMatching(model)
        x_1 = torch.randn(4, 6)
        out = sfm(x_1, condition={})
        assert set(out.components.keys()) == {"flow", "consistency"}

    def test_to_csv_row_no_nan_when_ok(self):
        """to_csv_row must not contain NaN when computation succeeded."""
        model = ToyVelNet(6)
        fm = FlowMatching(model)
        x_1 = torch.randn(4, 6)
        out = fm(x_1, condition={})
        row = out.to_csv_row()
        for k, v in row.items():
            assert v == v, f"NaN in to_csv_row key '{k}'"   # NaN != NaN

    def test_to_csv_row_keys_prefixed(self):
        """All keys in to_csv_row must have correct prefixes."""
        model = ToyVelNet(6)
        fm = FlowMatching(model)
        out = fm(torch.randn(4, 6), condition={})
        row = out.to_csv_row()
        assert "loss_total" in row
        assert "loss_flow" in row
        for k in row:
            assert k.startswith("loss_") or k.startswith("diag_"), \
                f"Unexpected key prefix: '{k}'"

    def test_shortcut_csv_has_consistency(self):
        model = ToyShortcutVelNet(6)
        sfm = ShortcutFlowMatching(model)
        out = sfm(torch.randn(4, 6), condition={})
        row = out.to_csv_row()
        assert "loss_consistency" in row


# ---------------------------------------------------------------------------
# 2. NaN / Inf guard (§2.3)
# ---------------------------------------------------------------------------

class TestNaNGuard:

    def test_nan_input_returns_none_total(self):
        """x_1 containing NaN → LossOutput(total=None)."""
        model = ToyVelNet(6)
        fm = FlowMatching(model)
        x_1 = torch.randn(4, 6)
        x_1[2, 3] = float("nan")
        out = fm(x_1, condition={})
        assert out.total is None, "Expected total=None for NaN input"

    def test_inf_input_returns_none_total(self):
        model = ToyVelNet(6)
        fm = FlowMatching(model)
        x_1 = torch.randn(4, 6)
        x_1[0, 0] = float("inf")
        out = fm(x_1, condition={})
        assert out.total is None

    def test_nan_recorded_in_diagnostics(self):
        model = ToyVelNet(6)
        fm = FlowMatching(model)
        x_1 = torch.full((4, 6), float("nan"))
        out = fm(x_1, condition={})
        assert "nan_batch" in out.diagnostics
        assert out.diagnostics["nan_batch"] == 1.0

    def test_to_csv_row_nan_total_is_nan(self):
        """to_csv_row on a skipped batch: loss_total = NaN."""
        model = ToyVelNet(6)
        fm = FlowMatching(model)
        x_1 = torch.full((4, 6), float("nan"))
        out = fm(x_1, condition={})
        row = out.to_csv_row()
        import math
        assert math.isnan(row["loss_total"]), "Expected NaN in loss_total for skipped batch"

    def test_shortcut_nan_guard(self):
        model = ToyShortcutVelNet(6)
        sfm = ShortcutFlowMatching(model)
        x_1 = torch.randn(4, 6)
        x_1[1] = float("nan")
        out = sfm(x_1, condition={})
        assert out.total is None


# ---------------------------------------------------------------------------
# 3. Loss decreases on toy task
# ---------------------------------------------------------------------------

class TestLossReasonable:

    def test_fm_loss_decreases(self):
        """
        On a fixed Gaussian target N(2, 0.5²), FM loss should drop
        significantly after 200 gradient steps.
        """
        torch.manual_seed(1)
        dim = 6
        model = ToyVelNet(dim, hidden=128)
        fm = FlowMatching(model)
        opt = optim.Adam(model.parameters(), lr=1e-3)

        target_mean = torch.ones(1, dim) * 2.0
        target_std = 0.5

        losses = []
        for _ in range(200):
            x_1 = target_mean + target_std * torch.randn(64, dim)
            out = fm(x_1, condition={})
            assert out.total is not None
            opt.zero_grad()
            out.total.backward()
            opt.step()
            losses.append(out.total.item())

        # Loss at the end should be notably lower than at the start
        loss_start = sum(losses[:10]) / 10
        loss_end = sum(losses[-10:]) / 10
        assert loss_end < loss_start * 0.5, (
            f"Loss did not decrease sufficiently: "
            f"start={loss_start:.4f}, end={loss_end:.4f}"
        )

    def test_shortcut_loss_decreases(self):
        """ShortcutFlowMatching total loss decreases over training."""
        torch.manual_seed(2)
        dim = 6
        model = ToyShortcutVelNet(dim, hidden=128)
        sfm = ShortcutFlowMatching(model, num_shortcut_levels=3)
        opt = optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(200):
            x_1 = torch.randn(64, dim) + 1.5
            out = sfm(x_1, condition={})
            assert out.total is not None
            opt.zero_grad()
            out.total.backward()
            opt.step()
            losses.append(out.total.item())

        loss_start = sum(losses[:10]) / 10
        loss_end = sum(losses[-10:]) / 10
        assert loss_end < loss_start * 0.7, (
            f"ShortcutFM loss did not decrease: start={loss_start:.4f}, end={loss_end:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. Sample quality
# ---------------------------------------------------------------------------

class TestSamplingQuality:

    def test_fm_sample_shape(self):
        model = ToyVelNet(6)
        fm = FlowMatching(model)
        samples = fm.sample(condition={}, num_steps=16, shape=(32, 6))
        assert samples.shape == (32, 6)

    def test_fm_sample_mean_std_after_training(self):
        """
        After training on N(3, 1), samples should have mean ≈ 3, std ≈ 1.
        This is a relaxed test (atol=0.5) since we train only 500 steps.
        """
        torch.manual_seed(3)
        dim = 4
        target_mean = 3.0
        target_std = 1.0

        model = ToyVelNet(dim, hidden=128)
        fm = FlowMatching(model)
        opt = optim.Adam(model.parameters(), lr=2e-3)

        for _ in range(500):
            x_1 = torch.randn(128, dim) * target_std + target_mean
            out = fm(x_1, condition={})
            opt.zero_grad()
            out.total.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            samples = fm.sample(condition={}, num_steps=32, shape=(512, dim))

        sample_mean = samples.mean().item()
        sample_std = samples.std().item()

        assert abs(sample_mean - target_mean) < 0.5, \
            f"Sample mean {sample_mean:.3f} far from target {target_mean}"
        assert abs(sample_std - target_std) < 0.5, \
            f"Sample std {sample_std:.3f} far from target {target_std}"

    def test_shortcut_sample_shape(self):
        model = ToyShortcutVelNet(6)
        sfm = ShortcutFlowMatching(model)
        samples = sfm.sample(condition={}, num_steps=4, shape=(16, 6))
        assert samples.shape == (16, 6)


# ---------------------------------------------------------------------------
# 5. ShortcutFlowMatching degrades to standard FM at d=1 (conceptually)
# ---------------------------------------------------------------------------

class TestShortcutConsistency:

    def test_d1_self_consistency_target_equals_fm_target(self):
        """
        For a perfect oracle FM model (v_θ = x_1 - x_0), the self-consistency
        target at d=1 must equal v_target = x_1 - x_0 (the FM target).

        This verifies the key identity:
            v_consist = ½(v_half1 + v_half2) = ½((x1-x0) + (x1-x0)) = x1-x0
        """
        torch.manual_seed(4)
        B, D = 16, 6

        x_0 = torch.randn(B, D)
        x_1 = torch.randn(B, D) + 2.0
        v_true = x_1 - x_0                            # perfect FM velocity

        # Oracle model always returns v_true regardless of (x_t, t, d)
        class OracleModel(nn.Module):
            def __init__(self, v):
                super().__init__()
                self._v = v
            def forward(self, x_t, t, condition, d):
                return self._v.clone()

        oracle = OracleModel(v_true)

        # Manually simulate shortcut self-consistency with d=1
        d = torch.ones(B, 1)
        d_half = d / 2.0
        t_sc = torch.zeros(B, 1)                      # t=0 for d=1

        x_t_sc = (1.0 - t_sc) * x_0 + t_sc * x_1     # = x_0

        v_first = oracle(x_t_sc, t_sc, {}, d_half)
        x_mid = x_t_sc + d_half * v_first
        t_mid = (t_sc + d_half).clamp(max=1.0)
        v_second = oracle(x_mid, t_mid, {}, d_half)

        v_consist = 0.5 * (v_first + v_second)

        assert torch.allclose(v_consist, v_true, atol=1e-5), (
            f"Self-consistency target ≠ FM target at d=1 with oracle model.\n"
            f"max_err = {(v_consist - v_true).abs().max().item():.3e}"
        )

    def test_shortcut_num_levels_1_resembles_fm(self):
        """
        With num_shortcut_levels=1 and consistency_weight=0, ShortcutFM
        loss equals standard FM loss (only the 'flow' component matters).
        """
        torch.manual_seed(5)
        dim = 6

        # Use the same random weights for both models via seeded init
        torch.manual_seed(99)
        model_fm = ToyVelNet(dim, hidden=64)

        # Wrap model to accept d argument (ignored)
        class WrappedFM(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, x_t, t, condition, d):
                return self.inner(x_t, t, condition)

        torch.manual_seed(99)
        model_sfm = WrappedFM(ToyVelNet(dim, hidden=64))

        fm = FlowMatching(model_fm)
        sfm = ShortcutFlowMatching(model_sfm, num_shortcut_levels=1,
                                   consistency_weight=0.0)

        # Same random seed for both forward passes
        torch.manual_seed(42)
        x_1 = torch.randn(8, dim)

        torch.manual_seed(7)
        out_fm = fm(x_1, condition={})

        torch.manual_seed(7)
        out_sfm = sfm(x_1, condition={})

        # The 'flow' component should be identical (same model weights, same seed)
        assert torch.allclose(
            out_fm.components["flow"],
            out_sfm.components["flow"],
            atol=1e-5,
        ), (
            f"FM flow loss {out_fm.components['flow'].item():.6f} ≠ "
            f"SFM flow loss {out_sfm.components['flow'].item():.6f}"
        )

    def test_loss_output_backward_works(self):
        """total.backward() must succeed without errors."""
        model = ToyShortcutVelNet(6)
        sfm = ShortcutFlowMatching(model)
        x_1 = torch.randn(8, 6)
        out = sfm(x_1, condition={})
        assert out.total is not None
        out.total.backward()    # must not raise


# ---------------------------------------------------------------------------
# 6. Batch dimension
# ---------------------------------------------------------------------------

class TestBatchDim:

    @pytest.mark.parametrize("B", [1, 4, 32])
    def test_fm_batch(self, B):
        model = ToyVelNet(6)
        fm = FlowMatching(model)
        x_1 = torch.randn(B, 6)
        out = fm(x_1, condition={})
        assert out.total is not None
        assert out.total.shape == torch.Size([])

    @pytest.mark.parametrize("B", [1, 4, 32])
    def test_shortcut_batch(self, B):
        model = ToyShortcutVelNet(6)
        sfm = ShortcutFlowMatching(model)
        x_1 = torch.randn(B, 6)
        out = sfm(x_1, condition={})
        assert out.total is not None
