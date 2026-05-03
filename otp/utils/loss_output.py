"""
Shared LossOutput dataclass (§2.2).

All multi-component loss functions in OTP-VLA must return LossOutput.
Training loop writes diagnostics via to_csv_row() — no manual key access.

Contract (M1): to_csv_row() keys are the single source of truth for CSV column names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Union

import torch


@dataclass
class LossOutput:
    """
    Unified loss container.

    Attributes:
        total:       Scalar tensor used for .backward().  Set to None to signal
                     the training loop that this batch should be skipped (NaN/Inf
                     guard — §2.3).
        components:  Named sub-losses (e.g. "flow", "consistency").
        diagnostics: Non-loss monitoring quantities (grad norms, trigger counts, etc.)
    """

    total: torch.Tensor | None
    components: Dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: Dict[str, Union[float, torch.Tensor]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # M1 — interface contract: single exit point for CSV serialisation
    # ------------------------------------------------------------------
    def to_csv_row(self) -> Dict[str, float]:
        """
        Flatten all fields into a flat dict suitable for CSV writing.

        Keys:
          loss_total           — from self.total
          loss_<name>          — from self.components
          diag_<name>          — from self.diagnostics

        Returns NaN for total when total is None (skipped batch).
        """
        def _scalar(v: Union[torch.Tensor, float]) -> float:
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
            return float(v)

        row: Dict[str, float] = {}

        row["loss_total"] = _scalar(self.total) if self.total is not None else float("nan")

        for k, v in self.components.items():
            row[f"loss_{k}"] = _scalar(v)

        for k, v in self.diagnostics.items():
            row[f"diag_{k}"] = _scalar(v)

        return row

    # ------------------------------------------------------------------
    # Convenience: assert that expected keys are present (M1 validation)
    # ------------------------------------------------------------------
    def assert_keys(self, expected_component_keys: list[str]) -> None:
        """
        Call immediately after constructing LossOutput to verify the
        component dict contains exactly the expected keys (M1).
        """
        missing = [k for k in expected_component_keys if k not in self.components]
        extra = [k for k in self.components if k not in expected_component_keys]
        assert not missing, f"LossOutput missing component keys: {missing}"
        assert not extra, f"LossOutput unexpected component keys: {extra}"
