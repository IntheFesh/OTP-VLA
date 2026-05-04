"""
FailureRecoveryRule: per-phase timeout detection.

Complements PhaseScheduler's global max_phase_steps with phase-specific
timeouts.  Returns an abort flag that FixedManipulationController checks
after each step.

Interface contract (§REPO_LAYOUT.md Stage 2):
  check(phase: str, phase_step_count: int) -> bool
  abort: bool — True once a timeout fires (latched until reset).
  abort_reason: str | None
"""

from __future__ import annotations

from typing import Dict, Optional


# Default per-phase timeouts (steps).  Caller may override via constructor.
_DEFAULT_TIMEOUTS: Dict[str, int] = {
    "APPROACH":   300,
    "PRE_GRASP":  150,
    "GRASP":       80,
    "TRANSPORT":  500,
    "RELEASE":     80,
    "DONE":        999_999,
}


class FailureRecoveryRule:
    """
    Detects per-phase timeouts and latches an abort flag.

    Args:
        phase_timeouts: Optional dict mapping phase name → max steps.
                        Missing phases fall back to _DEFAULT_TIMEOUTS.
    """

    def __init__(self, phase_timeouts: Optional[Dict[str, int]] = None) -> None:
        self._timeouts: Dict[str, int] = dict(_DEFAULT_TIMEOUTS)
        if phase_timeouts:
            self._timeouts.update(phase_timeouts)

        self.abort: bool = False
        self.abort_reason: Optional[str] = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear abort flag and reason (call alongside controller reset)."""
        self.abort = False
        self.abort_reason = None

    # ------------------------------------------------------------------
    def check(self, phase: str, phase_step_count: int) -> bool:
        """
        Check whether the current phase has exceeded its timeout.

        Args:
            phase:            Current phase string.
            phase_step_count: Number of steps spent in this phase so far.

        Returns:
            True if a timeout was detected (abort flag is latched to True).
        """
        if self.abort:
            return True

        timeout = self._timeouts.get(phase, _DEFAULT_TIMEOUTS["APPROACH"])
        if phase_step_count >= timeout:
            self.abort = True
            self.abort_reason = (
                f"Phase '{phase}' timed out after {phase_step_count} steps "
                f"(limit={timeout})."
            )
            return True

        return False
