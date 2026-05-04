"""
GripperController: maps manipulation phase to discrete gripper command.

Interface contract (§REPO_LAYOUT.md Stage 2, M8):
  get_command(phase: str) -> float ∈ {-1.0, +1.0}
  LIBERO gripper convention: -1.0 = open, +1.0 = closed.
"""

from __future__ import annotations


# Phases where the gripper should be CLOSED (holding the object).
_CLOSED_PHASES = frozenset({"GRASP", "TRANSPORT"})


class GripperController:
    """
    Returns a discrete gripper command (+1 = closed, -1 = open) for each phase.

    Phase → command mapping:
      APPROACH   : -1  (open — fingers clear of object)
      PRE_GRASP  : -1  (open — descending to grasp position)
      GRASP      : +1  (closed — grasping the object)
      TRANSPORT  : +1  (closed — carrying the object)
      RELEASE    : -1  (open — releasing the object)
      DONE       : -1  (open — neutral)
    """

    def get_command(self, phase: str) -> float:
        """
        Args:
            phase: one of APPROACH, PRE_GRASP, GRASP, TRANSPORT, RELEASE, DONE.

        Returns:
            +1.0 if phase requires gripper closed, -1.0 otherwise.
        """
        return 1.0 if phase in _CLOSED_PHASES else -1.0
