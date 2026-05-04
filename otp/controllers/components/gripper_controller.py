"""
GripperController: maps manipulation phase to discrete gripper command.

Interface contract (§REPO_LAYOUT.md Stage 2, M8):
  get_command(phase: str, ee_at_grasp_height: bool = True) -> float ∈ {-1.0, +1.0}
  LIBERO gripper convention: -1.0 = open, +1.0 = closed.

GRASP phase gating:
  In GRASP, the gripper must NOT close until the EE has actually descended
  to the grasp height — closing in mid-air results in an empty gripper.
  Caller passes `ee_at_grasp_height=True` only when |EE.z - grasp_target.z|
  is within tolerance.  When False, GRASP returns -1.0 (keep gripper open
  while OSC continues descending the EE to the grasp pose).
"""

from __future__ import annotations


# Phases where the gripper should be CLOSED (holding the object).
_CLOSED_PHASES = frozenset({"GRASP", "TRANSPORT"})


class GripperController:
    """
    Returns a discrete gripper command (+1 = closed, -1 = open) for each phase.

    Phase → command mapping:
      APPROACH   : -1  (open — fingers clear of object)
      PRE_GRASP  : -1  (open — descending to hover above grasp pose)
      GRASP      : +1  iff EE has reached grasp height; else -1 (still descending)
      TRANSPORT  : +1  (closed — carrying the object)
      RELEASE    : -1  (open — releasing the object)
      DONE       : -1  (open — neutral)
    """

    def get_command(
        self,
        phase: str,
        ee_at_grasp_height: bool = True,
    ) -> float:
        """
        Args:
            phase:               one of APPROACH, PRE_GRASP, GRASP, TRANSPORT,
                                 RELEASE, DONE.
            ee_at_grasp_height:  True iff EE has descended to grasp height
                                 within tolerance. Only consulted when
                                 phase == "GRASP".  Default True keeps the
                                 unit-test friendly behaviour of older
                                 callers that don't pass this argument.

        Returns:
            +1.0 if phase requires gripper closed, -1.0 otherwise.
        """
        if phase == "GRASP" and not ee_at_grasp_height:
            # EE has not yet descended to the grasp height. Keep gripper
            # open so OSC can continue lowering the EE; closing now would
            # close on empty space above the object.
            return -1.0
        return 1.0 if phase in _CLOSED_PHASES else -1.0
