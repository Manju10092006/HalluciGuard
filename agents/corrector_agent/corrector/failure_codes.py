"""Step 5 — the failure taxonomy, as a dependency-free enum.

WHY THIS IS ITS OWN MODULE
-------------------------
The gate layer (``validators.py``) and the extraction layer
(``text_features.py``) are deliberately pure: stdlib only, no pydantic, no ML
stack. They still need to name failures, and ``contracts.py`` — the natural home
for the enum — imports pydantic.

Putting the taxonomy here keeps ONE source of truth while letting the two purest
and most safety-critical layers stay importable and testable with nothing
installed. ``contracts.py`` re-exports ``ValidationFailureCode`` from this
module, so ``from .contracts import ValidationFailureCode`` keeps working
unchanged for every other caller.

These codes are INTERNAL diagnostics. None of them is a supervisor-level status,
none appears in ``orchestration/schemas.py``, and none is a new inter-agent
contract.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["ValidationFailureCode", "RETRYABLE_CODES", "TERMINAL_CODES"]


class ValidationFailureCode(str, Enum):
    """Why a candidate was not accepted. Internal diagnostics, never a status."""

    STRUCTURAL_FAILURE = "structural_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    TARGET_MISMATCH = "target_mismatch"
    EVIDENCE_UNSUPPORTED = "evidence_unsupported"
    EVIDENCE_CONTRADICTION = "evidence_contradiction"
    UNSUPPORTED_ADDITION = "unsupported_addition"
    ENTITY_MISMATCH = "entity_mismatch"
    NUMBER_MISMATCH = "number_mismatch"
    DATE_MISMATCH = "date_mismatch"
    NON_MINIMAL_EDIT = "non_minimal_edit"
    MODEL_FAILURE = "model_failure"
    # Guaranteed-no-raise fallback. Any unexpected exception inside a validator
    # becomes this and fails CLOSED (candidate rejected), never open.
    VALIDATION_ERROR = "validation_error"


# A retry can plausibly fix these: the model was asked for something it is
# capable of producing and got it wrong.
RETRYABLE_CODES = frozenset(
    {
        ValidationFailureCode.STRUCTURAL_FAILURE.value,
        ValidationFailureCode.EVIDENCE_CONTRADICTION.value,
        ValidationFailureCode.UNSUPPORTED_ADDITION.value,
        ValidationFailureCode.NUMBER_MISMATCH.value,
        ValidationFailureCode.DATE_MISMATCH.value,
        ValidationFailureCode.ENTITY_MISMATCH.value,
        ValidationFailureCode.NON_MINIMAL_EDIT.value,
        ValidationFailureCode.MODEL_FAILURE.value,
        # EVIDENCE_UNSUPPORTED is conditional: retryable only when supporting
        # evidence actually exists. The gate sets `retryable` per-instance, so it
        # is intentionally absent from this static set.
    }
)

# No prompt change can fix these, so retrying only burns attempts and risks
# pressuring the model into fabricating something that passes.
TERMINAL_CODES = frozenset(
    {
        ValidationFailureCode.AUTHORIZATION_FAILURE.value,
        ValidationFailureCode.TARGET_MISMATCH.value,
        ValidationFailureCode.VALIDATION_ERROR.value,
    }
)
