"""Canonical <-> internal adapter for the Corrector Agent (Phase 2, Step 1).

Responsibilities:
    * Coerce/validate an incoming payload into the canonical ``CorrectionRequest``
      (accepts either a ``CorrectionRequest`` instance or a plain ``dict`` so a
      supervisor may pass a serialized request).
    * Normalize the canonical request into internal models (``to_internal_request``).
    * Assemble a schema-valid canonical ``CorrectionResult`` from internal
      outcomes, using ONLY the existing canonical enum values.

The canonical models remain the SOLE inter-agent contract. This module is the
only place that translates between canonical and internal representations.

The result-builder functions below encode the APPROVED status/validation
mapping table exactly:

    completed              + valid        -> build_completed_result
    degraded               + warning      -> build_partial_result
    terminated_unresolved  + warning      -> build_unresolved_result
    degraded               + unvalidated  -> build_model_unavailable_result
    fallback               + invalid      -> build_reconstruction_fallback_result
    failed                 + invalid      -> build_failed_result

They perform NO targeting / evidence / model / validation / reconstruction
logic. They are pure result shaping; later steps decide *which* builder to call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from orchestration.schemas import (
    ClaimReport,
    CorrectionRequest,
    CorrectionResult,
    Evidence,
    ExecutionStatus,
    ValidationStatus,
)

from .contracts import (
    ChangedClaim,
    InternalClaim,
    InternalCorrectionOutcome,
    InternalCorrectionRequest,
    InternalEvidence,
)

__all__ = [
    "CorrectionInputError",
    "coerce_request",
    "to_internal_request",
    "make_correction_result",
    "build_result",
    "build_completed_result",
    "build_partial_result",
    "build_unresolved_result",
    "build_model_unavailable_result",
    "build_reconstruction_fallback_result",
    "build_failed_result",
]

RequestLike = Union[CorrectionRequest, Dict[str, Any]]
ChangedClaimLike = Union[ChangedClaim, Dict[str, Any]]


class CorrectionInputError(ValueError):
    """Raised when an incoming payload cannot be coerced to a canonical CorrectionRequest."""


# ---------------------------------------------------------------------------
# Canonical -> internal
# ---------------------------------------------------------------------------

def _enum_value(value: Any) -> str:
    """Return the string value whether given an Enum member or a raw string."""
    return str(getattr(value, "value", value))


def _evidence_to_internal(ev: Evidence) -> InternalEvidence:
    return InternalEvidence(
        evidence_id=ev.evidence_id,
        title=ev.title,
        source=ev.source,
        url=ev.url,
        snippet=ev.snippet,
        entailment_label=_enum_value(ev.entailment_label),
        entailment_score=ev.entailment_score,
        credibility_score=ev.credibility_score,
    )


def _claim_to_internal(claim: ClaimReport) -> InternalClaim:
    return InternalClaim(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        verdict=_enum_value(claim.verdict),
        support_score=claim.support_score,
        contradiction_score=claim.contradiction_score,
        confidence_score=claim.confidence_score,
        evidence=[_evidence_to_internal(e) for e in claim.evidence],
    )


def coerce_request(request: RequestLike) -> CorrectionRequest:
    """Coerce an incoming payload into a validated canonical ``CorrectionRequest``.

    Raises:
        CorrectionInputError: if the payload is neither a ``CorrectionRequest``
            nor a dict that validates against the canonical schema.
    """
    if isinstance(request, CorrectionRequest):
        return request
    if isinstance(request, dict):
        try:
            return CorrectionRequest.model_validate(request)
        except ValidationError as exc:
            raise CorrectionInputError(
                f"payload failed canonical CorrectionRequest validation: {exc}"
            ) from exc
    raise CorrectionInputError(
        f"unsupported request type: {type(request).__name__}"
    )


def to_internal_request(request: RequestLike) -> InternalCorrectionRequest:
    """Normalize a canonical ``CorrectionRequest`` (or dict) into internal models.

    Preserves the Judge's authorization boundary verbatim: ``claims_to_correct``
    and ``claims_to_preserve`` are carried across as separate lists. No target is
    invented, merged, or re-derived here.
    """
    cr = coerce_request(request)
    return InternalCorrectionRequest(
        execution_id=cr.execution_id,
        user_query=cr.user_query,
        original_response=cr.original_response,
        claims_to_correct=[_claim_to_internal(c) for c in cr.claims_to_correct],
        claims_to_preserve=[_claim_to_internal(c) for c in cr.claims_to_preserve],
        trusted_evidence=[_evidence_to_internal(e) for e in cr.trusted_evidence],
        contradictory_evidence=[
            _evidence_to_internal(e) for e in cr.contradictory_evidence
        ],
        correction_instructions=cr.correction_instructions,
    )


# ---------------------------------------------------------------------------
# Internal -> canonical
# ---------------------------------------------------------------------------

def _normalize_changed_claims(
    changed_claims: Optional[List[ChangedClaimLike]],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for cc in changed_claims or []:
        if isinstance(cc, ChangedClaim):
            normalized.append(cc.to_dict())
        elif isinstance(cc, dict):
            normalized.append(dict(cc))
        else:  # pragma: no cover - defensive
            raise CorrectionInputError(
                f"changed_claims entry must be ChangedClaim or dict, got {type(cc).__name__}"
            )
    return normalized


def make_correction_result(
    *,
    original_text: str,
    corrected_text: str,
    validation_status: ValidationStatus,
    status: ExecutionStatus,
    changed_claims: Optional[List[ChangedClaimLike]] = None,
    attempt_count: int = 1,
) -> CorrectionResult:
    """Single choke point that builds a schema-valid canonical ``CorrectionResult``.

    ``attempt_count`` is clamped to the canonical minimum of 1.
    """
    return CorrectionResult(
        original_text=original_text,
        corrected_text=corrected_text,
        changed_claims=_normalize_changed_claims(changed_claims),
        validation_status=ValidationStatus(validation_status),
        attempt_count=max(1, int(attempt_count)),
        status=ExecutionStatus(status),
    )


def build_result(outcome: InternalCorrectionOutcome) -> CorrectionResult:
    """Map an internal pipeline outcome to the canonical ``CorrectionResult``."""
    return make_correction_result(
        original_text=outcome.original_text,
        corrected_text=outcome.corrected_text,
        validation_status=ValidationStatus(outcome.validation_status),
        status=ExecutionStatus(outcome.status),
        changed_claims=outcome.changed_claims,
        attempt_count=outcome.attempt_count,
    )


# --- Approved status/validation mapping builders ---------------------------
# Each wrapper encodes exactly one row of the approved mapping table. Non-success
# builders fail safe by returning the original text unchanged.

def build_completed_result(
    original_text: str,
    corrected_text: str,
    changed_claims: Optional[List[ChangedClaimLike]] = None,
    attempt_count: int = 1,
) -> CorrectionResult:
    """completed + valid — all authorized targets corrected and validated."""
    return make_correction_result(
        original_text=original_text,
        corrected_text=corrected_text,
        validation_status=ValidationStatus.VALID,
        status=ExecutionStatus.COMPLETED,
        changed_claims=changed_claims,
        attempt_count=attempt_count,
    )


def build_partial_result(
    original_text: str,
    corrected_text: str,
    changed_claims: Optional[List[ChangedClaimLike]] = None,
    attempt_count: int = 1,
) -> CorrectionResult:
    """degraded + warning — some targets corrected, some abstained."""
    return make_correction_result(
        original_text=original_text,
        corrected_text=corrected_text,
        validation_status=ValidationStatus.WARNING,
        status=ExecutionStatus.DEGRADED,
        changed_claims=changed_claims,
        attempt_count=attempt_count,
    )


def build_unresolved_result(
    original_text: str,
    changed_claims: Optional[List[ChangedClaimLike]] = None,
    attempt_count: int = 1,
) -> CorrectionResult:
    """terminated_unresolved + warning — nothing safely corrected; original preserved."""
    return make_correction_result(
        original_text=original_text,
        corrected_text=original_text,
        validation_status=ValidationStatus.WARNING,
        status=ExecutionStatus.TERMINATED_UNRESOLVED,
        changed_claims=changed_claims,
        attempt_count=attempt_count,
    )


def build_model_unavailable_result(
    original_text: str,
    reason: str = "model_unavailable",
    changed_claims: Optional[List[ChangedClaimLike]] = None,
) -> CorrectionResult:
    """degraded + unvalidated — required model/adapter unavailable; original returned unchanged."""
    diag = changed_claims or [ChangedClaim(action="model_unavailable", reasons=[reason])]
    return make_correction_result(
        original_text=original_text,
        corrected_text=original_text,
        validation_status=ValidationStatus.UNVALIDATED,
        status=ExecutionStatus.DEGRADED,
        changed_claims=diag,
        attempt_count=1,
    )


def build_reconstruction_fallback_result(
    original_text: str,
    changed_claims: Optional[List[ChangedClaimLike]] = None,
    attempt_count: int = 1,
) -> CorrectionResult:
    """fallback + invalid — reconstruction integrity failed; original returned verbatim."""
    return make_correction_result(
        original_text=original_text,
        corrected_text=original_text,
        validation_status=ValidationStatus.INVALID,
        status=ExecutionStatus.FALLBACK,
        changed_claims=changed_claims,
        attempt_count=attempt_count,
    )


def build_failed_result(
    original_text: str,
    reason: str,
    changed_claims: Optional[List[ChangedClaimLike]] = None,
) -> CorrectionResult:
    """failed + invalid — unexpected/input error; original returned where possible."""
    diag = changed_claims or [ChangedClaim(action="input_error", reasons=[reason])]
    return make_correction_result(
        original_text=original_text,
        corrected_text=original_text,
        validation_status=ValidationStatus.INVALID,
        status=ExecutionStatus.FAILED,
        changed_claims=diag,
        attempt_count=1,
    )
