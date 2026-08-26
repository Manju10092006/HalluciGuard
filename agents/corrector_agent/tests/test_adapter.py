"""Step 1 tests — Contracts + Adapter for the Corrector Agent.

Scope (Step 1 only):
    * canonical CorrectionRequest -> internal normalization
    * dict payload coercion
    * use_enum_values string handling
    * authorization lists kept separate (no invent / no merge)
    * internal outcome -> canonical CorrectionResult
    * approved status/validation mapping table
    * changed_claims dict shape
    * invalid/malformed input -> safe failed result (never raises)
    * run_correction scaffold returns a schema-valid CorrectionResult

Targeting, evidence binding, model generation, validation, retry, and
reconstruction are intentionally NOT tested here (later steps).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[3])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest

from orchestration.schemas import (
    CorrectionRequest,
    CorrectionResult,
    ExecutionStatus,
    ValidationStatus,
)

from agents.corrector_agent.corrector import CorrectorAgent, run_correction
from agents.corrector_agent.corrector.adapter import (
    CorrectionInputError,
    build_completed_result,
    build_failed_result,
    build_model_unavailable_result,
    build_partial_result,
    build_reconstruction_fallback_result,
    build_result,
    build_unresolved_result,
    coerce_request,
    to_internal_request,
)
from agents.corrector_agent.corrector.contracts import (
    ChangedClaim,
    InternalCorrectionOutcome,
)


def _assert_schema_valid(result: CorrectionResult) -> None:
    """Prove the result actually validates against the canonical Pydantic model."""
    assert isinstance(result, CorrectionResult)
    revalidated = CorrectionResult.model_validate(result.model_dump())
    assert revalidated == result


# ---------------------------------------------------------------------------
# Canonical -> internal
# ---------------------------------------------------------------------------

def test_to_internal_request_maps_valid_request(valid_correction_request):
    internal = to_internal_request(valid_correction_request)

    assert internal.execution_id == "exec-123"
    assert internal.user_query == "Tell me about Python and the sky."
    assert internal.original_response == (
        "The sky is blue. Python was created by Elon Musk."
    )
    assert internal.correction_instructions == "Preserve c1 exactly. Rewrite c2 using ev-guido."

    # Authorization lists mapped 1:1, kept separate.
    assert len(internal.claims_to_correct) == 1
    assert len(internal.claims_to_preserve) == 1
    assert internal.claims_to_correct[0].claim_id == "c2"
    assert internal.claims_to_preserve[0].claim_id == "c1"

    # Per-claim evidence carried through.
    assert len(internal.claims_to_correct[0].evidence) == 1
    assert internal.claims_to_correct[0].evidence[0].evidence_id == "ev-guido"

    # Top-level evidence buckets mapped.
    assert len(internal.trusted_evidence) == 1
    assert internal.trusted_evidence[0].evidence_id == "ev-guido"
    assert internal.contradictory_evidence == []


def test_to_internal_request_accepts_dict(valid_correction_request):
    payload = valid_correction_request.model_dump()
    internal = to_internal_request(payload)

    assert internal.execution_id == "exec-123"
    assert len(internal.claims_to_correct) == 1
    assert internal.claims_to_correct[0].claim_id == "c2"
    assert internal.trusted_evidence[0].evidence_id == "ev-guido"


def test_enum_values_carried_as_strings(valid_correction_request):
    # Canonical models use use_enum_values=True -> attributes are already strings.
    assert valid_correction_request.claims_to_correct[0].verdict == "contradicted"
    assert valid_correction_request.trusted_evidence[0].entailment_label == "contradiction"

    internal = to_internal_request(valid_correction_request)
    assert internal.claims_to_correct[0].verdict == "contradicted"
    assert internal.claims_to_preserve[0].verdict == "verified"
    assert internal.claims_to_correct[0].evidence[0].entailment_label == "contradiction"
    assert isinstance(internal.claims_to_correct[0].verdict, str)


def test_authorization_lists_are_not_merged_or_invented(valid_correction_request):
    internal = to_internal_request(valid_correction_request)

    correct_ids = {c.claim_id for c in internal.claims_to_correct}
    preserve_ids = {c.claim_id for c in internal.claims_to_preserve}

    # No overlap, no invented targets, counts preserved exactly.
    assert correct_ids == {"c2"}
    assert preserve_ids == {"c1"}
    assert correct_ids.isdisjoint(preserve_ids)
    assert len(internal.claims_to_correct) == len(valid_correction_request.claims_to_correct)
    assert len(internal.claims_to_preserve) == len(valid_correction_request.claims_to_preserve)


def test_empty_authorization_request_maps_cleanly():
    cr = CorrectionRequest(
        execution_id="exec-empty",
        user_query="q",
        original_response="Some response text.",
    )
    internal = to_internal_request(cr)
    assert internal.claims_to_correct == []
    assert internal.claims_to_preserve == []
    assert internal.trusted_evidence == []
    assert internal.original_response == "Some response text."


# ---------------------------------------------------------------------------
# Input coercion / safe failure
# ---------------------------------------------------------------------------

def test_coerce_request_raises_on_malformed_dict():
    with pytest.raises(CorrectionInputError):
        coerce_request({})  # missing required execution_id/user_query/original_response


def test_coerce_request_raises_on_unsupported_type():
    with pytest.raises(CorrectionInputError):
        coerce_request(12345)  # type: ignore[arg-type]


def test_run_correction_malformed_dict_returns_failed_result():
    result = run_correction({"user_query": "q"})  # missing required fields
    _assert_schema_valid(result)
    assert result.status == "failed"
    assert result.validation_status == "invalid"
    # Fail safe: no original recoverable here -> empty string, never raises.
    assert result.corrected_text == result.original_text
    assert result.changed_claims and result.changed_claims[0]["action"] == "input_error"


def test_run_correction_unsupported_type_returns_failed_result():
    result = run_correction(42)  # type: ignore[arg-type]
    _assert_schema_valid(result)
    assert result.status == "failed"
    assert result.validation_status == "invalid"


def test_failed_result_preserves_recoverable_original_from_dict():
    result = run_correction(
        {"user_query": "q", "original_response": "Keep me safe."}  # missing execution_id
    )
    _assert_schema_valid(result)
    assert result.status == "failed"
    assert result.original_text == "Keep me safe."
    assert result.corrected_text == "Keep me safe."  # fail safe: original preserved


# ---------------------------------------------------------------------------
# Internal -> canonical
# ---------------------------------------------------------------------------

def test_build_result_roundtrips_outcome():
    outcome = InternalCorrectionOutcome(
        original_text="orig",
        corrected_text="fixed",
        changed_claims=[
            ChangedClaim(
                claim_id="c2",
                action="corrected",
                original="orig",
                corrected="fixed",
                evidence_ids_used=["ev-guido"],
                attempts=1,
            )
        ],
        validation_status=ValidationStatus.VALID.value,
        attempt_count=2,
        status=ExecutionStatus.COMPLETED.value,
    )
    result = build_result(outcome)
    _assert_schema_valid(result)
    assert result.original_text == "orig"
    assert result.corrected_text == "fixed"
    assert result.validation_status == "valid"
    assert result.status == "completed"
    assert result.attempt_count == 2


def test_changed_claims_serialize_to_plain_dicts():
    result = build_completed_result(
        original_text="orig",
        corrected_text="fixed",
        changed_claims=[
            ChangedClaim(claim_id="c2", action="corrected", original="orig", corrected="fixed")
        ],
    )
    _assert_schema_valid(result)
    assert isinstance(result.changed_claims, list)
    entry = result.changed_claims[0]
    assert isinstance(entry, dict)
    assert entry["claim_id"] == "c2"
    assert entry["action"] == "corrected"
    assert set(entry).issuperset({"claim_id", "action", "original", "corrected"})


def test_attempt_count_is_clamped_to_minimum_one():
    result = build_completed_result("orig", "fixed", attempt_count=0)
    _assert_schema_valid(result)
    assert result.attempt_count == 1  # canonical requires >= 1


@pytest.mark.parametrize(
    # `preserves_original` is declared per row rather than inferred from status:
    # build_completed_result and build_partial_result both emit corrected text
    # (fully vs. partially corrected); the remaining builders must fail safe and
    # return the original verbatim.
    "builder, args, expected_status, expected_validation, preserves_original",
    [
        (build_completed_result, ("orig", "fixed"), "completed", "valid", False),
        (build_partial_result, ("orig", "partly-fixed"), "degraded", "warning", False),
        (build_unresolved_result, ("orig",), "terminated_unresolved", "warning", True),
        (build_model_unavailable_result, ("orig",), "degraded", "unvalidated", True),
        (build_reconstruction_fallback_result, ("orig",), "fallback", "invalid", True),
        (build_failed_result, ("orig", "boom"), "failed", "invalid", True),
    ],
)
def test_approved_status_validation_mapping(
    builder, args, expected_status, expected_validation, preserves_original
):
    result = builder(*args)
    _assert_schema_valid(result)
    assert result.status == expected_status
    assert result.validation_status == expected_validation
    assert result.original_text == "orig"
    if preserves_original:
        assert result.corrected_text == "orig"
    else:
        assert result.corrected_text == args[1]


def test_mapping_uses_only_canonical_enum_values():
    valid_statuses = {s.value for s in ExecutionStatus}
    valid_validations = {v.value for v in ValidationStatus}
    for builder, args in [
        (build_completed_result, ("orig", "fixed")),
        (build_partial_result, ("orig", "partly")),
        (build_unresolved_result, ("orig",)),
        (build_model_unavailable_result, ("orig",)),
        (build_reconstruction_fallback_result, ("orig",)),
        (build_failed_result, ("orig", "boom")),
    ]:
        result = builder(*args)
        assert result.status in valid_statuses
        assert result.validation_status in valid_validations


# ---------------------------------------------------------------------------
# run_correction scaffold (Step 1 passthrough)
# ---------------------------------------------------------------------------

def test_run_correction_scaffold_returns_schema_valid_result(valid_correction_request):
    # Step 6: the real pipeline runs, but the Corrector test suite has no ML stack
    # and no fine-tuned model on disk, so the required model resolves as
    # unavailable. The no-silent-fallback rule then preserves the original response
    # verbatim and reports degraded/unvalidated — never a base-model correction.
    result = run_correction(valid_correction_request)
    _assert_schema_valid(result)
    assert result.original_text == valid_correction_request.original_response
    assert result.corrected_text == valid_correction_request.original_response
    assert result.validation_status == "unvalidated"
    assert result.status == "degraded"
    assert result.attempt_count == 1
    # Specifically the no-model fail-closed path, not some other degraded outcome.
    assert result.changed_claims
    assert result.changed_claims[0]["action"] == "model_unavailable"


def test_corrector_agent_and_run_correction_are_equivalent(valid_correction_request):
    agent_result = CorrectorAgent().correct(valid_correction_request)
    fn_result = run_correction(valid_correction_request)
    assert agent_result.model_dump() == fn_result.model_dump()


def test_run_correction_accepts_dict_request(valid_correction_request):
    # A dict payload takes the same real pipeline; with no model available it also
    # fails closed to degraded/unvalidated with the original preserved.
    result = run_correction(valid_correction_request.model_dump())
    _assert_schema_valid(result)
    assert result.status == "degraded"
    assert result.validation_status == "unvalidated"
    assert result.corrected_text == valid_correction_request.original_response
