"""Internal implementation models for the Corrector Agent (Phase 2).

IMPORTANT — SCOPE BOUNDARY:
    These models are INTERNAL to the Corrector. They are NOT an inter-agent
    contract and must never be used as a handoff schema or imported by another
    agent. The ONLY canonical inter-agent contracts are ``CorrectionRequest``
    (input) and ``CorrectionResult`` (output) defined in
    ``orchestration/schemas.py``.

Step 1 introduces the normalization + result-assembly models needed to
establish the canonical boundary:

    CorrectionRequest -> [adapter] -> InternalCorrectionRequest
    InternalCorrectionOutcome -> [adapter] -> CorrectionResult

Models for later steps (sentence spans, correction targets, candidates,
validation feedback, reconstruction results) are added by those steps.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .failure_codes import ValidationFailureCode

__all__ = [
    "InternalEvidence",
    "InternalClaim",
    "InternalCorrectionRequest",
    "ChangedClaim",
    "InternalCorrectionOutcome",
    "MatchStrategy",
    "SkipReason",
    "SentenceSpan",
    "CorrectionTarget",
    "SkippedClaim",
    "CorrectionPlan",
    "EvidenceRole",
    "EvidenceClassifier",
    "BoundEvidence",
    "GroundedTarget",
    "GroundedPlan",
    "PromptBuildError",
    "PromptBundle",
    "ModelKind",
    "ModelUnavailableReason",
    "ModelStatus",
    "CandidateStatus",
    "RejectionReason",
    "CorrectionCandidate",
    "GenerationOutcome",
    "ValidationFailureCode",
    "ValidationFailure",
    "CandidateValidation",
    "TargetDecision",
    "CandidateAttempt",
    "TargetResolution",
    "ValidationOutcome",
]


class InternalEvidence(BaseModel):
    """Normalized, decoupled view of a canonical ``Evidence`` item.

    Enum fields from the canonical schema are carried as plain strings because
    the canonical models use ``use_enum_values=True`` (values, not members).
    """

    model_config = ConfigDict(extra="ignore")

    evidence_id: str
    title: str = ""
    source: str = ""
    url: Optional[str] = None
    snippet: str = ""
    entailment_label: str = "neutral"
    entailment_score: float = 0.0
    credibility_score: float = 0.0


class InternalClaim(BaseModel):
    """Normalized, decoupled view of a canonical ``ClaimReport``.

    ``evidence`` carries the claim's OWN per-claim evidence verbatim. Binding
    logic (which evidence a target may use, and the no-broad-fallback rule) is
    Step 3 and does not live here.
    """

    model_config = ConfigDict(extra="ignore")

    claim_id: str
    claim_text: str
    verdict: str = ""
    support_score: float = 0.0
    contradiction_score: float = 0.0
    confidence_score: float = 0.0
    evidence: List[InternalEvidence] = Field(default_factory=list)


class InternalCorrectionRequest(BaseModel):
    """Normalized internal view of a canonical ``CorrectionRequest``.

    The authorization lists (``claims_to_correct`` / ``claims_to_preserve``) are
    kept strictly separate exactly as supplied by the Judge. The Corrector never
    merges, re-derives, or invents these lists.
    """

    model_config = ConfigDict(extra="ignore")

    execution_id: str
    user_query: str
    original_response: str
    claims_to_correct: List[InternalClaim] = Field(default_factory=list)
    claims_to_preserve: List[InternalClaim] = Field(default_factory=list)
    trusted_evidence: List[InternalEvidence] = Field(default_factory=list)
    contradictory_evidence: List[InternalEvidence] = Field(default_factory=list)
    correction_instructions: str = ""


class ChangedClaim(BaseModel):
    """Typed representation of a single ``changed_claims`` entry.

    Serializes to a plain ``dict`` for the canonical
    ``CorrectionResult.changed_claims`` (which is ``List[Dict[str, Any]]``).
    This is where genuine per-claim diagnostics live WITHOUT modifying the
    canonical inter-agent contract. Only values produced by real execution are
    recorded here; no fabricated metrics.
    """

    model_config = ConfigDict(extra="ignore")

    claim_id: str = ""
    action: str = ""  # e.g. corrected | preserved | abstained | input_error | model_unavailable
    original: str = ""
    corrected: str = ""
    evidence_ids_used: List[str] = Field(default_factory=list)
    attempts: int = 0
    reasons: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the ChangedClaim to a plain dictionary."""
        return self.model_dump()


class InternalCorrectionOutcome(BaseModel):
    """Internal pipeline result, mapped to the canonical ``CorrectionResult``.

    ``validation_status`` and ``status`` hold canonical enum *values* (strings)
    from ``ValidationStatus`` and ``ExecutionStatus`` respectively. The adapter
    is the single choke point that converts these back into the canonical model.
    """

    model_config = ConfigDict(extra="ignore")

    original_text: str
    corrected_text: str
    changed_claims: List[ChangedClaim] = Field(default_factory=list)
    validation_status: str
    attempt_count: int = 1
    status: str


# ---------------------------------------------------------------------------
# Step 2 — Authorization + Claim Targeting (internal only)
# ---------------------------------------------------------------------------

class MatchStrategy(str, Enum):
    """How a claim was resolved to a sentence span. Ordered most→least strict."""

    EXACT = "exact"
    NORMALIZED = "normalized"
    CONTAINMENT = "containment"
    TOKEN_OVERLAP = "token_overlap"


class SkipReason(str, Enum):
    """Why an authorized claim produced no correction target (always fail safe)."""

    CLAIM_NOT_FOUND = "claim_not_found"
    AMBIGUOUS_MATCH = "ambiguous_match"
    SPAN_LOCKED_BY_PRESERVED_CLAIM = "span_locked_by_preserved_claim"
    CLAIM_ALSO_IN_PRESERVE_LIST = "claim_also_in_preserve_list"
    EMPTY_CLAIM_TEXT = "empty_claim_text"
    NO_SENTENCE_SPANS = "no_sentence_spans"
    SPAN_INTEGRITY_FAILED = "span_integrity_failed"
    # Step 3 — target located and authorized, but no groundable evidence exists.
    NO_USABLE_EVIDENCE = "no_usable_evidence"


class SentenceSpan(BaseModel):
    """A byte-accurate sentence span of the ORIGINAL response.

    Invariant enforced by ``sentence_utils``:
        ``original_response[start_offset:end_offset] == text``
    """

    model_config = ConfigDict(extra="ignore")

    sentence_id: str
    text: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)

    @property
    def length(self) -> int:
        """Return the length of the sentence span in characters."""
        return self.end_offset - self.start_offset


class CorrectionTarget(BaseModel):
    """An AUTHORIZED, located correction site.

    A target exists only when the Judge authorized the claim via
    ``claims_to_correct`` AND the claim was unambiguously located in the
    original response AND the span is not locked by a preserved claim.

    ``claim_ids`` is a list because one sentence may carry several authorized
    claims; no authorization is dropped when claims collapse onto one span.
    """

    model_config = ConfigDict(extra="ignore")

    sentence_id: str
    claim_ids: List[str] = Field(default_factory=list)
    claim_texts: List[str] = Field(default_factory=list)
    original_text: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    match_strategy: str = MatchStrategy.EXACT.value
    match_score: float = 1.0


class SkippedClaim(BaseModel):
    """An authorized claim that was deliberately NOT targeted (fail safe)."""

    model_config = ConfigDict(extra="ignore")

    claim_id: str
    claim_text: str = ""
    reason: str
    detail: str = ""


class CorrectionPlan(BaseModel):
    """Output of Step 2 targeting. Carries no evidence and no generated text.

    ``targets`` is the complete, closed set of sites later steps may rewrite.
    Anything not in ``targets`` must be reproduced verbatim.
    """

    model_config = ConfigDict(extra="ignore")

    spans: List[SentenceSpan] = Field(default_factory=list)
    targets: List[CorrectionTarget] = Field(default_factory=list)
    locked_sentence_ids: List[str] = Field(default_factory=list)
    skipped_claims: List[SkippedClaim] = Field(default_factory=list)
    unlocated_preserved_claims: List[SkippedClaim] = Field(default_factory=list)
    integrity_verified: bool = False

    @property
    def has_targets(self) -> bool:
        """Return True if there are any correction targets in the plan."""
        return bool(self.targets)


# ---------------------------------------------------------------------------
# Step 3 — Evidence Binding (internal only)
# ---------------------------------------------------------------------------

class EvidenceRole(str, Enum):
    """Role a bound evidence item plays for its own claim.

    Kept distinct so supporting and contradictory evidence can never be
    conflated downstream (Step 3 requirement 8).
    """

    SUPPORTING = "supporting"
    CONTRADICTORY = "contradictory"
    NEUTRAL = "neutral"


class EvidenceClassifier(str, Enum):
    """Which signal decided an item's role — recorded for auditability."""

    TRUSTED_POOL = "trusted_pool"
    CONTRADICTORY_POOL = "contradictory_pool"
    POOL_CONFLICT = "pool_conflict"
    ENTAILMENT_LABEL = "entailment_label"
    ROLE_CONFLICT = "role_conflict"


class BoundEvidence(BaseModel):
    """One evidence item bound to a target, with full provenance preserved.

    ``evidence`` is carried VERBATIM from the claim's own evidence list. This
    model never edits snippet, source, url, title, or any score.
    """

    model_config = ConfigDict(extra="ignore")

    evidence: InternalEvidence
    role: str
    classified_by: str
    # Step 3 requirement 6: evidence text is DATA, never instructions. Flagged
    # here (never rewritten) so the Step 4 prompt layer can fence it.
    suspected_instruction_text: bool = False

    @property
    def evidence_id(self) -> str:
        """Return the evidence_id from the bound evidence item."""
        return self.evidence.evidence_id


class GroundedTarget(BaseModel):
    """A correction target bound to only its own claims' evidence."""

    model_config = ConfigDict(extra="ignore")

    target: CorrectionTarget
    supporting: List[BoundEvidence] = Field(default_factory=list)
    contradictory: List[BoundEvidence] = Field(default_factory=list)
    neutral: List[BoundEvidence] = Field(default_factory=list)

    @property
    def sentence_id(self) -> str:
        """Return the sentence_id from the correction target."""
        return self.target.sentence_id

    @property
    def has_usable_support(self) -> bool:
        """Return True if the target has at least one supporting evidence item."""
        return bool(self.supporting)

    @property
    def all_evidence_ids(self) -> List[str]:
        """Return all evidence IDs across supporting, contradictory, and neutral evidence."""
        return [
            b.evidence_id
            for b in (*self.supporting, *self.contradictory, *self.neutral)
        ]


class GroundedPlan(BaseModel):
    """Output of Step 3: targets that are authorized, located, AND grounded.

    ``grounded_targets`` is the closed set of sites later steps may rewrite.
    Targets dropped for lack of groundable evidence appear in ``skipped_claims``
    with reason ``no_usable_evidence`` — never silently discarded.
    """

    model_config = ConfigDict(extra="ignore")

    spans: List[SentenceSpan] = Field(default_factory=list)
    grounded_targets: List[GroundedTarget] = Field(default_factory=list)
    locked_sentence_ids: List[str] = Field(default_factory=list)
    skipped_claims: List[SkippedClaim] = Field(default_factory=list)
    unlocated_preserved_claims: List[SkippedClaim] = Field(default_factory=list)
    # Auditable record of evidence that could not be bound (e.g. blank
    # evidence_id, so provenance could not be preserved) or whose role was
    # ambiguous. Nothing is ever dropped silently.
    provenance_warnings: List[str] = Field(default_factory=list)
    integrity_verified: bool = False

    @property
    def has_targets(self) -> bool:
        """Return True if there are any grounded targets in the plan."""
        return bool(self.grounded_targets)


# ---------------------------------------------------------------------------
# Step 4 — Prompt + Model + Structured Output (internal only)
# ---------------------------------------------------------------------------

class PromptBuildError(str, Enum):
    """Why a per-target prompt could not be built (always fail safe: no prompt)."""

    PLAN_NOT_VERIFIED = "plan_not_verified"
    EMPTY_TARGET_TEXT = "empty_target_text"
    NO_SUPPORTING_EVIDENCE = "no_supporting_evidence"
    FENCE_COLLISION = "fence_collision"
    BUDGET_EXCEEDED = "budget_exceeded"


class PromptBundle(BaseModel):
    """A single per-target model input, with its sections kept separable.

    ``system_text`` holds INSTRUCTIONS ONLY. ``prompt_text`` holds the fenced
    DATA sections plus the (never-truncatable) output mandate. Nothing in the
    data sections is ever presented to the model as an instruction.

    ``evidence_ids`` lists exactly the SUPPORTING evidence actually rendered as
    grounding material, in render order. ``excluded_evidence_ids`` records the
    contradictory/neutral evidence that was deliberately withheld from the model
    (Step 4 requirement 7) so the exclusion stays auditable.
    """

    model_config = ConfigDict(extra="ignore")

    sentence_id: str
    claim_ids: List[str] = Field(default_factory=list)
    original_sentence: str = ""
    system_text: str = ""
    prompt_text: str = ""
    fence_tag: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    excluded_evidence_ids: List[str] = Field(default_factory=list)
    flagged_evidence_ids: List[str] = Field(default_factory=list)
    dropped_sections: List[str] = Field(default_factory=list)
    estimated_tokens: int = 0
    build_ok: bool = False
    build_error: str = ""
    build_detail: str = ""


class ModelKind(str, Enum):
    """What was actually loaded. Never a guess — recorded from resolution."""

    MERGED_FINETUNED = "merged_finetuned"
    BASE_PLUS_ADAPTER = "base_plus_adapter"
    # Only reachable when an operator explicitly sets
    # ``allow_base_model_fallback=True``. Never a silent default.
    BASE_MODEL_OPTIN = "base_model_optin"
    # Test/integration seam. Deliberately named so it can never be mistaken for
    # the fine-tuned corrector model.
    INJECTED_TEST_GENERATOR = "injected_test_generator"


class ModelUnavailableReason(str, Enum):
    """Explicit, non-silent reasons the correction model could not be used."""

    DEPENDENCY_MISSING = "dependency_missing"
    MODEL_PATH_MISSING = "model_path_missing"
    MODEL_ARTIFACTS_INCOMPLETE = "model_artifacts_incomplete"
    ADAPTER_CONFIG_INVALID = "adapter_config_invalid"
    ADAPTER_WEIGHTS_MISSING = "adapter_weights_missing"
    BASE_MODEL_FALLBACK_FORBIDDEN = "base_model_fallback_forbidden"
    TOKENIZER_LOAD_FAILED = "tokenizer_load_failed"
    ADAPTER_LOAD_FAILED = "adapter_load_failed"
    MODEL_LOAD_FAILED = "model_load_failed"
    GENERATION_FAILED = "generation_failed"


class ModelStatus(BaseModel):
    """Resolved availability of the correction model.

    SAFETY: ``available=False`` must never be reported as a successful
    correction. Step 5 maps this onto the canonical degraded/fallback mapping.
    """

    model_config = ConfigDict(extra="ignore")

    available: bool = False
    kind: str = ""
    resolved_path: str = ""
    adapter_path: str = ""
    base_model: str = ""
    reason: str = ""
    detail: str = ""
    tried_paths: List[str] = Field(default_factory=list)

    @property
    def is_finetuned(self) -> bool:
        """True only when genuine fine-tuned corrector weights were loaded."""
        return self.kind in (
            ModelKind.MERGED_FINETUNED.value,
            ModelKind.BASE_PLUS_ADAPTER.value,
        )


class CandidateStatus(str, Enum):
    """Outcome of one model generation for one authorized target."""

    PROPOSED = "proposed"
    REJECTED = "rejected"


class RejectionReason(str, Enum):
    """Why a model output was rejected instead of repaired (requirement 13)."""

    EMPTY_OUTPUT = "empty_output"
    NOT_JSON = "not_json"
    NOT_JSON_OBJECT = "not_json_object"
    MISSING_FIELD = "missing_field"
    UNEXPECTED_FIELDS = "unexpected_fields"
    WRONG_TYPE = "wrong_type"
    EMPTY_CORRECTED_SENTENCE = "empty_corrected_sentence"
    SENTENCE_ID_MISMATCH = "sentence_id_mismatch"
    PROMPT_BUILD_FAILED = "prompt_build_failed"
    GENERATION_FAILED = "generation_failed"


class CorrectionCandidate(BaseModel):
    """An UNVALIDATED per-target model proposal.

    ``status == proposed`` means only "structurally well-formed for the
    authorized target". It does NOT mean validated, grounded, or safe to splice:
    evidence alignment, unsupported-content checks and retry are Step 5, and
    splicing is Step 6.

    ``corrected_text`` is carried VERBATIM from the model's JSON. It is never
    stripped, reflowed, or repaired here.
    """

    model_config = ConfigDict(extra="ignore")

    sentence_id: str
    claim_ids: List[str] = Field(default_factory=list)
    original_text: str = ""
    corrected_text: str = ""
    status: str = CandidateStatus.REJECTED.value
    rejection_reason: str = ""
    rejection_detail: str = ""
    evidence_ids_used: List[str] = Field(default_factory=list)
    raw_output: str = ""

    @property
    def is_proposed(self) -> bool:
        """Return True if the candidate status is PROPOSED."""
        return self.status == CandidateStatus.PROPOSED.value


class GenerationOutcome(BaseModel):
    """Result of Step 4 for one request: prompts built + candidates produced.

    Carries no reconstructed text and no validation verdict. When
    ``model_status.available`` is False, ``candidates`` is empty by design —
    Step 4 never fabricates a correction it did not obtain from the model.
    """

    model_config = ConfigDict(extra="ignore")

    model_status: ModelStatus = Field(default_factory=ModelStatus)
    prompts: List[PromptBundle] = Field(default_factory=list)
    candidates: List[CorrectionCandidate] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @property
    def proposed(self) -> List[CorrectionCandidate]:
        """Return all candidates with PROPOSED status."""
        return [c for c in self.candidates if c.is_proposed]

    @property
    def has_proposals(self) -> bool:
        """Return True if any candidate has PROPOSED status."""
        return any(c.is_proposed for c in self.candidates)


# ---------------------------------------------------------------------------
# Step 5 — Candidate Validation + Bounded Retry (internal only)
# ---------------------------------------------------------------------------
#
# SCOPE BOUNDARY (restated): nothing below is an inter-agent contract. These
# models are not importable as a handoff schema and introduce NO new
# supervisor-level status. ``orchestration/schemas.py`` remains the sole
# canonical boundary; Step 6 maps ``ValidationOutcome`` onto the existing
# ``CorrectionResult`` builders in ``adapter.py``.


# ValidationFailureCode is DEFINED in `failure_codes.py` and re-exported here.
# The gate layer (`validators.py`) and extraction layer (`text_features.py`) must
# stay importable without pydantic, so the taxonomy lives in a dependency-free
# module. This keeps `from .contracts import ValidationFailureCode` working
# unchanged for every existing caller, with one source of truth.


class ValidationFailure(BaseModel):
    """One concrete, explainable reason a candidate was not accepted.

    ``detail`` carries the offending values (e.g.
    ``{"retained_unsupported": ["48", "48 hours"]}``) so no rejection is opaque.
    """

    model_config = ConfigDict(extra="ignore")

    code: str
    reason: str = ""
    detail: Dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False


class CandidateValidation(BaseModel):
    """Full verdict for ONE candidate against ONE authorized, grounded target.

    ``passed`` is conjunctive: every hard gate passed AND minimality was
    acceptable. A candidate with any hard-gate failure can never be accepted.

    ``is_abstention`` marks a legitimate safe abstention (the model declining for
    want of evidence). Per approved policy an abstention bypasses minimal-edit
    rejection and is NOT unsupported hallucinated content, but it is also NOT a
    factual correction: it never yields ``passed=True`` and its target routes to
    ``TargetDecision.UNRESOLVED``.
    """

    model_config = ConfigDict(extra="ignore")

    sentence_id: str
    passed: bool = False
    failures: List[ValidationFailure] = Field(default_factory=list)

    # Explicit per-dimension signals (recorded even when a gate did not run, in
    # which case the conservative default stands).
    structural_valid: bool = False
    target_authorized: bool = False
    evidence_grounded: bool = False
    contradiction_detected: bool = False
    unsupported_addition_detected: bool = False
    entity_consistent: bool = False
    numbers_consistent: bool = False
    dates_consistent: bool = False
    is_abstention: bool = False

    evidence_alignment_score: float = 0.0
    minimal_edit_similarity: float = 0.0
    entailment_available: bool = False
    evidence_ids_validated: List[str] = Field(default_factory=list)

    @property
    def failure_codes(self) -> List[str]:
        """Return a list of all failure codes."""
        return [f.code for f in self.failures]

    @property
    def failure_reasons(self) -> List[str]:
        """Return a list of all failure reasons."""
        return [f.reason for f in self.failures]

    @property
    def retryable(self) -> bool:
        """True only when at least one failure is retryable and none is terminal."""
        if not self.failures:
            return False
        return all(f.retryable for f in self.failures)


class TargetDecision(str, Enum):
    """Disposition of one authorized target after validation (+ any retries)."""

    ACCEPTED = "accepted"
    RETRY = "retry"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


class CandidateAttempt(BaseModel):
    """One generation attempt for one target, with its verdict."""

    model_config = ConfigDict(extra="ignore")

    attempt_number: int = Field(ge=1)
    candidate: CorrectionCandidate
    validation: CandidateValidation
    decision: str = TargetDecision.REJECTED.value


class TargetResolution(BaseModel):
    """Final disposition of one authorized target across all attempts.

    ``accepted_text`` is non-empty ONLY when ``decision == accepted``. On any
    other decision the original sentence is preserved verbatim and no
    Corrector-authored text exists — Step 5 never fabricates a correction.
    """

    model_config = ConfigDict(extra="ignore")

    sentence_id: str
    claim_ids: List[str] = Field(default_factory=list)
    original_text: str = ""
    accepted_text: str = ""
    decision: str = TargetDecision.UNRESOLVED.value
    attempts: List[CandidateAttempt] = Field(default_factory=list)
    attempt_count: int = 0
    evidence_ids_used: List[str] = Field(default_factory=list)
    final_failure_codes: List[str] = Field(default_factory=list)
    # Set when the target was resolved by a legitimate safe abstention.
    abstained: bool = False

    @property
    def is_accepted(self) -> bool:
        """Return True if the target decision is ACCEPTED."""
        return self.decision == TargetDecision.ACCEPTED.value


class ValidationOutcome(BaseModel):
    """Result of Step 5 for one request. Carries NO reconstructed text.

    Splicing accepted sentences back into the response is Step 6.
    """

    model_config = ConfigDict(extra="ignore")

    model_status: ModelStatus = Field(default_factory=ModelStatus)
    resolutions: List[TargetResolution] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    attempt_count: int = 0

    @property
    def accepted(self) -> List[TargetResolution]:
        """Return all resolutions with ACCEPTED decision."""
        return [r for r in self.resolutions if r.is_accepted]

    @property
    def has_accepted(self) -> bool:
        """Return True if any resolution has ACCEPTED decision."""
        return any(r.is_accepted for r in self.resolutions)

    @property
    def unresolved_targets(self) -> List[TargetResolution]:
        """Return all resolutions that were not accepted."""
        return [r for r in self.resolutions if not r.is_accepted]
