"""
Canonical Supervisor Contracts for HalluciGuard Inter-Agent Communication.

These Pydantic models define the single canonical source of truth for handoffs
between the Supervisor and all agent components:
  1. Base LLM -> Detector: DetectorResult
  2. Detector -> Verifier: Evidence, ClaimReport, VerifierResult
  3. Verifier -> Judge: JudgeResult, CorrectionRequest
  4. Judge -> Corrector: CorrectionRequest, CorrectionResult
  5. Corrector -> Re-verifier: ReverificationResult
  6. Re-verifier / Final -> Memory: MemoryResult
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ContractViolation(ValidationError):
    """Raised when an agent produces a value that violates the canonical contract."""
    pass


class ErrorType(str, Enum):
    """Structured error taxonomy for all agent failures."""
    TIMEOUT = "timeout"
    NETWORK = "network"
    CONTRACT = "contract"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    INTERNAL = "internal"


# ---------------------------------------------------------------------------
# Enums / Literal Type Definitions (snake_case and consistent values)
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    """Risk level classification for detector output."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class NextAction(str, Enum):
    """Next action recommendation from detector."""
    ACCEPT = "Accept"
    VERIFY = "Verify"


class EntailmentLabel(str, Enum):
    """Entailment classification labels for NLI scoring."""
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class VerdictLabel(str, Enum):
    """Verdict labels for claim verification status."""
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNVERIFIED = "unverified"
    CONFLICTED = "conflicted"


class VerificationStatus(str, Enum):
    """Answer-level verification status reflecting aggregated claim verdicts.

    Precedence rules (highest to lowest):
        1. Any CONTRADICTED -> CONTRADICTED
        2. Any CONFLICTED   -> CONFLICTED
        3. All VERIFIED     -> ALL_VERIFIED
        4. Mix of verified/unverified/unknown -> PARTIALLY_VERIFIED
        5. Everything else  -> UNVERIFIED
    """
    ALL_VERIFIED = "all_verified"
    PARTIALLY_VERIFIED = "partially_verified"
    CONTRADICTED = "contradicted"
    CONFLICTED = "conflicted"
    UNVERIFIED = "unverified"


class JudgeDecision(str, Enum):
    """Judge decision outcomes."""
    ACCEPT = "ACCEPT"
    CORRECT = "CORRECT"
    REJECT = "REJECT"
    VERIFY_AGAIN = "VERIFY_AGAIN"
    ABSTAIN = "ABSTAIN"


class SeverityLevel(str, Enum):
    """Severity levels for issues and risks."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ExecutionStatus(str, Enum):
    """Execution status for agent operations."""
    COMPLETED = "completed"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    DEGRADED = "degraded"
    FALLBACK = "fallback"
    TERMINATED_UNRESOLVED = "terminated_unresolved"


class MemoryStatus(str, Enum):
    """Status for memory storage operations."""
    STORED = "stored"
    SKIPPED = "skipped"
    FAILED = "failed"
    DUPLICATE = "duplicate"


class ValidationStatus(str, Enum):
    """Validation status for corrector output."""
    VALID = "valid"
    INVALID = "invalid"
    UNVALIDATED = "unvalidated"
    WARNING = "warning"


class PipelineStatus(str, Enum):
    """Clear pipeline-level status to prevent contradictory states."""
    RUNNING = "running"
    VERIFIED = "verified"
    NEEDS_CORRECTION = "needs_correction"
    REJECTED = "rejected"
    HUMAN_REVIEW = "human_review"
    FAILED = "failed"


class ErrorType(str, Enum):
    """Structured error taxonomy for all agent failures."""
    TIMEOUT = "timeout"
    NETWORK = "network"
    CONTRACT = "contract"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    INTERNAL = "internal"


class ContractViolation(Exception):
    """Raised when an agent produces data that violates a canonical contract.

    Attributes:
        agent: The name of the agent that produced the invalid data.
        field: The field name that violated the contract.
        received: The value that was received.
        expected: Description of valid values.
        ctx: Additional context dict with agent, field, received, expected_values.
    """

    def __init__(
        self,
        message: str,
        *,
        agent: str = "unknown",
        field: str = "",
        received: Any = None,
        expected: str = "",
        ctx: Optional[dict[str, Any]] = None,
    ):
        self.agent = agent
        self.field = field
        self.received = received
        self.expected = expected
        self.ctx = ctx or {}
        super().__init__(message)


class ValidationError(Exception):
    """Raised when input data fails validation."""
    pass


class VerificationStatus(str, Enum):
    """Answer-level verification status representing the aggregation
    of multiple claim-level verdicts."""
    ALL_VERIFIED = "all_verified"
    PARTIALLY_VERIFIED = "partially_verified"
    CONTRADICTED = "contradicted"
    CONFLICTED = "conflicted"
    UNVERIFIED = "unverified"


class PipelineStatus(str, Enum):
    """Clear pipeline-level status to prevent contradictory states."""
    RUNNING = "running"
    VERIFIED = "verified"
    NEEDS_CORRECTION = "needs_correction"
    REJECTED = "rejected"
    HUMAN_REVIEW = "human_review"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# 1. Detector Contract
# ---------------------------------------------------------------------------

class DetectorResult(BaseModel):
    """Canonical output from the Detector Agent."""
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    hallucination_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability that the response contains hallucinations (0.0 to 1.0).",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated confidence in the reliability assessment (0.0 to 1.0).",
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Categorized risk level (LOW, MEDIUM, HIGH).",
    )
    next_action: NextAction = Field(
        ...,
        description="Recommended action for downstream orchestration (Accept or Verify).",
    )
    model_source: str = Field(
        default="halueval-distilbert",
        description="Identifier of the detection model or heuristic used.",
    )
    status: ExecutionStatus = Field(
        default=ExecutionStatus.COMPLETED,
        description="Status of the detector execution.",
    )


# ---------------------------------------------------------------------------
# 2. Evidence Contract
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    """Canonical representation of an individual verified passage or citation."""
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    evidence_id: str = Field(
        ...,
        description="Unique identifier for the evidence passage.",
    )
    title: str = Field(
        default="",
        description="Title of the source document or article.",
    )
    source: str = Field(
        ...,
        description="Name or provider of the source (e.g. wikipedia, pubmed, sec_edgar).",
    )
    url: Optional[str] = Field(
        default=None,
        description="Direct URL to the source document if available.",
    )
    snippet: str = Field(
        ...,
        description="Text snippet extracted from the authoritative passage.",
    )
    entailment_label: EntailmentLabel = Field(
        ...,
        description="NLI relationship to the claim (entailment, contradiction, neutral).",
    )
    entailment_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="NLI classification confidence score (0.0 to 1.0).",
    )
    credibility_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Calibrated credibility and authority score of the source (0.0 to 1.0).",
    )


# ---------------------------------------------------------------------------
# 3. Claim Report Contract
# ---------------------------------------------------------------------------

class ClaimReport(BaseModel):
    """Canonical verification assessment for an individual atomic claim."""
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    claim_id: str = Field(
        ...,
        description="Unique identifier for the claim (e.g. c1, c2).",
    )
    claim_text: str = Field(
        ...,
        description="Text of the atomic claim being evaluated.",
    )
    verdict: VerdictLabel = Field(
        ...,
        description="Verification outcome (verified, contradicted, unverified, conflicted).",
    )
    support_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregated evidence support score (0.0 to 1.0).",
    )
    contradiction_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregated evidence contradiction score (0.0 to 1.0).",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Calibrated confidence score for the claim verdict (0.0 to 1.0).",
    )
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="Authoritative evidence passages matched to this claim.",
    )


# ---------------------------------------------------------------------------
# 4. Verifier Contract
# ---------------------------------------------------------------------------

class VerifierResult(BaseModel):
    """Canonical structured output returned by the Verifier Agent."""
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    query_id: str = Field(
        ...,
        description="Correlation ID for the verification request.",
    )
    domain: str = Field(
        default="general",
        description="Domain of verification (e.g. general, healthcare, finance, cybersecurity).",
    )
    claim_reports: List[ClaimReport] = Field(
        default_factory=list,
        description="Individual claim-level verification reports.",
    )
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="All aggregated evidence items retrieved across claims.",
    )
    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall trust and confidence score across all evaluated claims.",
    )
    retrieved_sources_count: int = Field(
        default=0,
        ge=0,
        description="Total raw passages retrieved from external domain adapters.",
    )
    verified_sources_count: int = Field(
        default=0,
        ge=0,
        description="Total passages that passed relevance, reranking, and NLI gates.",
    )
    status: ExecutionStatus = Field(
        default=ExecutionStatus.COMPLETED,
        description="Overall execution status of the verifier pipeline.",
    )


# ---------------------------------------------------------------------------
# 5. Correction Request Contract (Judge -> Corrector handoff)
# ---------------------------------------------------------------------------

class CorrectionRequest(BaseModel):
    """Canonical payload delivered to the Corrector Agent when correction is required."""
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    execution_id: str = Field(
        ...,
        description="Unique execution run identifier.",
    )
    user_query: str = Field(
        ...,
        description="The original user query or prompt.",
    )
    original_response: str = Field(
        ...,
        description="The draft response containing hallucinations to repair.",
    )
    claims_to_correct: List[ClaimReport] = Field(
        default_factory=list,
        description="Claims identified as hallucinated or contradicted needing evidence-grounded repair.",
    )
    claims_to_preserve: List[ClaimReport] = Field(
        default_factory=list,
        description="Verified factual claims that must be preserved verbatim.",
    )
    trusted_evidence: List[Evidence] = Field(
        default_factory=list,
        description="Authoritative supporting evidence passages to ground the corrected text.",
    )
    contradictory_evidence: List[Evidence] = Field(
        default_factory=list,
        description="Evidence refuting the hallucinated claims for context.",
    )
    correction_instructions: str = Field(
        default="",
        description="Specific guidance and policy constraints emitted by the Judge Agent.",
    )


# ---------------------------------------------------------------------------
# 6. Judge Contract
# ---------------------------------------------------------------------------

class JudgeResult(BaseModel):
    """Canonical decision output emitted by the Judge Agent."""
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    decision: JudgeDecision = Field(
        ...,
        description="Final arbitration decision (ACCEPT, CORRECT, REJECT, VERIFY_AGAIN, ABSTAIN).",
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.LOW,
        description="Assessed severity of identified contradictions or risks.",
    )
    reason: str = Field(
        default="",
        description="Concise summary of the decision rationale.",
    )
    explanation: str = Field(
        default="",
        description="Detailed multi-signal justification for the arbitration.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Calibrated Bayesian confidence in the decision (0.0 to 1.0).",
    )
    correction_request: Optional[CorrectionRequest] = Field(
        default=None,
        description="Targeted correction payload if decision == CORRECT, else None.",
    )
    decision_basis: str = Field(
        default="",
        description=(
            "Machine-readable reason code identifying the precedence rule that "
            "produced this decision (e.g. CONTRADICTION_PRESENT, "
            "ALL_CLAIMS_VERIFIED, PERIPHERAL_UNVERIFIED_TOLERATED, "
            "CORE_UNVERIFIED_RETRY, CORE_UNVERIFIED_ABSTAIN). Stable for metrics "
            "and audit; never a factual claim about the response."
        ),
    )
    decision_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured, machine-readable counts backing the decision (spec §3/§25): "
            "verified_claims, contradicted_claims, unverified_claims, "
            "material_contradictions, material_unknowns, and detector_role "
            "('triage_only' — the detector is never factual evidence). Purely "
            "observational; the decision itself is produced by the precedence tree."
        ),
    )
    status: ExecutionStatus = Field(
        default=ExecutionStatus.COMPLETED,
        description="Status of the judge agent evaluation.",
    )


# ---------------------------------------------------------------------------
# 7. Correction Result Contract (Corrector output)
# ---------------------------------------------------------------------------

class CorrectionResult(BaseModel):
    """Canonical output returned by the Corrector Agent."""
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    original_text: str = Field(
        ...,
        description="Original draft text before correction.",
    )
    corrected_text: str = Field(
        ...,
        description="Refined response with hallucinations repaired and verified facts preserved.",
    )
    changed_claims: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured diff log of actions taken per claim.",
    )
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.UNVALIDATED,
        description="Structural validation status of the corrected output.",
    )
    attempt_count: int = Field(
        default=1,
        ge=1,
        description="Number of refinement attempts performed.",
    )
    status: ExecutionStatus = Field(
        default=ExecutionStatus.COMPLETED,
        description="Overall execution status of the corrector.",
    )
    failure_category: Optional[str] = Field(
        default=None,
        description=(
            "When status=FAILED, the diagnostic class of the failure: "
            "LLM_PROVIDER_FAILURE, NO_LOCATABLE_CLAIM, MODEL_ECHO, NO_OP_MATCH, "
            "or OTHER. None on success."
        ),
    )
    provider_used: Optional[str] = Field(
        default=None,
        description="Hosted LLM provider that served the correction, if any.",
    )


# ---------------------------------------------------------------------------
# 8. Reverification Result Contract
# ---------------------------------------------------------------------------

class ReverificationResult(BaseModel):
    """Canonical outcome of running the Verifier on corrected candidate text."""
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    passed: bool = Field(
        ...,
        description="True if all claims are verified without remaining contradictions.",
    )
    verifier_result: VerifierResult = Field(
        ...,
        description="Complete verifier result for the re-verification pass.",
    )
    remaining_contradictions: int = Field(
        default=0,
        ge=0,
        description="Count of unresolved or newly introduced contradictions.",
    )
    status: ExecutionStatus = Field(
        default=ExecutionStatus.COMPLETED,
        description="Status of the reverification process.",
    )
    failure_category: Optional[str] = Field(
        default=None,
        description=(
            "When passed=False, the machine-readable class of the re-verification "
            "failure (spec §25): REMAINING_CONTRADICTION, DEGRADED_REVERIFICATION "
            "(retrieval could not re-ground the corrected answer), "
            "CLAIM_DECOMPOSITION_FAILED, or VERIFIER_FAILURE. None when passed."
        ),
    )


# ---------------------------------------------------------------------------
# 9. Memory Result Contract
# ---------------------------------------------------------------------------

class MemoryResult(BaseModel):
    """Canonical persistence outcome returned by the Memory Agent."""
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    status: MemoryStatus = Field(
        ...,
        description="Outcome of memory persistence (stored, skipped, failed, duplicate).",
    )
    stored_count: int = Field(
        default=0,
        ge=0,
        description="Number of verified facts committed to knowledge graph and vector memory.",
    )
    duplicate_count: int = Field(
        default=0,
        ge=0,
        description="Number of duplicate facts found during persistence.",
    )
    failed_count: int = Field(
        default=0,
        ge=0,
        description="Number of facts that failed persistence.",
    )
    fact_ids: List[str] = Field(
        default_factory=list,
        description="Unique IDs of facts stored or referenced in memory.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Explanation when storage was skipped, duplicate, or failed.",
    )


__all__ = [
    "RiskLevel",
    "NextAction",
    "EntailmentLabel",
    "VerdictLabel",
    "VerificationStatus",
    "JudgeDecision",
    "SeverityLevel",
    "ExecutionStatus",
    "MemoryStatus",
    "ValidationStatus",
    "PipelineStatus",
    "ErrorType",
    "ContractViolation",
    "DetectorResult",
    "Evidence",
    "ClaimReport",
    "VerifierResult",
    "CorrectionRequest",
    "JudgeResult",
    "CorrectionResult",
    "ReverificationResult",
    "MemoryResult",
]
