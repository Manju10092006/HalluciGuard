"""
Detector Agent Models — Pydantic I/O Schemas.

Defines the input and output data structures for the HalluciGuard Detector Agent.
The external interface (DetectionInput, DetectionResult, RiskLevel, NextAction)
is preserved exactly for backward compatibility with Verifier/Judge/Corrector agents.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Represents the estimated risk level of LLM hallucination."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class NextAction(str, Enum):
    """Represents the recommended action for downstream agents in the HalluciGuard pipeline."""
    ACCEPT = "Accept"
    VERIFY = "Verify"


class DetectionInput(BaseModel):
    """Input payload required by the Detector Agent."""
    user_query: str = Field(
        ...,
        description="The original user query or prompt submitted to the LLM.",
        min_length=1,
        examples=["What is the capital of France?"]
    )
    llm_response: str = Field(
        ...,
        description="The generated response from the LLM to be checked for hallucinations.",
        min_length=1,
        examples=["The capital of France is Paris."]
    )
    context: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional evidence/provenance corpus (document snippets, retrieved "
            "passages). When provided, the detector performs claim-level hybrid "
            "evidence verification (retrieval -> rerank -> NLI) and labels each "
            "claim VERIFIED / CONTRADICTED / INSUFFICIENT. When absent, claims "
            "are left UNVERIFIED for the Verifier Agent (default legacy triage)."
        ),
        examples=[["Paris is the capital city of France."]],
    )


class ClaimRisk(BaseModel):
    """Detector triage result for one atomic factual claim.

    The probability is a routing signal, not a factual verdict.  Only the
    evidence-backed Verifier is allowed to label a claim verified or
    contradicted.
    """

    claim_id: str
    text: str
    hallucination_probability: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    requires_verification: bool
    classifier_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    token_surprisal_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # --- Claim-level hybrid evidence verification (additive; safe defaults) ---
    claim_type: Optional[str] = Field(
        default=None,
        description=(
            "Deterministic claim type used to decide evidence needs: "
            "FACTUAL | NUMERICAL | TEMPORAL | ENTITY | RELATIONAL | "
            "COMPARATIVE | OPINION."
        ),
    )
    verification_status: Optional[str] = Field(
        default=None,
        description=(
            "Evidence-based label for this claim where verified against "
            "documents: 'VERIFIED' | 'CONTRADICTED' | 'INSUFFICIENT' | "
            "'UNVERIFIED' | 'SKIPPED' (opinion) | 'NOT_EVALUATED'. Only the "
            "Verifier Agent is allowed to finalize a truth verdict."
        ),
    )
    supported_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    contradicted_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    unknown_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evidence_snippets: List[str] = Field(default_factory=list)
    retrieval_method: Optional[str] = Field(default=None)
    nli_degraded: bool = Field(default=False)


class DetectionResult(BaseModel):
    """Structured output returned by the Detector Agent.
    
    This schema is the external contract between the Detector and downstream
    agents (Verifier, Judge, Corrector). It must be preserved exactly.
    """
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated confidence in the response reliability (0.0 to 1.0).",
        examples=[0.95]
    )
    hallucination_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability that the response contains hallucinations (0.0 to 1.0).",
        examples=[0.05]
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Categorized risk level based on hallucination probability (LOW, MEDIUM, HIGH).",
        examples=[RiskLevel.LOW]
    )
    next_action: NextAction = Field(
        ...,
        description="Recommended action for the HalluciGuard pipeline (Accept or Verify).",
        examples=[NextAction.ACCEPT]
    )
    model_source: str = Field(
        default="halueval-distilbert",
        description="Identifier of the model that produced this result."
    )
    status: str = Field(
        default="completed",
        description=(
            "Execution status aligned with orchestration ExecutionStatus values "
            "('completed' | 'degraded' | 'failed'). Orchestration reads this to "
            "force verification when the detector could not run real inference; "
            "a missing/empty status previously left that safety gate dead."
        ),
    )

    # --- §6 Detector execution diagnostics (additive; safe defaults) ---
    # These make it impossible for a failed detector load to masquerade as real
    # ML inference. They are informational only and do not change routing in the
    # resilient production path; certification mode reads them to fail-closed.
    detector_model_loaded: bool = Field(
        default=False,
        description="True iff the HaluEval classifier weights actually loaded into memory.",
    )
    detector_inference_executed: bool = Field(
        default=False,
        description="True iff a real forward pass produced this probability (not the heuristic baseline).",
    )
    detector_degraded: bool = Field(
        default=False,
        description="True when the detector fell back to the hardcoded baseline instead of running the model.",
    )
    detector_model_source: str = Field(
        default="",
        description="Concrete provenance: resolved model dir / HF id when loaded, or 'baseline-heuristic' when degraded.",
    )
    atomic_claims: List[str] = Field(
        default_factory=list,
        description="Complete atomic factual claims extracted from the response.",
    )
    per_claim_results: List[ClaimRisk] = Field(
        default_factory=list,
        description="Per-claim hallucination-risk triage used by orchestration.",
    )
    evaluator_inference_executed: bool = Field(default=False)
    evaluator_model_source: str = Field(default="")

    # --- Claim-level evidence verification summary (additive; safe defaults) ---
    claim_count: int = Field(default=0)
    supported_count: int = Field(default=0)
    contradicted_count: int = Field(default=0)
    unknown_count: int = Field(default=0)
    unverified_count: int = Field(default=0)
    opinion_count: int = Field(default=0)
    requires_verification: bool = Field(
        default=False,
        description=(
            "Answer-level flag: True when at least one factual claim still needs "
            "evidence verification (equivalent to next_action == Verify)."
        ),
    )
    evidence_available: bool = Field(
        default=False,
        description="True when the detector was given evidence documents to verify claims against.",
    )
    verification_risk: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Evidence-based verification risk (probability the response contains "
            "a hallucinated claim). Non-calibrated signal, not a truth verdict."
        ),
    )
    nli_engine_loaded: bool = Field(default=False)
    nli_inference_executed: bool = Field(default=False)
    nli_degraded: bool = Field(default=False)

    class Config:
        json_schema_extra = {
            "example": {
                "confidence_score": 0.95,
                "hallucination_probability": 0.05,
                "risk_level": "LOW",
                "next_action": "Accept",
                "model_source": "halueval-distilbert",
                "detector_model_loaded": True,
                "detector_inference_executed": True,
                "detector_degraded": False,
                "detector_model_source": "artifacts/halueval-detector-final"
            }
        }


__all__ = [
    "RiskLevel",
    "NextAction",
    "DetectionInput",
    "ClaimRisk",
    "DetectionResult",
]
