"""
halluciguard_judge / models.py
──────────────────────────────
Pydantic data models for the new standalone HalluciGuard Detector.

Pipeline contract (read-only by downstream agents):
    LLM Response -> Claim Extraction -> Classifier -> LLM Judge (uncertain) -> DetectorOutput

IMPORTANT: hallucination_probability here is a TRIAGE SIGNAL.
The Verifier is the only agent authorised to deliver a factual verdict.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Risk band for a single claim or the whole response."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RoutingDecision(str, Enum):
    """What should happen after Detector finishes."""
    ACCEPT = "ACCEPT"
    VERIFY = "VERIFY"


class ClaimType(str, Enum):
    """Semantic type of an extracted claim."""
    FACTUAL = "FACTUAL"
    NUMERICAL = "NUMERICAL"
    CAUSAL = "CAUSAL"
    ENTITY = "ENTITY"
    GENERAL = "GENERAL"


class Claim(BaseModel):
    """A single atomic factual statement extracted from the LLM response."""
    claim_id: str = Field(..., description="Unique identifier, e.g. 'C001'.")
    text: str = Field(..., description="The atomic factual claim text.")
    claim_type: ClaimType = Field(default=ClaimType.FACTUAL)
    source_span: Optional[str] = Field(default=None)
    importance: str = Field(default="MEDIUM")


class ClaimDetectionResult(BaseModel):
    """Detector triage result for a single atomic claim.
    
    hallucination_probability is a learned signal - NOT a factual verdict.
    Only the Verifier can determine actual factual correctness.
    """
    claim_id: str
    text: str
    hallucination_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    requires_verification: bool
    classifier_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    llm_judge_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    llm_judge_reasoning: Optional[str] = Field(default=None)
    method_used: str = Field(default="classifier")


class LLMJudgeResult(BaseModel):
    """Structured output from the two-stage LLM-as-a-judge call.
    
    Only triggered when classifier probability falls in the uncertain band (MEDIUM risk).
    """
    claim_text: str
    is_hallucination: bool
    probability: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(default="")
    flagged_as_unsupported: bool = Field(default=False)
    flagged_as_contradiction: bool = Field(default=False)
    raw_cot_output: Optional[str] = Field(default=None)


class DetectorOutput(BaseModel):
    """Final output of the halluciguard_judge Detector.
    
    Pipeline contract handed to the Verifier:
    {
      "claims": [...],
      "overall_risk": "HIGH",
      "routing": "VERIFY",
      "overall_hallucination_probability": 0.94,
      "model_source": "deberta-v3-base-finetuned",
      "status": "completed"
    }
    
    REMINDER: overall_hallucination_probability is a triage signal only.
    """
    claims: List[ClaimDetectionResult] = Field(default_factory=list)
    overall_hallucination_probability: float = Field(..., ge=0.0, le=1.0)
    overall_risk: RiskLevel
    routing: RoutingDecision
    model_source: str = Field(default="deberta-v3-base-finetuned")
    status: str = Field(default="completed")
    degraded: bool = Field(default=False)
    num_claims: int = Field(default=0)
    num_high_risk: int = Field(default=0)
    num_medium_risk: int = Field(default=0)
    num_low_risk: int = Field(default=0)
    llm_judge_invoked: bool = Field(default=False)


class DetectorInput(BaseModel):
    """Input to the new standalone JudgeDetector."""
    user_query: str = Field(..., min_length=1)
    llm_response: str = Field(..., min_length=1)
    context: Optional[str] = Field(default=None)


__all__ = [
    "RiskLevel", "RoutingDecision", "ClaimType",
    "Claim", "ClaimDetectionResult", "LLMJudgeResult",
    "DetectorOutput", "DetectorInput",
]
