"""
Detector Agent Models — Pydantic I/O Schemas.

Defines the input and output data structures for the HalluciGuard Detector Agent.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Represents the estimated risk level of LLM hallucination."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class NextAction(str, Enum):
    """Represents the recommended action in the HalluciGuard pipeline."""
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
    context: Optional[str] = Field(
        default=None,
        description="Optional grounding context for the detection.",
        examples=["France is a country in Western Europe whose capital is Paris."]
    )


class DetectionResult(BaseModel):
    """Structured output returned by the Detector Agent."""
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
        default="Manjunath2000006/halluciguard-detector",
        description="Identifier of the model that produced this result."
    )

    # Execution diagnostics
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

    class Config:
        json_schema_extra = {
            "example": {
                "confidence_score": 0.9719,
                "hallucination_probability": 0.0281,
                "risk_level": "LOW",
                "next_action": "Accept",
                "model_source": "Manjunath2000006/halluciguard-detector",
                "detector_model_loaded": True,
                "detector_inference_executed": True,
                "detector_degraded": False,
                "detector_model_source": "Manjunath2000006/halluciguard-detector"
            }
        }


__all__ = [
    "RiskLevel",
    "NextAction",
    "DetectionInput",
    "DetectionResult",
]
