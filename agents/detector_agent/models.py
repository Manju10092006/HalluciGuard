"""
Detector Agent Models — Pydantic I/O Schemas.

Defines the input and output data structures for the HalluciGuard Detector Agent.
The external interface (DetectionInput, DetectionResult, RiskLevel, NextAction)
is preserved exactly for backward compatibility with Verifier/Judge/Corrector agents.
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

    class Config:
        json_schema_extra = {
            "example": {
                "confidence_score": 0.95,
                "hallucination_probability": 0.05,
                "risk_level": "LOW",
                "next_action": "Accept",
                "model_source": "halueval-distilbert"
            }
        }


__all__ = [
    "RiskLevel",
    "NextAction",
    "DetectionInput",
    "DetectionResult",
]
