from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from .utils import init_dll_paths
init_dll_paths()

from .signals.entropy import EntropyMetrics
from .signals.semantic_similarity import SemanticSimilarityMetrics
from .signals.self_consistency import SelfConsistencyMetrics


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


class TokenProbabilityMetrics(BaseModel):
    """Structured metrics returned by the Token Probability computation module."""
    avg_logprob: float = Field(
        ...,
        description="Average token log probability across response sequence."
    )
    min_logprob: float = Field(
        ...,
        description="Minimum (worst-case dip) token log probability in response sequence."
    )
    low_confidence_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of generated tokens whose log probability fell below confidence threshold."
    )


class SignalMetricsDetail(BaseModel):
    """Detailed per-signal metrics breakdown collected during hallucination detection."""
    token_probability: Optional[TokenProbabilityMetrics] = Field(
        default=None, description="Metrics from Token Probability signal."
    )
    entropy: Optional[EntropyMetrics] = Field(
        default=None, description="Metrics from Predictive Entropy signal."
    )
    semantic_similarity: Optional[SemanticSimilarityMetrics] = Field(
        default=None, description="Metrics from Semantic Similarity signal."
    )
    self_consistency: Optional[SelfConsistencyMetrics] = Field(
        default=None, description="Metrics from Self-Consistency signal."
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
    metrics: Optional[SignalMetricsDetail] = Field(
        default=None,
        description="Detailed breakdown of metrics collected from all active signal modules."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "confidence_score": 0.95,
                "hallucination_probability": 0.05,
                "risk_level": "LOW",
                "next_action": "Accept"
            }
        }


__all__ = [
    "RiskLevel",
    "NextAction",
    "DetectionInput",
    "TokenProbabilityMetrics",
    "EntropyMetrics",
    "SemanticSimilarityMetrics",
    "SelfConsistencyMetrics",
    "SignalMetricsDetail",
    "DetectionResult",
]
