import math
from pydantic import BaseModel, Field, model_validator

try:
    from pydantic_settings import BaseSettings
    ConfigBase = BaseSettings
except ImportError:
    ConfigBase = BaseModel


class SignalWeights(BaseModel):
    """Configurable weights assigned to each detection signal for score aggregation."""
    token_probability: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Weight assigned to Token Probability signal."
    )
    entropy: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight assigned to Predictive Entropy signal."
    )
    semantic_similarity: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight assigned to Semantic Similarity signal."
    )
    self_consistency: float = Field(
        default=0.00,
        ge=0.0,
        le=1.0,
        description="Weight assigned to Self-Consistency signal."
    )

    @model_validator(mode="after")
    def validate_weights_sum(self):
        """Validate that all weights are in [0, 1] and sum to exactly 1.0 (within 1e-4 tolerance)."""
        weights_dict = self.model_dump()
        for name, val in weights_dict.items():
            if val < 0.0 or val > 1.0:
                raise ValueError(f"Signal weight '{name}' must be between 0.0 and 1.0, got {val}")
        
        total_sum = sum(weights_dict.values())
        if not math.isclose(total_sum, 1.0, abs_tol=1e-4):
            raise ValueError(
                f"Signal weights must sum to 1.0. Current sum is {total_sum:.4f} for configuration: {weights_dict}"
            )
        return self


class DetectorConfig(ConfigBase):
    """Configuration settings for the Detector Agent."""
    low_risk_threshold: float = Field(
        default=0.40,
        description="Calibrated hallucination probability threshold at or below which risk is LOW."
    )
    high_risk_threshold: float = Field(
        default=0.55,
        description="Calibrated hallucination probability threshold at or above which risk is HIGH."
    )
    default_confidence_score: float = Field(
        default=0.95,
        description="Baseline dummy confidence score for Phase 1 fallback."
    )
    default_hallucination_prob: float = Field(
        default=0.05,
        description="Baseline dummy hallucination probability for Phase 1 fallback."
    )
    model_name: str = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        description="HuggingFace causal language model for computing token log probabilities and sampling."
    )
    low_confidence_logprob_threshold: float = Field(
        default=-2.0,
        description="Log probability threshold below which a token is considered low-confidence."
    )
    signal_weights: SignalWeights = Field(
        default_factory=SignalWeights,
        description="Signal weighting configuration for score aggregation."
    )
    # Self-Consistency Sampling Parameters
    num_samples: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of candidate responses to sample for self-consistency evaluation."
    )
    temperature: float = Field(
        default=0.7,
        gt=0.0,
        le=2.0,
        description="Sampling temperature for stochastic decoding."
    )
    top_p: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Top-p nucleus sampling threshold."
    )
    max_new_tokens: int = Field(
        default=100,
        ge=1,
        le=512,
        description="Maximum new tokens to generate per candidate response."
    )
