import math
from pydantic import BaseModel, Field, model_validator

try:
    from pydantic_settings import BaseSettings
    ConfigBase = BaseSettings
except ImportError:
    ConfigBase = BaseModel


class SignalWeights(BaseModel):
    """Legacy signal-weight model retained for config compatibility."""
    token_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    entropy: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    self_consistency: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights_sum(self):
        total = sum(self.model_dump().values())
        if not math.isclose(total, 0.0, abs_tol=1e-4):
            raise ValueError("Legacy detector signal weights must be zero; HaluEval is the sole detector.")
        return self


class DetectorConfig(ConfigBase):
    """Configuration for the HaluEval-based Detector Agent."""
    low_risk_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Hallucination probability at or below which the response is LOW risk."
    )
    high_risk_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Hallucination probability at or above which the Verifier Agent is invoked."
    )
    halueval_model_path: str = Field(
        default="./artifacts/halueval-detector",
        description="Path to the fine-tuned HaluEval classifier. Override with HALUEVAL_MODEL_PATH."
    )
    halueval_base_model: str = Field(
        default="distilbert-base-uncased",
        description="Base encoder used when training the HaluEval classifier."
    )
    halueval_dataset: str = Field(
        default="pminervini/HaluEval",
        description="Hugging Face HaluEval dataset used for detector training."
    )
    default_confidence_score: float = Field(default=0.95, ge=0.0, le=1.0)
    default_hallucination_prob: float = Field(default=0.05, ge=0.0, le=1.0)
    model_name: str = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        description="Legacy field retained for compatibility; not used by the HaluEval detector."
    )
    low_confidence_logprob_threshold: float = Field(default=-2.0)
    signal_weights: SignalWeights = Field(default_factory=SignalWeights)
    num_samples: int = Field(default=5, ge=2, le=20)
    temperature: float = Field(default=0.7, gt=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=100, ge=1, le=512)
