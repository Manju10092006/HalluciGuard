"""
halluciguard_judge / config.py
──────────────────────────────
Configuration for the new standalone detector.
Reads from environment variables with sensible defaults.
"""

import os
from pydantic import BaseModel, Field

try:
    from pydantic_settings import BaseSettings
    _Base = BaseSettings
except ImportError:
    _Base = BaseModel


class JudgeDetectorConfig(_Base):
    """Configuration for the JudgeDetector pipeline."""

    # ── Classifier model ─────────────────────────────────────────────────────
    classifier_model_path: str = Field(
        default=os.environ.get(
            "JUDGE_CLASSIFIER_MODEL",
            "halluciguard_judge/checkpoints/deberta-v3-hallucination"
        ),
        description="Path to fine-tuned DeBERTa-v3 checkpoint or HF model id."
    )
    classifier_max_length: int = Field(
        default=int(os.environ.get("JUDGE_MAX_LENGTH", "512")),
        description="Max token length for classifier input."
    )

    # ── Risk thresholds ───────────────────────────────────────────────────────
    low_risk_threshold: float = Field(
        default=float(os.environ.get("JUDGE_LOW_THRESHOLD", "0.30")),
        description="P(H) <= this -> LOW risk."
    )
    high_risk_threshold: float = Field(
        default=float(os.environ.get("JUDGE_HIGH_THRESHOLD", "0.60")),
        description="P(H) >= this -> HIGH risk."
    )
    # Between low and high -> MEDIUM -> optional LLM judge

    # ── LLM Judge (optional, uncertain cases only) ────────────────────────────
    llm_judge_enabled: bool = Field(
        default=os.environ.get("JUDGE_LLM_ENABLED", "true").lower() == "true",
        description="Whether to call LLM judge for uncertain (MEDIUM) claims."
    )
    openrouter_api_key: str = Field(
        default=os.environ.get("OPENROUTER_API_KEY", ""),
    )
    openrouter_base_url: str = Field(
        default=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    judge_llm_model: str = Field(
        default=os.environ.get("JUDGE_LLM_MODEL", os.environ.get("OPENROUTER_MODEL", "qwen/qwen3-14b")),
        description="LLM model used for the optional judge pass."
    )
    judge_llm_timeout: int = Field(
        default=int(os.environ.get("JUDGE_LLM_TIMEOUT", "25")),
    )

    # ── Claim extractor ───────────────────────────────────────────────────────
    max_claims_per_response: int = Field(
        default=int(os.environ.get("JUDGE_MAX_CLAIMS", "15")),
        description="Hard cap on extracted claims to prevent runaway cost."
    )
    min_claim_length: int = Field(
        default=int(os.environ.get("JUDGE_MIN_CLAIM_LEN", "10")),
        description="Discard claims shorter than this (characters)."
    )

    # ── Routing ───────────────────────────────────────────────────────────────
    always_verify: bool = Field(
        default=os.environ.get("ALWAYS_VERIFY", "false").lower() == "true",
        description="If True, always route to Verifier regardless of risk."
    )


__all__ = ["JudgeDetectorConfig"]
