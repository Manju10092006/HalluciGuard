"""
Detector Agent Configuration — HaluEval Classifier Edition.

Configuration for the HaluEval-trained hallucination detector,
replacing the old heuristic signal weights.
"""

import os

try:
    from pydantic_settings import BaseSettings
    ConfigBase = BaseSettings
except ImportError:
    from pydantic import BaseModel
    ConfigBase = BaseModel

from pydantic import Field


class DetectorConfig(ConfigBase):
    """Configuration settings for the HaluEval-based Detector Agent."""

    # --- Risk Threshold Configuration ---
    low_risk_threshold: float = Field(
        default=0.30,
        description="Hallucination probability at or below which risk is LOW."
    )
    high_risk_threshold: float = Field(
        default=0.50,
        description="Hallucination probability at or above which risk is HIGH."
    )

    # --- HaluEval Model Configuration ---
    halueval_model_path: str = Field(
        default=os.environ.get("HALUEVAL_MODEL_PATH", "Manjunath2000006/halluciguard-detector"),
        description="Path to the fine-tuned HaluEval model directory or HF repo."
    )
    halueval_max_length: int = Field(
        default=int(os.environ.get("HALUEVAL_MAX_LENGTH", "384")),
        description="Maximum token sequence length for the HaluEval classifier."
    )

    # --- Legacy fields kept for backward compatibility ---
    model_name: str = Field(
        default="distilbert-base-uncased",
        description="Base model name (for reference only)."
    )

