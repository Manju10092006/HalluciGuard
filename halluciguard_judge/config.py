"""
HalluciGuard - Judge Agent Configuration
Defines model choices, confidence thresholds, risk parameters, and decision weights.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class JudgeConfig:
    # NLI Model Settings
    # Primary model options: 'cross-encoder/nli-deberta-v3-base', 'facebook/bart-large-mnli'
    default_nli_model: str = "cross-encoder/nli-deberta-v3-base"
    fallback_nli_model: str = "facebook/bart-large-mnli"
    use_huggingface: bool = True
    device: str = "cpu"  # 'cuda' or 'cpu'

    # Decision Thresholds
    accept_confidence_threshold: float = 0.82
    correct_contradiction_threshold: float = 0.45
    reject_contradiction_threshold: float = 0.75
    verify_again_confidence_range: tuple = (0.50, 0.81)

    # Weights for Confidence Calibration
    weight_nli_entailment: float = 0.50
    weight_verifier_evidence: float = 0.30
    weight_detector_signal: float = 0.20

    # Severity Thresholds (0.0 to 1.0)
    severity_critical: float = 0.80
    severity_high: float = 0.60
    severity_medium: float = 0.35
    severity_low: float = 0.0

    # Default Domain Contexts
    supported_domains: list = field(default_factory=lambda: [
        "Healthcare", "Finance", "Law", "Cybersecurity", "General"
    ])

DEFAULT_CONFIG = JudgeConfig()
