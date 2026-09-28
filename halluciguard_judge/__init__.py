"""
halluciguard_judge
──────────────────
New standalone hallucination detector for HalluciGuard.

Pipeline:
    LLM Response -> Claim Extraction -> DeBERTa Classifier
    -> (uncertain) LLM Judge -> DetectorOutput -> Verifier

This module is COMPLETELY INDEPENDENT of agents/detector_agent/.
It will replace the old detector only after passing benchmark evaluation.

Quick start:
    from halluciguard_judge import JudgeDetector

    detector = JudgeDetector()
    result = detector.detect(
        user_query="Who created Java?",
        llm_response="Java was created by Dennis Ritchie in 1972."
    )
    print(result.overall_risk)     # HIGH
    print(result.routing)          # VERIFY
    print(result.claims[0].text)   # "Java was created by Dennis Ritchie in 1972."
    print(result.claims[0].hallucination_probability)  # 0.94
"""

from .detector import JudgeDetector
from .models import (
    Claim,
    ClaimDetectionResult,
    ClaimType,
    DetectorInput,
    DetectorOutput,
    LLMJudgeResult,
    RiskLevel,
    RoutingDecision,
)
from .config import JudgeDetectorConfig
from .claim_extractor import ClaimExtractor

__version__ = "0.1.0"
__all__ = [
    "JudgeDetector",
    "JudgeDetectorConfig",
    "ClaimExtractor",
    "Claim",
    "ClaimDetectionResult",
    "ClaimType",
    "DetectorInput",
    "DetectorOutput",
    "LLMJudgeResult",
    "RiskLevel",
    "RoutingDecision",
]
