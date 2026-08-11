from __future__ import annotations

import os
from typing import Optional

from .config import DetectorConfig
from .halueval_detector import HaluEvalDetector
from .models import DetectionResult, NextAction, RiskLevel, SignalMetricsDetail


class DetectorAgent:
    """HaluEval-based first-stage hallucination detector.

    The Detector no longer synthesizes token probability, entropy, semantic
    similarity, or self-consistency heuristics. It uses a binary classifier
    fine-tuned on HaluEval to estimate hallucination probability. Only HIGH
    risk responses are routed to the Verifier Agent.
    """

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        model_path: Optional[str] = None,
    ) -> None:
        self.config = config or DetectorConfig()
        resolved_path = model_path or os.getenv(
            "HALUEVAL_MODEL_PATH", self.config.halueval_model_path
        )
        self.model_path = resolved_path
        self.classifier = HaluEvalDetector(model_path=resolved_path)

    def detect(self, user_query: str, llm_response: str) -> DetectionResult:
        """Score the generated response and decide whether verification is needed."""
        prediction = self.classifier.predict(
            user_query=user_query,
            llm_response=llm_response,
        )
        hallucination_probability = float(prediction["hallucination_probability"])
        confidence_score = float(prediction["confidence_score"])

        risk_level = self._determine_risk_level(hallucination_probability)
        # Architecture requirement: only HIGH hallucination risk enters verification.
        next_action = (
            NextAction.VERIFY
            if risk_level == RiskLevel.HIGH
            else NextAction.ACCEPT
        )

        return DetectionResult(
            confidence_score=confidence_score,
            hallucination_probability=hallucination_probability,
            risk_level=risk_level,
            next_action=next_action,
            metrics=SignalMetricsDetail(),
        )

    def _determine_risk_level(self, hallucination_probability: float) -> RiskLevel:
        if hallucination_probability >= self.config.high_risk_threshold:
            return RiskLevel.HIGH
        if hallucination_probability > self.config.low_risk_threshold:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
