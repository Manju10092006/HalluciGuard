"""
HalluciGuard - Judge Agent Confidence Calibrator & Risk Engine
Aggregates detector confidence, verifier evidence strength, and NLI predictions to output
a calibrated confidence score and risk assessment.
"""

from typing import Dict, List, Any
import math
from config import JudgeConfig, DEFAULT_CONFIG

class ConfidenceCalibrator:
    def __init__(self, config: JudgeConfig = DEFAULT_CONFIG):
        self.config = config

    def calibrate(
        self,
        detector_prob: float,
        detector_confidence: float,
        evaluated_pairs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calibrates overall decision confidence score and calculates risk metrics.
        """
        if not evaluated_pairs:
            # No evidence provided by Verifier
            calibrated_confidence = max(0.0, round(detector_confidence * (1.0 - detector_prob) * 0.5, 4))
            return {
                "calibrated_confidence": calibrated_confidence,
                "overall_entailment": 0.0,
                "overall_contradiction": 0.5,
                "evidence_strength": 0.0,
                "risk_score": round(1.0 - calibrated_confidence, 4),
                "severity": self._calculate_severity(1.0 - calibrated_confidence, 0.5)
            }

        total_weight = 0.0
        weighted_entailment = 0.0
        weighted_contradiction = 0.0
        weighted_evidence_conf = 0.0

        for pair in evaluated_pairs:
            evidence_conf = pair.get("evidence_confidence", 0.70)
            nli = pair.get("nli_scores", {"entailment": 0.33, "neutral": 0.33, "contradiction": 0.34})
            
            # Evidence weight scales with verifier confidence and evidence rank
            rank = pair.get("rank", 1)
            rank_discount = 1.0 / (1.0 + 0.2 * (rank - 1))
            weight = evidence_conf * rank_discount

            weighted_entailment += nli["entailment"] * weight
            weighted_contradiction += nli["contradiction"] * weight
            weighted_evidence_conf += evidence_conf * rank_discount
            total_weight += weight

        if total_weight > 0:
            avg_entailment = weighted_entailment / total_weight
            avg_contradiction = weighted_contradiction / total_weight
            avg_evidence_strength = weighted_evidence_conf / total_weight
        else:
            avg_entailment = 0.0
            avg_contradiction = 0.5
            avg_evidence_strength = 0.0

        # Bayesian confidence fusion equation:
        # P(Truth | Evidence, Detector) = w_nli * Entailment + w_verif * Ev_Strength + w_det * (1 - Det_Hallucination_Prob)
        detector_factuality = max(0.0, (1.0 - detector_prob) * detector_confidence)
        
        calibrated_conf = (
            self.config.weight_nli_entailment * avg_entailment +
            self.config.weight_verifier_evidence * avg_evidence_strength +
            self.config.weight_detector_signal * detector_factuality
        )

        # Penalty for explicit contradiction
        contradiction_penalty = avg_contradiction * 0.4
        calibrated_conf = max(0.0, min(1.0, round(calibrated_conf - contradiction_penalty, 4)))

        # Risk score calculation
        risk_score = round(
            0.5 * avg_contradiction + 
            0.3 * (1.0 - calibrated_conf) + 
            0.2 * detector_prob, 4
        )

        severity = self._calculate_severity(risk_score, avg_contradiction)

        return {
            "calibrated_confidence": calibrated_conf,
            "overall_entailment": round(avg_entailment, 4),
            "overall_contradiction": round(avg_contradiction, 4),
            "evidence_strength": round(avg_evidence_strength, 4),
            "risk_score": risk_score,
            "severity": severity
        }

    def _calculate_severity(self, risk_score: float, contradiction_prob: float) -> str:
        """
        Determines severity level based on risk score and contradiction level.
        """
        if risk_score >= self.config.severity_critical or contradiction_prob >= self.config.reject_contradiction_threshold:
            return "CRITICAL"
        elif risk_score >= self.config.severity_high or contradiction_prob >= self.config.correct_contradiction_threshold:
            return "HIGH"
        elif risk_score >= self.config.severity_medium:
            return "MEDIUM"
        else:
            return "LOW"
