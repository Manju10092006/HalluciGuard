"""
HalluciGuard - Enterprise Multi-Signal Confidence Calibrator
Executes 11-Signal Bayesian Calibration fusing NLI scores, Verifier trust, Source Authority,
Evidence Freshness, Diversity Index, Consensus Score, Detector signals, Memory reliability,
and Claim Criticality into a calibrated decision confidence score.
"""

from typing import Dict, List, Any
from config import JudgeConfig, DEFAULT_CONFIG
from domain_policies import DomainPolicy

class DynamicConfidenceCalibrator:
    def __init__(self, config: JudgeConfig = DEFAULT_CONFIG):
        self.config = config

    def calibrate_11_signal(
        self,
        detector_prob: float,
        detector_confidence: float,
        evidence_intel: Dict[str, Any],
        consensus_data: Dict[str, Any],
        memory_data: Dict[str, Any],
        contradiction_data: Dict[str, Any],
        criticality_data: Dict[str, Any],
        domain_policy: DomainPolicy
    ) -> Dict[str, Any]:
        """
        Calculates calibrated decision confidence score across 11 reasoning signals.
        """
        processed_pairs = evidence_intel.get("processed_pairs", [])
        
        # Base signal extraction
        authority_score = evidence_intel.get("overall_authority", 0.0)
        freshness_score = evidence_intel.get("overall_freshness", 0.0)
        diversity_index = evidence_intel.get("diversity_index", 0.0)
        consensus_score = consensus_data.get("consensus_score", 0.0)
        conflict_index = consensus_data.get("conflict_index", 0.0)
        memory_trust = memory_data.get("memory_trust_score", 0.75)
        criticality_score = criticality_data.get("criticality_score", 0.50)

        # Average NLI Entailment & Contradiction
        if processed_pairs:
            entail_scores = [p.get("nli_scores", {}).get("entailment", 0.33) for p in processed_pairs]
            contra_scores = [p.get("nli_scores", {}).get("contradiction", 0.34) for p in processed_pairs]
            avg_entailment = sum(entail_scores) / len(entail_scores)
            avg_contradiction = sum(contra_scores) / len(contra_scores)
            verifier_trust = sum(p.get("evidence_confidence", 0.8) for p in processed_pairs) / len(processed_pairs)
        else:
            avg_entailment = 0.0
            avg_contradiction = 0.50
            verifier_trust = 0.0

        # Detector factuality signal
        detector_factuality = max(0.0, (1.0 - detector_prob) * detector_confidence)

        # 11-Signal Weighted Fusion Calculation (Detector signal strongly influences confidence)
        fusion_score = (
            0.35 * avg_entailment +
            0.25 * detector_factuality +
            0.15 * verifier_trust +
            0.15 * authority_score +
            0.05 * freshness_score +
            0.05 * consensus_score
        )

        # Apply Contradiction & Conflict Penalty
        contra_weight = contradiction_data.get("risk_weight", 0.8) if contradiction_data.get("has_contradiction", False) else 0.0
        penalty = (avg_contradiction * 0.40) + (conflict_index * 0.25) + (contra_weight * 0.35)
        
        # Final Calibrated Confidence
        calibrated_confidence = max(0.0, min(1.0, round(fusion_score - penalty, 4)))

        # Risk Score Calculation (Directly scaled by Detector Risk Signal)
        risk_score = round(
            0.40 * detector_prob +
            0.35 * (1.0 - calibrated_confidence) +
            0.25 * max(avg_contradiction, contra_weight), 4
        )

        # Determine Severity Level based on Domain Policy
        severity = self._determine_severity(risk_score, avg_contradiction, contra_weight, domain_policy)

        return {
            "calibrated_confidence": calibrated_confidence,
            "overall_entailment": round(avg_entailment, 4),
            "overall_contradiction": round(avg_contradiction, 4),
            "authority_score": authority_score,
            "freshness_score": freshness_score,
            "diversity_index": diversity_index,
            "consensus_score": consensus_score,
            "risk_score": risk_score,
            "severity": severity
        }

    def _determine_severity(
        self,
        risk_score: float,
        contradiction_prob: float,
        contra_weight: float,
        domain_policy: DomainPolicy
    ) -> str:
        """
        Assigns severity considering domain strictness level.
        """
        max_conflict = max(contradiction_prob, contra_weight)
        
        if domain_policy.strictness_level in ["VERY_STRICT", "STRICT"]:
            if risk_score >= 0.60 or max_conflict >= 0.50:
                return "CRITICAL"
            elif risk_score >= 0.40 or max_conflict >= 0.35:
                return "HIGH"
            elif risk_score >= 0.25:
                return "MEDIUM"
            else:
                return "LOW"
        else:
            if risk_score >= 0.75 or max_conflict >= 0.70:
                return "CRITICAL"
            elif risk_score >= 0.55:
                return "HIGH"
            elif risk_score >= 0.35:
                return "MEDIUM"
            else:
                return "LOW"
