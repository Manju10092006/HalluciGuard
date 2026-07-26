"""
HalluciGuard - Judge Agent Main Module
Orchestrates the entire Judge Agent workflow:
Inputs: Detector Signals + Verifier Evidence + Memory Context
Outputs: Trust Decision (Accept, Verify Again, Correct, Reject, Abstain), Severity, Explanation, Corrector Payload.
"""

import time
import logging
from typing import Dict, List, Any, Optional

from config import JudgeConfig, DEFAULT_CONFIG
from nli_engine import NLIEngine
from confidence_calibrator import ConfidenceCalibrator
from decision_engine import DecisionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HalluciGuard.JudgeAgent")

class JudgeAgent:
    def __init__(self, config: Optional[JudgeConfig] = None):
        self.config = config or DEFAULT_CONFIG
        logger.info(f"Initializing HalluciGuard Judge Agent (Model: {self.config.default_nli_model})...")
        
        self.nli_engine = NLIEngine(
            model_name=self.config.default_nli_model,
            use_hf=self.config.use_huggingface
        )
        self.calibrator = ConfidenceCalibrator(config=self.config)
        self.decision_engine = DecisionEngine(config=self.config)
        logger.info("Judge Agent initialized and ready.")

    def evaluate(
        self,
        detector_output: Dict[str, Any],
        verifier_output: Dict[str, Any],
        user_query: str = "",
        draft_response: str = "",
        memory_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes full evaluation pipeline for a request:
        1. Extract Detector hallucination probabilities and Verifier claim-evidence pairs.
        2. Perform NLI entailment / contradiction scoring via DeBERTa/BART engine.
        3. Calibrate confidence scores & compute Bayesian risk scores.
        4. Make trust decision and output payload for Corrector Agent.
        """
        start_time = time.time()
        logger.info("Starting Judge Agent evaluation pipeline...")

        # 1. Normalize Detector Inputs
        detector_prob = detector_output.get("hallucination_probability", 0.3)
        detector_conf = detector_output.get("confidence_score", 0.8)
        detector_risk = detector_output.get("risk_level", "MEDIUM")

        # 2. Extract Verifier Claims & Evidence
        raw_pairs = verifier_output.get("claim_evidence_pairs", [])
        if not raw_pairs and "claims" in verifier_output and "evidence" in verifier_output:
            # Reformat if provided as separate lists
            claims = verifier_output.get("claims", [])
            evidence = verifier_output.get("evidence", [])
            raw_pairs = [
                {"claim": c, "evidence": e if i < len(evidence) else "", "evidence_confidence": 0.85, "rank": i+1}
                for i, c in enumerate(claims)
            ]

        # 3. NLI Engine Inference (Evidence -> Claim entailment/contradiction)
        evaluated_pairs = self.nli_engine.batch_predict(raw_pairs)

        # 4. Confidence Calibration & Risk Assessment
        calibration_results = self.calibrator.calibrate(
            detector_prob=detector_prob,
            detector_confidence=detector_conf,
            evaluated_pairs=evaluated_pairs
        )

        # 5. Decision & Action Selection
        decision_payload = self.decision_engine.evaluate_decision(
            calibration_results=calibration_results,
            evaluated_pairs=evaluated_pairs,
            user_query=user_query,
            draft_response=draft_response
        )

        execution_latency_ms = round((time.time() - start_time) * 1000, 2)

        # Combine complete standardized Judge Agent output schema
        full_judge_output = {
            "agent": "JUDGE_AGENT",
            "status": "SUCCESS",
            "decision": decision_payload["judge_decision"],
            "severity": decision_payload["severity"],
            "reason": decision_payload["reason"],
            "explanation": decision_payload["explanation"],
            "next_action": decision_payload["next_action"],
            "metrics": {
                "calibrated_confidence": calibration_results["calibrated_confidence"],
                "risk_score": calibration_results["risk_score"],
                "overall_entailment": calibration_results["overall_entailment"],
                "overall_contradiction": calibration_results["overall_contradiction"],
                "evidence_strength": calibration_results["evidence_strength"],
                "detector_hallucination_prob": detector_prob,
                "latency_ms": execution_latency_ms
            },
            "claim_evidence_analysis": evaluated_pairs,
            "corrector_payload": decision_payload,
            "observability_bus_signal": {
                "event": "JUDGE_DECISION_EMITTED",
                "decision": decision_payload["judge_decision"],
                "severity": decision_payload["severity"],
                "hallucinated_claim_count": len(decision_payload["hallucinated_claims"]),
                "timestamp_ms": time.time()
            }
        }

        logger.info(f"Evaluation complete. Decision: {decision_payload['judge_decision']} (Severity: {decision_payload['severity']}) in {execution_latency_ms}ms.")
        return full_judge_output


if __name__ == "__main__":
    # Quick sanity check
    agent = JudgeAgent()
    sample_detector = {"hallucination_probability": 0.75, "confidence_score": 0.88, "risk_level": "HIGH"}
    sample_verifier = {
        "claim_evidence_pairs": [
            {
                "claim": "Metformin is used to treat Type 1 Diabetes as first-line therapy.",
                "evidence": "Metformin is indicated as first-line therapy for Type 2 Diabetes mellitus, not Type 1.",
                "evidence_confidence": 0.95,
                "rank": 1,
                "source": "PubMed / NIH"
            }
        ]
    }
    res = agent.evaluate(sample_detector, sample_verifier, user_query="What is Metformin used for?", draft_response="Metformin is first-line for Type 1.")
    print("Decision:", res["decision"])
    print("Reason:", res["reason"])
