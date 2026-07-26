"""
HalluciGuard - Judge Agent Unit & Integration Test Suite
Validates NLI scoring, confidence calibration formulas, decision engine outputs, and Corrector payload formats.
"""

import unittest
from config import JudgeConfig
from nli_engine import NLIEngine
from confidence_calibrator import ConfidenceCalibrator
from decision_engine import DecisionEngine
from judge_agent import JudgeAgent

class TestJudgeAgent(unittest.TestCase):

    def setUp(self):
        self.config = JudgeConfig(use_huggingface=False) # Use deterministic fallback for unit tests
        self.nli_engine = NLIEngine(use_hf=False)
        self.calibrator = ConfidenceCalibrator(config=self.config)
        self.decision_engine = DecisionEngine(config=self.config)
        self.agent = JudgeAgent(config=self.config)

    def test_nli_entailment(self):
        evidence = "Metformin is indicated for Type 2 Diabetes."
        claim = "Metformin is indicated for Type 2 Diabetes."
        scores = self.nli_engine.predict(evidence, claim)
        self.assertGreater(scores["entailment"], 0.60)
        self.assertLess(scores["contradiction"], 0.20)

    def test_nli_contradiction(self):
        evidence = "Aspirin is not recommended for pediatric viral infections."
        claim = "Aspirin is recommended for pediatric viral infections."
        scores = self.nli_engine.predict(evidence, claim)
        self.assertGreater(scores["contradiction"], 0.50)

    def test_accept_decision_flow(self):
        detector_out = {"hallucination_probability": 0.05, "confidence_score": 0.95}
        verifier_out = {
            "claim_evidence_pairs": [
                {
                    "claim": "Revenue increased by 15% in Q3.",
                    "evidence": "Q3 revenue increased by 15% year-over-year.",
                    "evidence_confidence": 0.95,
                    "rank": 1
                }
            ]
        }
        res = self.agent.evaluate(detector_out, verifier_out)
        self.assertEqual(res["decision"], "ACCEPT")
        self.assertGreaterEqual(res["metrics"]["calibrated_confidence"], 0.80)

    def test_correct_decision_flow(self):
        detector_out = {"hallucination_probability": 0.80, "confidence_score": 0.90}
        verifier_out = {
            "claim_evidence_pairs": [
                {
                    "claim": "The patient was prescribed 500mg Aspirin.",
                    "evidence": "The patient was not prescribed Aspirin; 500mg Paracetamol was given.",
                    "evidence_confidence": 0.90,
                    "rank": 1
                }
            ]
        }
        res = self.agent.evaluate(detector_out, verifier_out)
        self.assertIn(res["decision"], ["CORRECT", "REJECT"])
        self.assertIn(res["severity"], ["HIGH", "CRITICAL"])
        self.assertGreater(len(res["corrector_payload"]["hallucinated_claims"]), 0)

    def test_abstain_decision_flow(self):
        detector_out = {"hallucination_probability": 0.50, "confidence_score": 0.30}
        verifier_out = {"claim_evidence_pairs": []}  # No evidence
        res = self.agent.evaluate(detector_out, verifier_out)
        self.assertEqual(res["decision"], "ABSTAIN")

if __name__ == "__main__":
    unittest.main()
