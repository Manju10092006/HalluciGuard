"""
HalluciGuard - Legacy Decision Engine Compatibility Layer
Delegates directly to the canonical JudgeAgent implementation in judge_agent.py.
"""

from typing import Dict, Any, Optional
from judge_agent import JudgeAgent


class DecisionIntelligenceEngine:
    """
    Legacy compatibility wrapper delegating to canonical JudgeAgent.
    """

    def __init__(self, config: Optional[Any] = None):
        self.agent = JudgeAgent(config=config)

    def evaluate_decision(
        self,
        calibration_results: Dict[str, Any],
        evidence_intel: Dict[str, Any],
        consensus_data: Dict[str, Any],
        memory_data: Dict[str, Any],
        criticality_data: Dict[str, Any],
        domain_policy: Any,
        user_query: str,
        draft_response: str
    ) -> Any:
        """Delegates decision arbitration to canonical JudgeAgent."""
        verifier_res = {
            "domain": getattr(domain_policy, "domain_name", "General Knowledge"),
            "claim_evidence_pairs": evidence_intel.get("processed_pairs", [])
        }
        res = self.agent.evaluate(
            verifier_result=verifier_res,
            user_query=user_query,
            draft_response=draft_response
        )
        return {
            "decision": res.decision.value if hasattr(res.decision, "value") else str(res.decision),
            "reason": res.reason,
            "explanation": res.explanation,
            "confidence": res.confidence,
            "correction_request": res.correction_request
        }
