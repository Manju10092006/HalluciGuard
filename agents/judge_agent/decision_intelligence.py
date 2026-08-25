"""
HalluciGuard - Decision Intelligence Compatibility Layer
Delegates directly to the canonical JudgeAgent implementation in judge_agent.py.
"""

from typing import Dict, Any, Optional, Union
from judge_agent import JudgeAgent
from orchestration.schemas import JudgeResult, CorrectionRequest, ReverificationResult


class DecisionIntelligenceEngine:
    """
    Compatibility wrapper delegating to canonical JudgeAgent.
    """

    def __init__(self, config: Optional[Any] = None):
        self.agent = JudgeAgent(config=config)

    def evaluate(
        self,
        user_query: str = "",
        draft_response: str = "",
        detector_output: Optional[Dict[str, Any]] = None,
        verifier_output: Optional[Dict[str, Any]] = None,
        domain: str = "",
        memory_context: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
        **kwargs
    ) -> JudgeResult:
        """Delegates evaluation to canonical JudgeAgent."""
        # Handle dict or positional argument variations gracefully
        v_res = verifier_output if verifier_output is not None else kwargs.get("verifier_result", {})
        d_res = detector_output if detector_output is not None else kwargs.get("detector_result")
        
        return self.agent.evaluate(
            verifier_result=v_res,
            detector_result=d_res,
            user_query=user_query,
            draft_response=draft_response,
            domain=domain,
            retry_count=retry_count
        )

    def build_correction_request(
        self,
        execution_id: str,
        user_query: str,
        original_response: str,
        judge_verdict: Any
    ) -> Optional[CorrectionRequest]:
        """Returns correction_request if attached to JudgeResult."""
        if hasattr(judge_verdict, "correction_request"):
            return judge_verdict.correction_request
        return None

    def evaluate_reverification(
        self,
        reverification_result: Union[ReverificationResult, Dict[str, Any]]
    ) -> JudgeResult:
        """Delegates post-correction reverification evaluation to canonical JudgeAgent."""
        return self.agent.evaluate(
            verifier_result={},
            reverification_result=reverification_result
        )
