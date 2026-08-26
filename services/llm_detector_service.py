from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Optional

from agents.detector_agent import DetectionResult, DetectorAgent
from services.base_llm_service import BaseLLMConfig, BaseLLMService, GenerationResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMDetectorSliceResult:
    """Acceptance contract structure for Base LLM -> Detector vertical slice."""

    user_query: str
    draft_response: str
    generation: dict[str, Any]
    detector: dict[str, Any] | None

    def model_dump(self) -> dict[str, Any]:
        """Convert the slice result to a dictionary."""
        return asdict(self)


class BaseLLMDetectorService:
    """Orchestrates Step 2 vertical slice: Base LLM -> Detector.
    
    Data Flow:
        user_query -> BaseLLMService -> draft_response -> DetectorAgent -> Risk & Decision
    """

    def __init__(
        self,
        llm_service: Optional[BaseLLMService] = None,
        detector_agent: Optional[DetectorAgent] = None,
    ) -> None:
        self.llm_service = llm_service or BaseLLMService()
        self.detector_agent = detector_agent or DetectorAgent()

    async def execute_slice(
        self,
        user_query: str,
        conversation_history: list[dict[str, str]] | None = None,
        generation_mode: str = "normal",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMDetectorSliceResult:
        """
        Execute the Base LLM -> Detector vertical slice.

        Args:
            user_query: The user's question or prompt.
            conversation_history: Optional conversation context.
            generation_mode: Either "normal" or "stress_test".
            temperature: Optional temperature override.
            max_tokens: Optional token limit.

        Returns:
            A LLMDetectorSliceResult containing the draft response, generation metadata, and detector decision.
        """
        # Step 1: Base LLM Generation
        gen_result: GenerationResult = await self.llm_service.generate(
            user_query=user_query,
            conversation_history=conversation_history,
            generation_mode=generation_mode,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        gen_dict: dict[str, Any] = {
            "status": gen_result.status,
            "provider": gen_result.provider,
            "model": gen_result.model,
            "latency_ms": gen_result.latency_ms,
            "request_id": gen_result.request_id,
            "finish_reason": gen_result.finish_reason,
        }
        if gen_result.error:
            gen_dict["error"] = gen_result.error
            gen_dict["error_code"] = gen_result.error_code

        # Guard: If LLM generation failed or produced empty draft_response, DO NOT execute Detector
        draft_response = gen_result.draft_response or ""
        if gen_result.status != "success" or not draft_response.strip():
            logger.warning(
                "[LLMDetectorService] Skipping detector: LLM generation failed or returned empty response."
            )
            return LLMDetectorSliceResult(
                user_query=user_query,
                draft_response=draft_response,
                generation=gen_dict,
                detector=None,
            )

        # Step 2: Detector Agent Execution
        detection_result: DetectionResult = self.detector_agent.detect(
            user_query=user_query,
            llm_response=draft_response,
        )

        next_act_str = str(detection_result.next_action.value).upper()
        decision = "VERIFY" if next_act_str == "VERIFY" else "ACCEPT"

        detector_dict: dict[str, Any] = {
            "confidence_score": detection_result.confidence_score,
            "hallucination_probability": detection_result.hallucination_probability,
            "risk_tier": str(detection_result.risk_level.value),
            "decision": decision,
            "model_source": detection_result.model_source,
        }

        return LLMDetectorSliceResult(
            user_query=user_query,
            draft_response=draft_response,
            generation=gen_dict,
            detector=detector_dict,
        )
