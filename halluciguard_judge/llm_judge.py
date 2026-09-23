"""
halluciguard_judge / llm_judge.py
──────────────────────────────────
Optional LLM-as-a-judge for UNCERTAIN (MEDIUM-risk) claims.

This module is ONLY invoked when the DeBERTa classifier returns a
probability in the uncertain band (LOW_THRESHOLD < P < HIGH_THRESHOLD).

Design (Datadog two-stage approach):
    Stage 1: Chain-of-thought reasoning (no format restriction)
              -> allows LLM to reason freely before committing
    Stage 2: JSON extraction from CoT
              -> deterministic parsing, small model

Cost control:
    - Called only for MEDIUM risk claims
    - Single claim per call (no batching to keep latency bounded)
    - Timeout enforced per call
    - Falls back to classifier probability if API fails

References:
    Datadog (2025): "Detecting hallucinations with LLM-as-a-judge"
    - Two-stage: free reasoning -> structured output
    - Rubric: contradictions + unsupported claims separately
    - Context framed as "expert advice" (asymmetry for grounding)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import httpx

from .models import LLMJudgeResult
from .prompts import (
    COT_FAITHFULNESS_TEMPLATE,
    COT_SYSTEM_PROMPT,
    COT_USER_TEMPLATE,
    EXTRACT_SYSTEM_PROMPT,
    EXTRACT_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


class LLMJudge:
    """Two-stage LLM-as-a-judge for uncertain hallucination cases.

    Usage
    -----
        judge = LLMJudge(api_key=..., model="qwen/qwen3-14b")
        result = judge.evaluate(
            user_query="Who created Java?",
            claim_text="Java was created by Dennis Ritchie.",
        )
        # result.is_hallucination -> True
        # result.probability      -> 0.92
        # result.reasoning        -> "Dennis Ritchie created C, not Java..."

    Design note: uses httpx (sync) so it is compatible with FastAPI/Flask
    endpoints that may already manage their own event loop.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "qwen/qwen3-14b",
        timeout: int = 25,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _call_llm(self, system: str, user: str, max_tokens: int = 512) -> str:
        """Make a single OpenRouter chat completion call. Returns assistant text."""
        if not self.api_key:
            raise ValueError("LLM judge requires OPENROUTER_API_KEY.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://halluciguard.app",
            "X-Title": "HalluciGuard-Judge",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,  # low temp for deterministic judging
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    def _extract_json(self, text: str) -> dict:
        """Extract JSON object from LLM output (handles markdown fences)."""
        # Strip markdown code fences
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning("[LLMJudge] Could not parse JSON from: %r", text[:200])
        return {}

    def evaluate(
        self,
        user_query: str,
        claim_text: str,
        context: Optional[str] = None,
        classifier_probability: Optional[float] = None,
    ) -> LLMJudgeResult:
        """Run two-stage LLM judge evaluation on a single claim.

        Args:
            user_query:              Original user question.
            claim_text:              Atomic claim to evaluate.
            context:                 Optional RAG context (for faithfulness).
            classifier_probability:  Fallback probability if judge fails.

        Returns:
            LLMJudgeResult with refined probability and reasoning.
        """
        fallback_prob = classifier_probability if classifier_probability is not None else 0.5

        try:
            # ── Stage 1: Chain-of-thought reasoning ──────────────────────────
            if context:
                cot_user = COT_FAITHFULNESS_TEMPLATE.format(
                    context=context[:3000],  # truncate to avoid token overflow
                    user_query=user_query,
                    claim_text=claim_text,
                )
            else:
                cot_user = COT_USER_TEMPLATE.format(
                    user_query=user_query,
                    claim_text=claim_text,
                )

            cot_output = self._call_llm(
                system=COT_SYSTEM_PROMPT,
                user=cot_user,
                max_tokens=600,
            )
            logger.debug("[LLMJudge] CoT output: %s", cot_output[:200])

            # ── Stage 2: Structured extraction ───────────────────────────────
            extract_user = EXTRACT_USER_TEMPLATE.format(cot_reasoning=cot_output)
            extract_output = self._call_llm(
                system=EXTRACT_SYSTEM_PROMPT,
                user=extract_user,
                max_tokens=200,
            )

            parsed = self._extract_json(extract_output)

            is_hallucination = bool(parsed.get("is_hallucination", False))
            probability = float(parsed.get("probability", fallback_prob))
            probability = max(0.0, min(1.0, probability))
            reasoning = str(parsed.get("reasoning_summary", ""))
            flagged_contradiction = bool(parsed.get("flagged_as_contradiction", False))
            flagged_unsupported = bool(parsed.get("flagged_as_unsupported", False))

            return LLMJudgeResult(
                claim_text=claim_text,
                is_hallucination=is_hallucination,
                probability=probability,
                reasoning=reasoning,
                flagged_as_contradiction=flagged_contradiction,
                flagged_as_unsupported=flagged_unsupported,
                raw_cot_output=cot_output,
            )

        except httpx.TimeoutException:
            logger.warning("[LLMJudge] Timeout — falling back to classifier probability.")
            return LLMJudgeResult(
                claim_text=claim_text,
                is_hallucination=fallback_prob >= 0.5,
                probability=fallback_prob,
                reasoning="LLM judge timed out; using classifier probability.",
            )
        except Exception as exc:
            logger.error("[LLMJudge] Evaluation failed: %s", exc)
            return LLMJudgeResult(
                claim_text=claim_text,
                is_hallucination=fallback_prob >= 0.5,
                probability=fallback_prob,
                reasoning=f"LLM judge error: {exc}",
            )


__all__ = ["LLMJudge"]
