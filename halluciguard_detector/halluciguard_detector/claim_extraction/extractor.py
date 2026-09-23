"""
ClaimExtractor module.
Positions claim extraction as a pre-processing stage.
Outputs canonical immutable ExtractedClaim objects.
"""

from __future__ import annotations
import json
import logging
import os
import re
from typing import List, Tuple, Optional, Dict, Any

from ..schemas import ExtractedClaim, ExtractionMode
from .prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_TEMPLATE
from .sentence_fallback import sentence_split
from .validator import validate_extracted_claim

logger = logging.getLogger(__name__)


class ClaimExtractor:
    """
    Claim Extractor supporting LLM extraction with deterministic fallback.
    """

    def __init__(
        self,
        max_claims: int = 100,
        min_claim_len: int = 10,
        openrouter_api_key: Optional[str] = None,
        llm_model: str = "qwen/qwen3-14b",
    ) -> None:
        self.max_claims = max_claims
        self.min_claim_len = min_claim_len
        self.api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.llm_model = llm_model
        if self.llm_model in ("qwen/qwen3-4b", "qwen/qwen3-4b:free"):
            self.llm_model = "qwen/qwen3-14b"

    def extract(
        self,
        user_query: str,
        draft_answer: str,
        supplied_claims: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[ExtractedClaim], ExtractionMode, List[str]]:
        """
        Main extraction entry point.
        Returns (claims, extraction_mode, degraded_reasons).
        """
        reasons = []

        # If user explicitly supplied claims, use PASSTHROUGH
        if supplied_claims is not None:
            claims = []
            for idx, item in enumerate(supplied_claims, start=1):
                cid = item.get("claim_id", f"claim_{idx:03d}")
                text = item.get("text", str(item))
                claims.append(
                    ExtractedClaim(
                        claim_id=cid,
                        text=text,
                        source_sentence_idx=item.get("source_sentence_idx", 0),
                        char_span=item.get("char_span", None),
                    )
                )
            if len(claims) > 40:
                reasons.append("CLAIM_CAP")
            return claims, ExtractionMode.PASSTHROUGH, reasons

        if not draft_answer or not draft_answer.strip():
            return [], ExtractionMode.SENTENCE_FALLBACK, ["EMPTY_ANSWER"]

        # Attempt LLM extraction if API key available
        if self.api_key:
            try:
                claims, err = self._llm_extract(user_query, draft_answer)
                if claims:
                    if len(claims) > 40:
                        reasons.append("CLAIM_CAP")
                    return claims, ExtractionMode.LLM, reasons
                if err:
                    reasons.append(f"LLM_EXTRACTION_FAILED: {err}")
            except Exception as exc:
                logger.warning("[ClaimExtractor] LLM extraction error: %s", exc)
                reasons.append(f"LLM_EXTRACTION_EXCEPTION: {str(exc)}")

        # Deterministic fallback
        claims = self._fallback_extract(user_query, draft_answer)
        if len(claims) > 40:
            reasons.append("CLAIM_CAP")
        return claims, ExtractionMode.SENTENCE_FALLBACK, reasons

    def _llm_extract(self, user_query: str, draft_answer: str) -> Tuple[List[ExtractedClaim], Optional[str]]:
        """Call OpenRouter LLM for claim extraction."""
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        user_prompt = EXTRACTION_USER_TEMPLATE.format(
            user_query=user_query, draft_answer=draft_answer
        )
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1000,
        }

        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

        # Parse JSON
        content_clean = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`").strip()
        raw_list = json.loads(content_clean)
        if not isinstance(raw_list, list):
            return [], "NOT_A_JSON_LIST"

        claims = []
        for idx, item in enumerate(raw_list, start=1):
            if not isinstance(item, str) or len(item.strip()) < self.min_claim_len:
                continue
            claim_text = item.strip()

            # Validate against hallucinated entities/nums
            is_valid, flags = validate_extracted_claim(claim_text, draft_answer, user_query)
            if not is_valid:
                logger.warning("[ClaimExtractor] Validation dropped claim: %s (flags: %s)", claim_text, flags)
                continue

            cid = f"claim_{idx:03d}"
            claims.append(
                ExtractedClaim(
                    claim_id=cid,
                    text=claim_text,
                    source_sentence_idx=0,
                )
            )

        return claims, None

    def _fallback_extract(self, user_query: str, draft_answer: str) -> List[ExtractedClaim]:
        """Sentence splitter fallback."""
        sent_tuples = sentence_split(draft_answer)
        claims = []
        for count, (s_idx, s_text) in enumerate(sent_tuples, start=1):
            cid = f"claim_{count:03d}"
            claims.append(
                ExtractedClaim(
                    claim_id=cid,
                    text=s_text,
                    source_sentence_idx=s_idx,
                )
            )
        return claims
