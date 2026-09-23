"""
llm_judge.py
────────────
OPTIONAL uncertainty resolver for the standalone HalluciGuard Detector.

Purpose (handoff spec §25-32):
    A verification-RISK re-rater for claims that DeBERTa leaves in the
    UNCERTAIN band. It answers "how verification-worthy does this claim look?"
    It DOES NOT decide factual truth, retrieve evidence, or call the Verifier.

Hard constraints, enforced by construction:
    - DISABLED by default (v1 ships with enabled=false).
    - Only ever invoked on UNCERTAIN-band claims.
    - Strict budget: at most `max_calls_per_request` calls, spent on the
      most-uncertain claims first.
    - Must never search the web, retrieve, call Tavily/Wikipedia/RAG/NLI,
      access Memory, the Verifier, or the Corrector.
    - Must never emit TRUE / FALSE / VERIFIED / HALLUCINATED.

Output vocabulary is strictly {LOW_RISK, UNCERTAIN, HIGH_RISK}.
On any error/timeout the judge abstains (returns None) and the caller keeps
the DeBERTa result — failure never lowers risk.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)

_ALLOWED_RISK = {"LOW_RISK", "UNCERTAIN", "HIGH_RISK"}
_FORBIDDEN_TERMS = {"TRUE", "FALSE", "VERIFIED", "HALLUCINATED"}

JUDGE_SYSTEM_PROMPT = (
    "You are a verification-triage rater inside a hallucination-detection "
    "pipeline. Given a user query, a draft answer, and one atomic claim, judge "
    "ONLY how much the claim warrants deeper verification. You do NOT decide "
    "whether the claim is true or false, and you have NO access to external "
    "evidence, search, or tools. Respond with STRICT JSON: "
    '{\"risk\": \"LOW_RISK|UNCERTAIN|HIGH_RISK\", \"confidence\": 0.0-1.0, '
    '\"reason\": \"short phrase\"}. Never output the words TRUE, FALSE, '
    "VERIFIED, or HALLUCINATED."
)


@dataclass
class JudgeVerdict:
    claim_id: str
    risk: str                 # LOW_RISK | UNCERTAIN | HIGH_RISK
    confidence: float
    reason: str


@dataclass
class JudgeConfig:
    enabled: bool = False
    provider: str = "openrouter"
    model: str = "qwen/qwen3-14b"
    temperature: float = 0.0
    timeout_seconds: float = 20.0
    max_calls_per_request: int = 5

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "JudgeConfig":
        d = d or {}
        return cls(
            enabled=bool(d.get("enabled", False)),
            provider=str(d.get("provider", "openrouter")),
            model=str(d.get("model", "qwen/qwen3-14b")),
            temperature=float(d.get("temperature", 0.0)),
            timeout_seconds=float(d.get("timeout_seconds", 20.0)),
            max_calls_per_request=int(d.get("max_calls_per_request", 5)),
        )


# A CompletionFn takes (system_prompt, user_prompt) and returns raw model text.
# Injecting it keeps this module free of any network/provider dependency and
# trivially testable. None = no client wired => judge abstains.
CompletionFn = Callable[[str, str], str]


class LLMJudge:
    """Optional UNCERTAIN-band verification-risk re-rater."""

    def __init__(self, config: JudgeConfig, completion_fn: Optional[CompletionFn] = None) -> None:
        self.config = config
        self._completion_fn = completion_fn

    @property
    def available(self) -> bool:
        return self.config.enabled and self._completion_fn is not None

    def _build_user_prompt(self, query: str, answer: str, claim: str, deberta_risk: float) -> str:
        return (
            f"[QUERY]\n{query}\n\n"
            f"[ANSWER]\n{answer}\n\n"
            f"[CLAIM]\n{claim}\n\n"
            f"[DEBERTA_RISK]\n{deberta_risk:.4f}\n\n"
            "Return the strict JSON verdict now."
        )

    def _parse_verdict(self, claim_id: str, raw: str) -> Optional[JudgeVerdict]:
        try:
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end == -1:
                return None
            data = json.loads(raw[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            logger.warning("[LLMJudge] Unparseable verdict for %s -> abstain.", claim_id)
            return None

        risk = str(data.get("risk", "")).upper().strip()
        if risk not in _ALLOWED_RISK:
            logger.warning("[LLMJudge] Illegal risk '%s' for %s -> abstain.", risk, claim_id)
            return None

        # Safety: reject any smuggled factual-truth verdict.
        blob = json.dumps(data).upper()
        if any(term in blob for term in _FORBIDDEN_TERMS):
            logger.warning("[LLMJudge] Forbidden factual term in verdict for %s -> abstain.", claim_id)
            return None

        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        reason = str(data.get("reason", ""))[:200]
        return JudgeVerdict(claim_id=claim_id, risk=risk, confidence=conf, reason=reason)

    def judge_claim(
        self, query: str, answer: str, claim_id: str, claim_text: str, deberta_risk: float
    ) -> Optional[JudgeVerdict]:
        """Judge a single claim. Returns None (abstain) on any failure."""
        if not self.available:
            return None
        try:
            raw = self._completion_fn(  # type: ignore[misc]
                JUDGE_SYSTEM_PROMPT,
                self._build_user_prompt(query, answer, claim_text, deberta_risk),
            )
        except Exception as exc:  # network/timeout/etc -> abstain
            logger.warning("[LLMJudge] Completion failed for %s: %s -> abstain.", claim_id, exc)
            return None
        return self._parse_verdict(claim_id, raw)

    def judge_uncertain_claims(
        self,
        query: str,
        answer: str,
        uncertain_claims: Sequence[dict],
    ) -> List[JudgeVerdict]:
        """
        Judge only UNCERTAIN-band claims, spending the budget on the most
        uncertain first (raw score closest to 0.5).

        `uncertain_claims`: list of {claim_id, text, raw_score}.
        Returns the list of non-abstaining verdicts (may be empty).
        """
        if not self.available or not uncertain_claims:
            return []

        ordered = sorted(
            uncertain_claims, key=lambda c: abs(float(c.get("raw_score", 0.5)) - 0.5)
        )
        budget = max(0, int(self.config.max_calls_per_request))
        verdicts: List[JudgeVerdict] = []
        for c in ordered[:budget]:
            v = self.judge_claim(
                query=query,
                answer=answer,
                claim_id=str(c.get("claim_id", "")),
                claim_text=str(c.get("text", "")),
                deberta_risk=float(c.get("raw_score", 0.5)),
            )
            if v is not None:
                verdicts.append(v)
        return verdicts
