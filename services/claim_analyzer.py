"""LLM Claim Analyzer — the single gate between the Base LLM draft and retrieval.

Architecture (per the HalluciGuard claim-gating contract):

    BASE LLM  ->  CLAIM ANALYZER (ONE llm call over the whole draft)
              ->  FACTUAL CLAIMS ONLY
              ->  (one batched retrieval request, grouped by claim_id)
              ->  Verifier -> Judge -> Corrector / ReVerifier

The Claim Analyzer receives the *entire* Base LLM response in a single LLM call
(never one call per sentence), identifies the domain, and extracts the
independently verifiable factual claims. It discards meta / discourse / opinion
/ instruction / transition / question / disclaimer text so that sentences like
"Actually, that isn't correct." never reach retrieval or the Verifier.

CRITICAL INVARIANT: the Claim Analyzer NEVER decides whether a claim is true or
false. It only classifies *whether a span is a checkable factual assertion* and
prepares search queries. Truth is the Verifier's job, exclusively.

When the hosted LLM layer is unavailable (offline tests, no credentials, or a
provider error), this module degrades deterministically to the existing
:class:`ClaimDecomposer` filtering so the pipeline — and its tests — keep working
without a network. The deterministic path produces the same structured output
shape, only with heuristic classification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger("HalluciGuard.ClaimAnalyzer")


# ---------------------------------------------------------------------------
# Classification taxonomy (contract §"Claim classification types").
# Only FACTUAL_CLAIM is ever routed to retrieval / the Verifier.
# ---------------------------------------------------------------------------
FACTUAL_CLAIM = "FACTUAL_CLAIM"
NON_FACTUAL = "NON_FACTUAL"
META = "META"
OPINION = "OPINION"
INSTRUCTION = "INSTRUCTION"
TRANSITION = "TRANSITION"
QUESTION = "QUESTION"
DISCLAIMER = "DISCLAIMER"
OTHER = "OTHER"

CLAIM_TYPES = frozenset(
    {FACTUAL_CLAIM, NON_FACTUAL, META, OPINION, INSTRUCTION, TRANSITION, QUESTION, DISCLAIMER, OTHER}
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


# Feature flag + budget. The analyzer is on by default but always has a
# deterministic offline fallback, so disabling it (or losing the network) never
# breaks the pipeline — it only trades LLM classification for heuristics.
ANALYZER_ENABLED = _env_flag("HALLUCIGUARD_CLAIM_ANALYZER_ENABLED", True)
ANALYZER_MAX_TOKENS = _env_int("HALLUCIGUARD_CLAIM_ANALYZER_MAX_TOKENS", 1024)


@dataclass
class ClaimCandidate:
    """One span extracted from the Base LLM draft, with its classification.

    A candidate is *routable* (sent to retrieval + the Verifier) only when
    ``claim_type == FACTUAL_CLAIM``. All other spans are retained purely for
    observability (§34: every discarded sentence is auditable).
    """

    claim_id: str
    claim_text: str
    original_sentence: str
    claim_type: str = FACTUAL_CLAIM
    domain: str = "general"
    search_queries: list[str] = field(default_factory=list)
    # Free-text rationale for the classification. Never a truth judgement.
    reason: str = ""

    @property
    def is_factual(self) -> bool:
        return self.claim_type == FACTUAL_CLAIM

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "text": self.claim_text,  # legacy key consumed by _verifier_node
            "original_sentence": self.original_sentence,
            "claim_type": self.claim_type,
            "domain": self.domain,
            "search_queries": list(self.search_queries),
            "is_factual": self.is_factual,
            "retrieval_triggered": self.is_factual,
            "reason": self.reason,
        }


@dataclass
class ClaimAnalysis:
    """Result of analyzing an entire Base LLM draft in one pass."""

    domain: str
    candidates: list[ClaimCandidate]
    source: str  # "llm" | "fallback"
    llm_error: str | None = None

    @property
    def factual_claims(self) -> list[ClaimCandidate]:
        return [c for c in self.candidates if c.is_factual]

    @property
    def discarded(self) -> list[ClaimCandidate]:
        return [c for c in self.candidates if not c.is_factual]

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "source": self.source,
            "llm_error": self.llm_error,
            "factual_count": len(self.factual_claims),
            "discarded_count": len(self.discarded),
            "candidates": [c.to_dict() for c in self.candidates],
        }


# ---------------------------------------------------------------------------
# LLM prompt. One call, whole draft in, structured JSON out.
# ---------------------------------------------------------------------------
_SYSTEM_INSTRUCTION = """\
You are a CLAIM ANALYZER inside a hallucination-verification pipeline. You are \
given the COMPLETE response written by another AI (the "draft"). Your ONLY job \
is to extract the independently verifiable FACTUAL claims from the draft and \
prepare them for evidence retrieval.

You MUST NOT decide whether any claim is true or false. Truth is judged later by \
a separate Verifier using retrieved evidence. Extraction is not endorsement.

Rules:
1. Identify the overall DOMAIN of the draft (e.g. general, technology, medicine, \
science, law, finance, security, history).
2. Split the draft into spans and classify each span as exactly one of:
   FACTUAL_CLAIM  - an objectively checkable assertion about the world
                    (who/what/when/where/how-many; entities, dates, quantities).
   NON_FACTUAL    - a statement that is not objectively checkable.
   META           - self-referential / discourse about the answer itself
                    (e.g. "Actually, that isn't correct.", "As I mentioned").
   OPINION        - a subjective judgement or preference.
   INSTRUCTION    - a command or step directed at the reader.
   TRANSITION     - a connective / filler phrase ("Let me explain.", "Here's why").
   QUESTION       - an interrogative.
   DISCLAIMER     - a hedge or caveat ("I am not a doctor", "this may be outdated").
   OTHER          - anything that fits none of the above.
3. Only FACTUAL_CLAIM spans are verified downstream. Do NOT over-filter: a short \
   factual statement like "Google was founded in 1998." IS a FACTUAL_CLAIM.
4. Prefer ATOMIC claims: split compound facts into separate claims, but preserve \
   semantic context and NEVER invent claims not present in the draft.
5. For each FACTUAL_CLAIM, write 1-3 concise web SEARCH QUERIES that would \
   retrieve evidence for or AGAINST it (counter-evidence matters).
6. Keep original_sentence = the verbatim source sentence for each span.

Return ONLY a JSON object, no prose, of the form:
{
  "domain": "<domain>",
  "claims": [
    {
      "claim_id": "c1",
      "claim_text": "<atomic claim>",
      "original_sentence": "<verbatim source sentence>",
      "claim_type": "FACTUAL_CLAIM|NON_FACTUAL|META|OPINION|INSTRUCTION|TRANSITION|QUESTION|DISCLAIMER|OTHER",
      "search_queries": ["...", "..."]
    }
  ]
}
"""


def _build_prompt(draft: str, user_query: str, domain_hint: str) -> str:
    parts = [_SYSTEM_INSTRUCTION, ""]
    if user_query:
        parts.append(f"USER QUESTION (context only, do not verify it):\n{user_query}\n")
    if domain_hint:
        parts.append(f"DOMAIN HINT (optional): {domain_hint}\n")
    parts.append("DRAFT TO ANALYZE:\n\"\"\"\n" + (draft or "") + "\n\"\"\"\n")
    parts.append("Respond with the JSON object only.")
    return "\n".join(parts)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from an LLM reply (handles code fences/prose)."""
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", candidate).strip()
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(candidate)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _coerce_candidates(obj: dict[str, Any]) -> tuple[str, list[ClaimCandidate]]:
    """Turn a parsed analyzer JSON object into validated candidates."""
    domain = str(obj.get("domain") or "general").strip().lower() or "general"
    raw_claims = obj.get("claims")
    if not isinstance(raw_claims, list):
        return domain, []

    candidates: list[ClaimCandidate] = []
    for idx, raw in enumerate(raw_claims, start=1):
        if not isinstance(raw, dict):
            continue
        claim_text = str(raw.get("claim_text") or raw.get("text") or "").strip()
        original = str(raw.get("original_sentence") or claim_text).strip()
        if not claim_text and not original:
            continue
        ctype = str(raw.get("claim_type") or FACTUAL_CLAIM).strip().upper()
        if ctype not in CLAIM_TYPES:
            ctype = OTHER
        queries = raw.get("search_queries")
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list):
            queries = []
        search_queries = [str(q).strip() for q in queries if str(q).strip()]
        if ctype == FACTUAL_CLAIM and not search_queries:
            search_queries = [claim_text or original]
        candidates.append(
            ClaimCandidate(
                claim_id=str(raw.get("claim_id") or f"c{idx}").strip() or f"c{idx}",
                claim_text=claim_text or original,
                original_sentence=original or claim_text,
                claim_type=ctype,
                domain=str(raw.get("domain") or domain).strip().lower() or domain,
                search_queries=search_queries,
                reason=str(raw.get("reason") or "").strip(),
            )
        )
    return domain, candidates


# ---------------------------------------------------------------------------
# Deterministic offline fallback. No network, no LLM — reuses the existing
# ClaimDecomposer._is_checkable filter and adds coarse classification of the
# non-factual spans purely for observability (§34).
# ---------------------------------------------------------------------------
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_META_RE = re.compile(
    r"\b(actually|to clarify|as (i|we) (mentioned|said|noted)|in summary|"
    r"that('?s| is)( not| n'?t)? correct|isn'?t correct|let me (explain|clarify)|"
    r"here'?s why|note that|keep in mind|as an ai)\b",
    re.IGNORECASE,
)
_OPINION_RE = re.compile(
    r"\b(i (think|believe|feel|guess)|in my opinion|arguably|probably|"
    r"seems? (to|like)|the best|the worst|beautiful|amazing|terrible)\b",
    re.IGNORECASE,
)
_INSTRUCTION_RE = re.compile(
    r"^\s*(please\b|you (should|must|can|could|may)\b|try\b|consider\b|make sure\b|run\b|"
    r"install\b|use\b|do not\b|don'?t\b)",
    re.IGNORECASE,
)
_DISCLAIMER_RE = re.compile(
    r"\b(i am not a (doctor|lawyer|financial)|not (medical|legal|financial) advice|"
    r"consult (a|your)|this (may|might) be outdated|i cannot|i can'?t (verify|confirm))\b",
    re.IGNORECASE,
)
_TRANSITION_RE = re.compile(
    r"^\s*(however|therefore|thus|so|well|okay|ok|sure|first|second|next|finally|"
    r"in conclusion|moreover|furthermore|additionally)\b[\s,]*$",
    re.IGNORECASE,
)


def _classify_nonfactual(sentence: str) -> tuple[str, str]:
    """Coarsely classify a span the checkable-filter rejected. (type, reason)."""
    s = sentence.strip()
    if not s:
        return OTHER, "empty span"
    if s.endswith("?"):
        return QUESTION, "interrogative"
    if _DISCLAIMER_RE.search(s):
        return DISCLAIMER, "hedge/caveat phrasing"
    if _META_RE.search(s):
        return META, "discourse/self-reference phrasing"
    if _INSTRUCTION_RE.search(s):
        return INSTRUCTION, "imperative phrasing"
    if _OPINION_RE.search(s):
        return OPINION, "subjective phrasing"
    if _TRANSITION_RE.match(s) or len(s.split()) <= 3:
        return TRANSITION, "connective/short filler"
    return NON_FACTUAL, "not an objectively checkable assertion"


def fallback_analyze(draft: str, user_query: str = "", domain_hint: str = "") -> ClaimAnalysis:
    """Deterministic, network-free analysis. Always available."""
    # Import lazily so the module has no hard dependency on the verifier package
    # at import time (keeps unit tests that only touch heuristics lightweight).
    try:
        from agents.verifier_agent.claims.claim_decomposer import ClaimDecomposer

        is_checkable = ClaimDecomposer._is_checkable  # classmethod
    except Exception:  # pragma: no cover - defensive
        is_checkable = None

    domain = (domain_hint or "general").strip().lower() or "general"
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(draft or "") if s.strip()]
    candidates: list[ClaimCandidate] = []
    factual_idx = 0
    for i, sentence in enumerate(sentences, start=1):
        # A clearly non-factual span (meta/discourse/opinion/instruction/
        # disclaimer/question/transition) is filtered regardless of what the
        # decomposer's checkable-filter thinks — the decomposer is tuned for
        # extraction, not discourse removal, and misses spans like
        # "Let me explain." that must never reach retrieval.
        forced_type, forced_reason = _classify_nonfactual(sentence)
        if forced_type not in (NON_FACTUAL, OTHER):
            candidates.append(
                ClaimCandidate(
                    claim_id=f"x{i}",
                    claim_text=sentence,
                    original_sentence=sentence,
                    claim_type=forced_type,
                    domain=domain,
                    search_queries=[],
                    reason=f"deterministic: {forced_reason}",
                )
            )
            continue
        checkable = bool(is_checkable(sentence)) if is_checkable else _fallback_checkable(sentence)
        if checkable:
            factual_idx += 1
            candidates.append(
                ClaimCandidate(
                    claim_id=f"c{factual_idx}",
                    claim_text=sentence,
                    original_sentence=sentence,
                    claim_type=FACTUAL_CLAIM,
                    domain=domain,
                    search_queries=[sentence],
                    reason="deterministic: passed checkable-claim filter",
                )
            )
        else:
            ctype, reason = _classify_nonfactual(sentence)
            candidates.append(
                ClaimCandidate(
                    claim_id=f"x{i}",
                    claim_text=sentence,
                    original_sentence=sentence,
                    claim_type=ctype,
                    domain=domain,
                    search_queries=[],
                    reason=f"deterministic: {reason}",
                )
            )
    return ClaimAnalysis(domain=domain, candidates=candidates, source="fallback")


def _fallback_checkable(sentence: str) -> bool:
    """Minimal checkable heuristic used only if ClaimDecomposer is unavailable.

    Treat a span as a factual claim when it is a declarative statement of at
    least four tokens that none of the non-factual classifiers match.
    """
    s = sentence.strip()
    if not s or s.endswith("?") or len(s.split()) < 4:
        return False
    if (
        _META_RE.search(s)
        or _OPINION_RE.search(s)
        or _INSTRUCTION_RE.search(s)
        or _DISCLAIMER_RE.search(s)
        or _TRANSITION_RE.match(s)
    ):
        return False
    return True


class ClaimAnalyzer:
    """One-call factual-claim extractor with a deterministic offline fallback.

    ``llm_service`` is injected for testability; when omitted a shared
    :class:`BaseLLMService` is lazily constructed. Any LLM failure (no key,
    network error, unparseable output) degrades to :func:`fallback_analyze`
    instead of raising, so the pipeline never blocks on the analyzer.
    """

    def __init__(self, llm_service: Any = None, *, enabled: bool | None = None) -> None:
        self._llm_service = llm_service
        self._enabled = ANALYZER_ENABLED if enabled is None else enabled

    def _get_llm_service(self) -> Any:
        if self._llm_service is None:
            from services.base_llm_service import BaseLLMService

            self._llm_service = BaseLLMService()
        return self._llm_service

    async def analyze(
        self, draft: str, user_query: str = "", domain_hint: str = ""
    ) -> ClaimAnalysis:
        """Analyze the whole draft in ONE LLM call; fall back deterministically."""
        if not (draft and draft.strip()):
            return ClaimAnalysis(domain=(domain_hint or "general"), candidates=[], source="fallback")

        if not self._enabled:
            return fallback_analyze(draft, user_query, domain_hint)

        try:
            service = self._get_llm_service()
            prompt = _build_prompt(draft, user_query, domain_hint)
            result = await service.generate(
                user_query=prompt,
                generation_mode="normal",
                temperature=0.0,
                max_tokens=ANALYZER_MAX_TOKENS,
            )
            status = getattr(result, "status", None)
            content = getattr(result, "draft_response", "") or ""
            if status != "success" or not content.strip():
                err = getattr(result, "error", None) or f"status={status}"
                logger.warning("Claim Analyzer LLM unusable (%s); using fallback.", err)
                fb = fallback_analyze(draft, user_query, domain_hint)
                fb.llm_error = str(err)
                return fb

            parsed = _extract_json(content)
            if parsed is None:
                logger.warning("Claim Analyzer LLM returned non-JSON; using fallback.")
                fb = fallback_analyze(draft, user_query, domain_hint)
                fb.llm_error = "non-JSON LLM output"
                return fb

            domain, candidates = _coerce_candidates(parsed)
            if not candidates:
                # Empty extraction is a valid outcome ONLY if the draft truly has
                # no checkable claims; a malformed empty list more likely means a
                # parsing miss, so cross-check with the deterministic path.
                fb = fallback_analyze(draft, user_query, domain_hint)
                if fb.factual_claims:
                    logger.info("Claim Analyzer LLM produced no claims; using fallback claims.")
                    fb.llm_error = "empty LLM claim list"
                    return fb
                return ClaimAnalysis(domain=domain or fb.domain, candidates=[], source="llm")
            return ClaimAnalysis(domain=domain, candidates=candidates, source="llm")
        except Exception as exc:  # never block the pipeline on the analyzer
            logger.warning("Claim Analyzer LLM error (%s); using fallback.", exc)
            fb = fallback_analyze(draft, user_query, domain_hint)
            fb.llm_error = str(exc)
            return fb

    def analyze_sync(
        self, draft: str, user_query: str = "", domain_hint: str = ""
    ) -> ClaimAnalysis:
        """Synchronous wrapper for use in non-async call sites (graph nodes)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.analyze(draft, user_query, domain_hint))
        # Already inside an event loop: run the blocking-safe deterministic path
        # rather than nesting loops. (Graph nodes here are synchronous, so this
        # branch is only hit in unusual embedded-async contexts.)
        return fallback_analyze(draft, user_query, domain_hint)


__all__ = [
    "FACTUAL_CLAIM",
    "NON_FACTUAL",
    "META",
    "OPINION",
    "INSTRUCTION",
    "TRANSITION",
    "QUESTION",
    "DISCLAIMER",
    "OTHER",
    "CLAIM_TYPES",
    "ClaimCandidate",
    "ClaimAnalysis",
    "ClaimAnalyzer",
    "fallback_analyze",
]
