"""Offline tests for the LLM Claim Analyzer (deterministic fallback + LLM path).

These never touch the network: the deterministic path is exercised via
``enabled=False`` and the LLM path via an injected fake service, so the
Snehith/Google meta-filtering guarantee is verifiable without credentials.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from services.claim_analyzer import (
    FACTUAL_CLAIM,
    META,
    QUESTION,
    ClaimAnalyzer,
    fallback_analyze,
)


# --------------------------------------------------------------------------
# Deterministic fallback
# --------------------------------------------------------------------------
def _types(analysis, text_fragment):
    return [c.claim_type for c in analysis.candidates if text_fragment.lower() in c.claim_text.lower()]


def test_meta_sentence_is_not_a_factual_claim():
    """The exact sentence that broke the pipeline must never be routed."""
    analysis = fallback_analyze("Actually, that isn't correct.")
    factual = analysis.factual_claims
    assert factual == [], f"meta sentence leaked as factual: {[c.claim_text for c in factual]}"
    assert any(c.claim_type == META for c in analysis.candidates)


def test_genuine_short_factual_claim_is_preserved():
    """Must NOT over-filter a real short fact."""
    analysis = fallback_analyze("Google was founded in 1998.")
    assert any(c.is_factual for c in analysis.factual_claims)


def test_question_is_filtered():
    analysis = fallback_analyze("Who founded Google?")
    assert all(not c.is_factual for c in analysis.candidates)
    assert any(c.claim_type == QUESTION for c in analysis.candidates)


def test_transition_and_factual_mixed():
    draft = "Let me explain. Python 3.12 was released in 2023."
    analysis = fallback_analyze(draft)
    factual = [c.claim_text for c in analysis.factual_claims]
    assert any("Python 3.12" in t for t in factual)
    assert all("Let me explain" not in t for t in factual)


def test_founder_claim_routed_and_search_query_present():
    analysis = fallback_analyze("Snehith is the founder of Google.")
    factual = analysis.factual_claims
    assert len(factual) == 1
    assert factual[0].search_queries  # a query is prepared for retrieval


def test_fallback_strips_markdown_from_retrieval_claim_but_keeps_original():
    draft = "Microsoft was founded by **Bill Gates** and **Paul Allen** in April\u202f1975."
    analysis = fallback_analyze(draft)
    factual = analysis.factual_claims
    assert [c.claim_text for c in factual] == [
        "Microsoft was founded by Bill Gates in April 1975.",
        "Microsoft was founded by Paul Allen in April 1975.",
    ]
    assert all("**Bill Gates**" in c.original_sentence for c in factual)
    assert all(c.search_queries == [c.claim_text] for c in factual)


def test_every_span_is_recorded_for_observability():
    draft = "Actually, that isn't correct. Google was founded in 1998."
    analysis = fallback_analyze(draft)
    # both spans present in candidates (audit trail), only one factual
    assert len(analysis.candidates) == 2
    assert len(analysis.factual_claims) == 1
    assert len(analysis.discarded) == 1


# --------------------------------------------------------------------------
# LLM path with an injected fake service
# --------------------------------------------------------------------------
@dataclass
class _FakeResult:
    status: str
    draft_response: str
    error: str | None = None


class _FakeLLM:
    def __init__(self, payload: str, status: str = "success"):
        self._payload = payload
        self._status = status

    async def generate(self, **kwargs):
        return _FakeResult(status=self._status, draft_response=self._payload)


def test_llm_path_parses_structured_json():
    payload = (
        '{"domain":"technology","claims":[{"claim_id":"c1",'
        '"claim_text":"Google was founded in 1998.",'
        '"original_sentence":"Google was founded in 1998.",'
        '"claim_type":"FACTUAL_CLAIM","search_queries":["when was Google founded"]}]}'
    )
    analyzer = ClaimAnalyzer(llm_service=_FakeLLM(payload), enabled=True)
    analysis = asyncio.run(analyzer.analyze("Google was founded in 1998."))
    assert analysis.source == "llm"
    assert analysis.domain == "technology"
    assert len(analysis.factual_claims) == 1
    assert analysis.factual_claims[0].search_queries == ["when was Google founded"]


def test_llm_failure_falls_back_deterministically():
    analyzer = ClaimAnalyzer(llm_service=_FakeLLM("not json at all", status="success"), enabled=True)
    analysis = asyncio.run(analyzer.analyze("Google was founded in 1998."))
    assert analysis.source == "fallback"
    assert analysis.llm_error


def test_llm_never_asserts_truth_only_classifies():
    """A FACTUAL_CLAIM carries no truth field; downstream Verifier decides."""
    payload = (
        '{"domain":"general","claims":[{"claim_id":"c1",'
        '"claim_text":"Snehith is the founder of Google.",'
        '"original_sentence":"Snehith is the founder of Google.",'
        '"claim_type":"FACTUAL_CLAIM","search_queries":["Google founder"]}]}'
    )
    analyzer = ClaimAnalyzer(llm_service=_FakeLLM(payload), enabled=True)
    analysis = asyncio.run(analyzer.analyze("Snehith is the founder of Google."))
    c = analysis.factual_claims[0].to_dict()
    assert "is_true" not in c and "verdict" not in c
    assert c["claim_type"] == FACTUAL_CLAIM
