"""Graph-node level regression tests for the Claim Analyzer gate.

Offline: the analyzer is forced onto its deterministic fallback so no network
or credentials are required. Verifies the Snehith/Google meta-sentence never
reaches the Verifier as a claim and that the whole-draft fallback is disabled
once claims are gated.
"""

from __future__ import annotations

import asyncio

import pytest

from orchestration.graph import _claim_analyzer_node


@pytest.fixture(autouse=True)
def _force_deterministic(monkeypatch):
    # Force the analyzer's deterministic offline path (no LLM/network).
    monkeypatch.setattr("services.claim_analyzer.ANALYZER_ENABLED", False)


def _run(state):
    return asyncio.run(_claim_analyzer_node(state))


def test_meta_and_fact_mixed_response_gates_to_fact_only():
    state = {
        "llm_response": "Actually, that isn't correct. Google was founded in 1998.",
        "user_query": "Who founded Google?",
        "domain": "general",
    }
    out = _run(state)
    assert out["claims_gated"] is True
    texts = [c["text"] for c in out["detected_claims"]]
    assert any("Google was founded in 1998" in t for t in texts)
    assert all("isn't correct" not in t for t in texts)


def test_pure_meta_response_yields_no_factual_claims():
    state = {
        "llm_response": "Actually, that isn't correct.",
        "user_query": "Is Snehith the founder of Google?",
        "domain": "general",
    }
    out = _run(state)
    assert out["claims_gated"] is True
    assert out["detected_claims"] == []


def test_snehith_founder_claim_is_gated_through_for_verification():
    state = {
        "llm_response": "Snehith is the founder of Google.",
        "user_query": "Who founded Google?",
        "domain": "general",
    }
    out = _run(state)
    assert len(out["detected_claims"]) == 1
    assert "claim_analysis" in out
    assert out["claim_analysis"]["factual_count"] == 1
