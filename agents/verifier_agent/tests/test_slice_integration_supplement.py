"""
Supplementary Unit Tests — Section 29 Coverage Completion.

Section 29 asks for 12 focused unit tests. Eleven are already covered by the
existing suites (which must NOT be deleted):

  orchestration/tests/test_llm_detector_verifier_slice.py
    · Case 1  — Detector LOW risk    -> verifier NOT called
    · Case 2  — Detector MEDIUM/HIGH -> verifier (n8n retrieval) called
    · Case 12 — Slice result stays JSON-serializable

  agents/verifier_agent/tests/test_n8n_retrieval.py
    · Case 3  — n8n success       -> evidence normalized to Passage schema
    · Case 4  — n8n HTTP failure  -> controlled N8NRetrievalResult (403/500/timeout)
    · Case 5  — n8n invalid JSON  -> controlled result
    · Case 6  — empty evidence    -> controlled result
    · Case 8  — DeBERTa NLI receives the real claim + real snippet
    · Case 9  — evidence scoring preserves neutral / irrelevant (not contradiction)
    · Case 10 — final verdict computed by Python (not adopted from n8n)
    · Case 11 — no duplicate retrieval when n8n succeeds

This file adds the ONE remaining case plus a reinforcing invariant, without
modifying any existing test:

    · Case 7  — BGE reranker receives the REAL claim + REAL retrieved evidence
    · Section 9 — adapter_score (n8n hint) and bge_score (Python reranker) stay
                  distinct all the way to the final EvidenceItem

These are hermetic unit tests: the n8n boundary, BGE reranker, and DeBERTa NLI
are mocked so the suite runs fast and offline. They prove the *wiring* (that BGE
receives real inputs and that the two scores are kept separate). Real BGE/DeBERTa
model execution is proven end-to-end by the Section-30 live validation:
    python scripts/test_llm_detector_verifier_slice.py --all-tests
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.n8n_retrieval_client import N8NRetrievalResult
from schemas.models import Passage, SuspiciousClaim, VerifierInputV2
from api.pipeline import VerificationPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _n8n_result_with(snippet: str, adapter_hint: float) -> N8NRetrievalResult:
    """Build a successful n8n retrieval result with a single normalized passage.

    Mirrors the real n8n client contract: relevance_score is left at 0.0
    (reserved for the Python BGE reranker) and the n8n adapter score is carried
    in source_confidence_hint.
    """
    return N8NRetrievalResult(
        success=True,
        passages=[
            Passage(
                title="Wikipedia: Paris",
                source="wikipedia",
                url="https://en.wikipedia.org/wiki/Paris",
                publication_date="2024-01-01",
                snippet=snippet,
                source_id="wiki_paris",
                relevance_score=0.0,               # reserved for Python BGE
                source_confidence_hint=adapter_hint,  # adapter score from n8n
            )
        ],
    )


_ENTAILING_NLI = [
    {
        "label": "entailment",
        "entailment_score": 0.95,
        "contradiction_score": 0.02,
        "neutral_score": 0.03,
        "confidence": 0.95,
        "degraded": False,
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# Section 29 · Case 7 — BGE reranker receives real claim + real evidence
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bge_reranker_receives_real_claim_and_evidence():
    """The BGE reranker must be invoked with the REAL claim and the REAL retrieved snippet.

    We spy on VerificationPipeline.reranker.rerank via a MagicMock whose
    side_effect returns the same passages with a deterministic relevance score,
    so downstream stages proceed without loading the ~1.3GB BGE model. The
    assertion is on the reranker's *inputs*, which is exactly what Case 7 requires.
    """
    pipeline = VerificationPipeline()

    real_claim = "Paris is the capital of France."
    real_snippet = "Paris is the capital and most populous city of France."

    def _rerank_spy(claim, passages, k=5, model_name=None):
        return [p.model_copy(update={"relevance_score": 0.88}) for p in passages]

    rerank_mock = MagicMock(side_effect=_rerank_spy)

    with patch.object(pipeline.cache, "get", new_callable=AsyncMock, return_value=None), \
         patch.object(pipeline.n8n_client, "retrieve_evidence", new_callable=AsyncMock) as mock_n8n, \
         patch.object(pipeline.reranker, "rerank", rerank_mock), \
         patch.object(pipeline.nli_engine, "batch_classify") as mock_nli:
        mock_n8n.return_value = _n8n_result_with(real_snippet, adapter_hint=0.67)
        mock_nli.return_value = list(_ENTAILING_NLI)

        payload = VerifierInputV2(
            query_id="bge-real-evidence-test",
            domain="general",
            suspicious_claims=[SuspiciousClaim(claim_id="c1", text=real_claim)],
        )

        await pipeline.verify(payload)

        rerank_mock.assert_called_once()
        call = rerank_mock.call_args
        passed_claim = call.args[0]
        passed_passages = call.args[1]

        assert real_claim.lower().rstrip(".") in str(passed_claim).lower(), (
            f"BGE reranker did not receive the real claim (got: {passed_claim!r})"
        )
        assert any(real_snippet in getattr(p, "snippet", "") for p in passed_passages), (
            "BGE reranker did not receive the real retrieved snippet"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — adapter_score and bge_score remain distinct end-to-end
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_adapter_and_bge_scores_remain_separate_end_to_end():
    """adapter_score (from the n8n hint) and bge_score (from the Python reranker)
    must stay distinct all the way to the final EvidenceItem — bge_score must NOT
    be silently set equal to adapter_score.
    """
    pipeline = VerificationPipeline()

    real_claim = "Paris is the capital of France."
    real_snippet = "Paris is the capital and most populous city of France."
    adapter_hint = 0.67
    bge_relevance = 0.88

    def _rerank_spy(claim, passages, k=5, model_name=None):
        return [p.model_copy(update={"relevance_score": bge_relevance}) for p in passages]

    with patch.object(pipeline.cache, "get", new_callable=AsyncMock, return_value=None), \
         patch.object(pipeline.n8n_client, "retrieve_evidence", new_callable=AsyncMock) as mock_n8n, \
         patch.object(pipeline.reranker, "rerank", MagicMock(side_effect=_rerank_spy)), \
         patch.object(pipeline.nli_engine, "batch_classify") as mock_nli:
        mock_n8n.return_value = _n8n_result_with(real_snippet, adapter_hint=adapter_hint)
        mock_nli.return_value = list(_ENTAILING_NLI)

        payload = VerifierInputV2(
            query_id="score-separation-test",
            domain="general",
            suspicious_claims=[SuspiciousClaim(claim_id="c1", text=real_claim)],
        )

        output = await pipeline.verify(payload)

        report = output.claim_evidence[0]
        assert len(report.evidence) >= 1, "expected at least one decision-grade evidence item"
        ev = report.evidence[0]

        assert ev.adapter_score == pytest.approx(adapter_hint), (
            f"adapter_score should reflect the n8n hint; got {ev.adapter_score}"
        )
        assert ev.bge_score == pytest.approx(bge_relevance), (
            f"bge_score should reflect the Python reranker relevance; got {ev.bge_score}"
        )
        assert ev.adapter_score != ev.bge_score, (
            "adapter_score and bge_score must remain distinct (bge_score must not be the adapter score)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 29 · Case 10 (reinforcement) — verdict is one of the four Python states
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_final_verdict_is_python_four_state_taxonomy():
    """The pipeline must emit a verdict drawn from the Python four-state taxonomy
    (VERIFIED / CONTRADICTED / UNVERIFIED / CONFLICTED) computed locally — n8n
    supplies retrieval only and N8NRetrievalResult carries no verdict field.
    """
    pipeline = VerificationPipeline()

    # N8NRetrievalResult has no verdict attribute — retrieval-only boundary.
    assert not hasattr(N8NRetrievalResult(success=True, passages=[]), "verdict")

    real_claim = "Paris is the capital of France."
    real_snippet = "Paris is the capital and most populous city of France."

    def _rerank_spy(claim, passages, k=5, model_name=None):
        return [p.model_copy(update={"relevance_score": 0.90}) for p in passages]

    with patch.object(pipeline.cache, "get", new_callable=AsyncMock, return_value=None), \
         patch.object(pipeline.n8n_client, "retrieve_evidence", new_callable=AsyncMock) as mock_n8n, \
         patch.object(pipeline.reranker, "rerank", MagicMock(side_effect=_rerank_spy)), \
         patch.object(pipeline.nli_engine, "batch_classify") as mock_nli:
        mock_n8n.return_value = _n8n_result_with(real_snippet, adapter_hint=0.70)
        mock_nli.return_value = list(_ENTAILING_NLI)

        payload = VerifierInputV2(
            query_id="python-verdict-test",
            domain="general",
            suspicious_claims=[SuspiciousClaim(claim_id="c1", text=real_claim)],
        )

        output = await pipeline.verify(payload)
        verdict = output.claim_evidence[0].verdict.value

        assert verdict in {"verified", "contradicted", "unverified", "conflicted"}, (
            f"verdict '{verdict}' is not in the Python four-state taxonomy"
        )
