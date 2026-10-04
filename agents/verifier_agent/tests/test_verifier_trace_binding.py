"""Actual pipeline boundary helpers with offline fixtures, not model inference."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api.pipeline import VerificationPipeline
from schemas.models import Passage, SuspiciousClaim
from schemas.models import VerifierInputV2
from schemas.retrieval_trace import RetrievalTrace, GateRelevanceAuditTrace
from services.n8n_retrieval_client import N8NRetrievalResult


def _passage(text, url="https://example.test/company", source_id="document"):
    return Passage(title="Company", source="fixture", url=url,
                   source_id=source_id, publication_date="unknown", snippet=text)


def test_raw_retrieval_preserves_multiple_excerpts_from_same_document():
    values = [_passage("Company was founded in 2002."),
              _passage("Company launched in 2003."),
              _passage("Company  was founded\n in 2002.")]
    result = VerificationPipeline._deduplicate_retrieved_passages(values)
    assert result == values[:2]
    assert result[0] is values[0]
    assert result[1].source_id == "document"


def test_anonymous_passages_are_not_silently_dropped():
    passage = Passage(title="", source="fixture", url="", source_id="",
                      publication_date="unknown", snippet="A factual assertion.")
    assert VerificationPipeline._deduplicate_retrieved_passages([passage]) == [passage]


def test_nli_selection_preserves_distinct_chunks_at_same_section_url():
    first = _passage("Company was founded in 2002.").model_copy(update={"relevance_score": 0.9})
    second = _passage("Company launched in 2003.").model_copy(update={"relevance_score": 0.8})
    weak_duplicate = first.model_copy(update={"snippet": "Company  was\n founded in 2002.", "relevance_score": 0.7})
    assert VerificationPipeline._select_relevant_passages([first, second, weak_duplicate]) == [first, second]


def test_pre_model_relevance_gate_does_not_crowd_out_parent_founding_passages(monkeypatch):
    monkeypatch.setenv("HALLUCIGUARD_EVIDENCE_RANKING", "true")
    values = [Passage(title="Microsoft AI (History)", source="wikipedia",
        url=f"https://en.wikipedia.org/wiki/Microsoft_AI#History-{i}", source_id=f"ai-{i}",
        publication_date="unknown", source_confidence_hint=0.8,
        snippet=f"Microsoft AI was founded in 2024 and oversees Microsoft consumer products. Office {i} opened in London.")
        for i in range(3)]
    founding = Passage(title="Microsoft (1972-1985: Founding)", source="wikipedia",
        url="https://en.wikipedia.org/wiki/Microsoft#Founding", source_id="founding",
        publication_date="unknown", source_confidence_hint=0.8,
        snippet="Gates and Allen established Microsoft on April 4, 1975, with Gates as CEO.")
    irrelevant = _passage("Boil water, add the pasta and cook for five minutes.",
                          "https://example.test/pasta", "pasta")
    selected = VerificationPipeline._apply_evidence_ranking([*values, founding, irrelevant],
        claim_text="Snehith is the founder of Microsoft.",
        search_queries=["Microsoft founder", "who is the founder of Microsoft", "Microsoft founded by"])
    assert any(p.source_id == "founding" and p.snippet == founding.snippet for p in selected)
    assert all(p.source_id != "pasta" for p in selected)
    # This assertion is candidate survival, not a supported/contradicted verdict.
    assert all(p.relation_check is None for p in selected)


def test_cache_reuse_binds_new_claim_id_without_replaying_execution_proof():
    cached = {"claim_id": "c1", "claim_text": "Company launched in 2003.",
              "evidence": [], "support_score": 0.9, "contradiction_score": 0.0,
              "trust_score": 0.8, "verdict": "verified",
              "retrieved_documents": 5, "reranked_documents": 4,
              "retrieval_trace": {"backend_execution": {"route": "hybrid_rrf",
                                  "dense_executed": True},
                                  "nli_execution": {"inference_executed": True}}}
    before = deepcopy(cached)
    claim = SuspiciousClaim(claim_id="rev-1", text=cached["claim_text"])
    report = VerificationPipeline._bind_cached_claim_report(cached, claim)
    assert cached == before
    assert report.claim_id == "rev-1"
    assert report.claim_text == claim.text
    assert report.retrieved_documents == report.reranked_documents == 0
    assert report.retrieval_trace["trace_scope"] == "cache_reuse"
    assert report.retrieval_trace["backend_execution"]["route"] == "cache"
    assert report.retrieval_trace["backend_execution"]["dense_attempted"] is False
    assert report.retrieval_trace["nli_execution"]["inference_executed"] is False
    assert report.retrieval_trace["origin"] == before["retrieval_trace"]
    # A caller mutating a returned trace cannot mutate cached provenance.
    report.retrieval_trace["origin"]["backend_execution"]["route"] = "changed"
    assert cached == before


def _isolated_pipeline(monkeypatch, responses):
    pipeline = VerificationPipeline()
    pipeline.cache_enabled = False
    pipeline.certification_mode = False
    pipeline._n8n_client = SimpleNamespace(retrieve_evidence=AsyncMock(side_effect=responses))
    adapter = SimpleNamespace(name="fixture", last_retrieval_trace=RetrievalTrace(
        gate_relevance_audit=GateRelevanceAuditTrace()),
        sources_attempted=["fixture"], sources_succeeded=["fixture"], sources_failed=[])
    monkeypatch.setattr("api.pipeline.get_registry", lambda: SimpleNamespace(get_adapter=lambda domain: adapter))
    monkeypatch.setattr(pipeline, "_apply_evidence_ranking", lambda passages, **kwargs: passages)
    monkeypatch.setattr(pipeline.hybrid_retriever, "retrieve", lambda query, passages, **kwargs: passages)
    monkeypatch.setattr(pipeline.hybrid_retriever, "diagnostics", lambda: {
        "route": "sparse_rrf_lexical", "dense_executed": False, "degraded": True,
        "fallback_reason": "dense_model_unavailable"}, raising=False)
    monkeypatch.setattr(pipeline.reranker, "rerank", lambda claim, passages, **kwargs: passages)
    monkeypatch.setattr(pipeline.reranker, "diagnostics", lambda: {
        "component": "bge_reranker", "status": "unavailable", "inference_executed": False,
        "degraded": True, "model": "fixture", "fallback_reason": "model_load_failed"})
    monkeypatch.setattr(pipeline.nli_engine, "batch_classify", lambda claim, evidences, **kwargs: [
        {"label": "entailment", "entailment_score": 0.99,
         "contradiction_score": 0.005, "neutral_score": 0.005} for _ in evidences])
    monkeypatch.setattr(pipeline.nli_engine, "diagnostics", lambda: {
        "component": "deberta_nli", "model": "fixture", "status": "executed",
        "inference_executed": True, "batch_size": 1})
    return pipeline, adapter


@pytest.mark.asyncio
async def test_fallback_trace_reaches_claim_report_without_fabricated_bge_audit(monkeypatch):
    evidence = _passage("Paris is the capital of France.").model_copy(update={"relevance_score": 0.88})
    pipeline, _ = _isolated_pipeline(monkeypatch, [N8NRetrievalResult(success=True, passages=[evidence])])
    monkeypatch.setattr(pipeline.claim_decomposer, "decompose", lambda text: [text])
    result = await pipeline.verify(VerifierInputV2(query_id="trace", domain="general",
        suspicious_claims=[SuspiciousClaim(claim_id="c1", text="Paris is the capital of France.")]))
    trace = result.claim_evidence[0].retrieval_trace
    assert trace["backend_execution"]["dense_executed"] is False
    assert trace["backend_execution"]["route"] == "sparse_rrf_lexical"
    assert trace["reranker_execution"]["inference_executed"] is False
    assert trace["gate_relevance_audit"]["final_bge_relevance_score"] == 0.0
    assert trace["gate_relevance_audit"]["final_bge_inference_executed"] is False
    snapshot = trace["subclaim_execution"][0]
    assert snapshot["counts"]["nli_selected"] == 1
    assert snapshot["parent_claim_id"] == "c1"
    assert snapshot["reranked_passages"][0]["url"] == evidence.url
    assert snapshot["reranked_passages"][0]["relevance_score"] == evidence.relevance_score
    assert snapshot["nli_selected_passages"][0]["source_id"] == evidence.source_id
    assert snapshot["nli_selected_passages"][0]["snippet"] == evidence.snippet
    assert snapshot["tokenizer_survival"] == "not_exposed"


@pytest.mark.asyncio
async def test_empty_subclaim_does_not_inherit_previous_nli_execution(monkeypatch):
    evidence = _passage("Paris is the capital of France.").model_copy(update={"relevance_score": 0.88})
    pipeline, _ = _isolated_pipeline(monkeypatch, [
        N8NRetrievalResult(success=True, passages=[evidence]),
        N8NRetrievalResult(success=True, passages=[])])
    texts = ["Paris is the capital of France.", "A second assertion lacks evidence."]
    monkeypatch.setattr(pipeline.claim_decomposer, "decompose", lambda text: texts)
    result = await pipeline.verify(VerifierInputV2(query_id="trace-two", domain="general",
        suspicious_claims=[SuspiciousClaim(claim_id="c1", text=" ".join(texts))]))
    snapshots = result.claim_evidence[0].retrieval_trace["subclaim_execution"]
    assert len(snapshots) == 2
    assert snapshots[0]["nli_execution"]["inference_executed"] is True
    assert snapshots[1]["counts"]["nli_selected"] == 0
    assert snapshots[1]["nli_execution"]["status"] == "not_run"
    assert snapshots[1]["nli_execution"]["inference_executed"] is False


@pytest.mark.asyncio
async def test_retrieved_count_precedes_relevance_selection(monkeypatch):
    evidence = _passage("Paris is the capital of France.").model_copy(update={"relevance_score": 0.88})
    irrelevant = _passage("The company sells office furniture.",
                          url="https://example.test/other", source_id="other")
    pipeline, _ = _isolated_pipeline(monkeypatch, [
        N8NRetrievalResult(success=True, passages=[evidence, irrelevant])])
    monkeypatch.setattr(pipeline, "_apply_evidence_ranking",
                        lambda passages, **kwargs: [evidence])
    monkeypatch.setattr(pipeline.claim_decomposer, "decompose", lambda text: [text])
    result = await pipeline.verify(VerifierInputV2(query_id="trace-filter", domain="general",
        suspicious_claims=[SuspiciousClaim(claim_id="c1", text="Paris is the capital of France.")]))
    report = result.claim_evidence[0]
    counts = report.retrieval_trace["subclaim_execution"][0]["counts"]
    assert report.retrieved_documents == counts["retrieved"] == 2
    assert counts["relevance_selected"] == counts["aggregated"] == 1
    assert counts["nli_selected"] == 1
