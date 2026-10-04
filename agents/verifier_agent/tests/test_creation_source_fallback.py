"""Offline integration tests for technical-domain creator-source routing."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api.pipeline import VerificationPipeline
from schemas.models import Passage, SuspiciousClaim, VerifierInputV2
from services.n8n_retrieval_client import N8NRetrievalResult
from agents.verifier_agent.tests.test_verifier_trace_binding import _isolated_pipeline


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_technical_creator_claim_attempts_general_sources_without_fabricating_evidence(monkeypatch, failure):
    pipeline, technical = _isolated_pipeline(monkeypatch, [
        N8NRetrievalResult(success=True, passages=[])])
    technical.name = "computer_science"
    technical.search = AsyncMock(return_value=[])
    evidence = Passage(title="Microsoft", source="wikipedia", publication_date="unknown",
        url="https://en.wikipedia.org/wiki/Microsoft", source_id="microsoft",
        snippet="Microsoft was founded by Bill Gates and Paul Allen.", relevance_score=0.88)
    general = SimpleNamespace(name="general", search=AsyncMock(
        side_effect=RuntimeError("dummy-secret-never-public") if failure else None,
        return_value=[evidence]))
    monkeypatch.setattr("api.pipeline.get_registry", lambda: SimpleNamespace(
        get_adapter=lambda domain: general if domain == "general" else technical))
    monkeypatch.setattr(pipeline.claim_decomposer, "decompose", lambda text: [text])
    claim = "Snehith is the founder of Microsoft."
    result = await pipeline.verify(VerifierInputV2(query_id="creator-source", domain="computer_science",
        source_mode="wikipedia_only", suspicious_claims=[SuspiciousClaim(claim_id="c1", text=claim)]))
    assert technical.search.await_count > 0
    assert general.search.await_count == 2
    for call in general.search.await_args_list:
        assert "Microsoft" in call.args[0] and "Snehith" not in call.args[0]
        assert call.kwargs["source_mode"] == "wikipedia_only"
    report = result.claim_evidence[0]
    snapshot = report.retrieval_trace["subclaim_execution"][0]
    observations = snapshot["supplementary_retrieval"]
    assert len(observations) == 2
    if failure:
        assert all(o["status"] == "failed" and o["error_type"] == "RuntimeError" for o in observations)
        assert snapshot["counts"]["retrieved"] == 0
        assert snapshot["nli_selected_passages"] == []
        assert str(report.verdict.value) == "unverified"
        assert "dummy-secret-never-public" not in result.model_dump_json()
    else:
        assert all(o["status"] == "completed" and o["returned_count"] == 1 for o in observations)
        assert snapshot["counts"]["retrieved"] == 1  # Same passage fetched twice, deduplicated.
        assert snapshot["nli_selected_passages"][0]["url"] == evidence.url


@pytest.mark.asyncio
@pytest.mark.parametrize("domain,claim", [
    ("general", "Snehith is the founder of Microsoft."),
    ("computer_science", "This compiler optimizes loops."),
])
async def test_supplementary_creation_lookup_does_not_change_unrelated_routes(monkeypatch, domain, claim):
    pipeline, adapter = _isolated_pipeline(monkeypatch, [
        N8NRetrievalResult(success=True, passages=[])])
    adapter.search = AsyncMock(return_value=[])
    domains = []

    def get_adapter(selected):
        domains.append(selected)
        return adapter

    monkeypatch.setattr("api.pipeline.get_registry", lambda: SimpleNamespace(get_adapter=get_adapter))
    monkeypatch.setattr(pipeline.claim_decomposer, "decompose", lambda text: [text])
    result = await pipeline.verify(VerifierInputV2(query_id="creator-scope", domain=domain,
        suspicious_claims=[SuspiciousClaim(claim_id="c1", text=claim)]))
    assert domains == [domain]
    assert result.claim_evidence[0].retrieval_trace["subclaim_execution"][0]["supplementary_retrieval"] == []
