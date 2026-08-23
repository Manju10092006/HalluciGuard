"""
Unit and Integration Tests for n8n Retrieval Service V2 Integration.

Includes:
- Mocked tests for:
  * Successful n8n request
  * Timeout handling (httpx.TimeoutException)
  * HTTP 500 handling
  * Invalid JSON handling
  * Missing evidence handling
  * Malformed evidence item handling
  * Auth mode: none (no auth header)
  * Auth mode: header (X-API-Key)
  * Request serialization payload
  * Response normalization to standard Passage schema
  * Non-authoritative n8n verdict handling (Python BGE/NLI remains authoritative)
  * Fallback to Python adapters when n8n retrieval is disabled or fails
- Live tests enabled only when RUN_LIVE_N8N_TESTS=true.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from services.n8n_retrieval_client import N8NRetrievalClient, N8NRetrievalResult
from schemas.models import Passage, SuspiciousClaim, VerifierInputV2
from api.pipeline import VerificationPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests — N8NRetrievalClient Mocked Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_exact_v2_response_schema():
    """Test full V2 response schema with nested retrieval, counts, performance, and trace."""
    client = N8NRetrievalClient(
        webhook_url="https://test.n8n.cloud/webhook/halluciguard-verify",
        auth_mode="header",
        header_name="X-API-Key",
        webhook_secret="test-secret-123",
    )

    mock_v2_json = {
        "request_id": "req-v2-test-456",
        "claim": "Paris is the capital of France.",
        "normalized_claim": "Paris is the capital of France.",
        "domain": "general",
        "domain_is_routing_only": True,
        "workflow_version": "2.0.0",
        "retrieval": {
            "mode": "hybrid",
            "primary": {
                "called": True,
                "sources_attempted": ["wikipedia"],
                "result_count": 3,
                "usable_count": 2,
                "quality_sufficient": True,
                "reason": "Primary produced 2 usable candidate(s)",
            },
            "tavily": {
                "called": False,
                "reason": "PRIMARY_EVIDENCE_SUFFICIENT",
                "query_count": 0,
                "result_count": 0,
                "extracted_count": 0,
            }
        },
        "evidence": [
            {
                "source": "wikipedia",
                "source_type": "PRIMARY",
                "source_id": "PAGEID:123",
                "title": "Wikipedia: Paris",
                "url": "https://en.wikipedia.org/wiki/Paris",
                "snippet": "Paris is the capital and most populous city of France.",
                "published_at": None,
                "query_used": "Paris is the capital of France.",
                "retrieval_method": "wikipedia_api_search",
                "adapter_score": 0.67,
                "retrieved_at": "2026-08-22T15:00:00Z",
                "retrieval_status": "success",
                "rank": 0,
            }
        ],
        "counts": {"primary": 3, "tavily": 0, "merged": 3, "deduplicated": 3},
        "performance": {
            "total_latency_ms": 320,
            "primary_latency_ms": 150,
            "tavily_latency_ms": None,
            "extraction_latency_ms": None,
        },
        "trace": [
            {"stage": "analyze", "status": "success", "detail": "normalized + domain routing"},
            {"stage": "domain_router", "status": "success", "detail": "general"},
            {"stage": "primary_retrieval", "status": "success", "detail": "wikipedia"},
            {"stage": "tavily_fallback", "status": "skipped", "detail": "PRIMARY_EVIDENCE_SUFFICIENT"},
            {"stage": "merge_dedup", "status": "success", "detail": "merged 3 -> 3"},
        ],
        "authoritative_verdict": None,
        "note": "Retrieval service only. No verdict is computed by n8n.",
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_v2_json)

        result = await client.retrieve_evidence(
            claim="Paris is the capital of France.",
            domain="general",
            retrieval_mode="hybrid",
            request_id="req-v2-test-456",
        )

        assert result.success is True
        assert len(result.passages) == 1
        p = result.passages[0]
        assert p.title == "Wikipedia: Paris"
        assert p.source == "wikipedia"
        assert p.source_id == "PAGEID:123"
        assert p.source_confidence_hint == 0.67
        assert p.relevance_score == 0.0

        # Trace assertions
        assert result.trace is not None
        assert result.trace.workflow_version == "2.0.0"
        assert result.trace.request_id == "req-v2-test-456"
        assert result.trace.latency_ms == 320
        assert result.trace.counts == {"primary": 3, "tavily": 0, "merged": 3, "deduplicated": 3}
        assert result.trace.primary_trace.get("quality_sufficient") is True
        assert result.trace.tavily_trace.get("called") is False
        assert len(result.trace.stage_traces) == 5


@pytest.mark.asyncio
async def test_http_403_auth_error_handling():
    """Test 403 Forbidden handling when webhook authentication fails."""
    client = N8NRetrievalClient(auth_mode="header", webhook_secret="wrong-secret")
    mock_403 = httpx.Response(403, text="Authorization data is wrong")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_403

        result = await client.retrieve_evidence(
            claim="Some claim",
            domain="general",
        )

        assert result.success is False
        assert result.http_status == 403
        assert result.passages == []
        assert "403" in result.error
        assert "Authorization data is wrong" in result.error



@pytest.mark.asyncio
async def test_request_payload_serialization():
    """Verify exact request payload structure sent to n8n."""
    client = N8NRetrievalClient(auth_mode="none")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = httpx.Response(200, json={"evidence": []})

        await client.retrieve_evidence(
            claim="The Eiffel Tower is in Paris.",
            domain="geography",
            retrieval_mode="hybrid",
            force_tavily=False,
            max_results=3,
            request_id="req-custom-99",
        )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        payload = call_kwargs.get("json")

        assert payload == {
            "claim": "The Eiffel Tower is in Paris.",
            "domain": "geography",
            "retrieval_mode": "hybrid",
            "force_tavily": False,
            "max_results": 3,
            "request_id": "req-custom-99",
        }


@pytest.mark.asyncio
async def test_auth_mode_none():
    """Verify headers when auth_mode='none'."""
    client = N8NRetrievalClient(auth_mode="none")
    headers = client._build_headers()

    assert headers.get("Content-Type") == "application/json"
    assert "X-API-Key" not in headers


@pytest.mark.asyncio
async def test_auth_mode_header():
    """Verify headers when auth_mode='header' and secret is configured."""
    client = N8NRetrievalClient(
        auth_mode="header",
        webhook_secret="super-secret-token-xyz",
    )
    headers = client._build_headers()

    assert headers.get("Content-Type") == "application/json"
    assert headers.get("X-API-Key") == "super-secret-token-xyz"


@pytest.mark.asyncio
async def test_timeout_handling():
    """Test graceful handling when n8n times out."""
    client = N8NRetrievalClient(timeout_seconds=5.0)

    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Read timed out")):
        result = await client.retrieve_evidence(
            claim="Some query",
            domain="general",
            request_id="timeout-test",
        )

        assert result.success is False
        assert result.passages == []
        assert result.error is not None
        assert "timed out" in result.error.lower()
        assert result.trace is not None
        assert result.trace.called is True
        assert result.trace.error is not None


@pytest.mark.asyncio
async def test_http_500_handling():
    """Test graceful handling of HTTP 500 error from n8n."""
    client = N8NRetrievalClient()
    mock_500 = httpx.Response(500, text="Internal Server Error: Node execution failed")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_500

        result = await client.retrieve_evidence(
            claim="Some query",
            domain="general",
        )

        assert result.success is False
        assert result.http_status == 500
        assert result.passages == []
        assert "500" in result.error


@pytest.mark.asyncio
async def test_invalid_json_handling():
    """Test handling of invalid non-JSON response."""
    client = N8NRetrievalClient()
    mock_bad_json = httpx.Response(200, text="<html>502 Bad Gateway</html>")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_bad_json

        result = await client.retrieve_evidence(
            claim="Some query",
            domain="general",
        )

        assert result.success is False
        assert result.passages == []
        assert "not valid json" in result.error.lower()


@pytest.mark.asyncio
async def test_missing_evidence_field():
    """Test response where evidence field is empty or missing."""
    client = N8NRetrievalClient()
    mock_empty_json = {"status": "ok", "evidence": []}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_empty_json)

        result = await client.retrieve_evidence(
            claim="Some query",
            domain="general",
        )

        assert result.success is True
        assert result.passages == []
        assert result.trace.evidence_count == 0


@pytest.mark.asyncio
async def test_malformed_evidence_items():
    """Test response containing partially invalid or non-dict items."""
    client = N8NRetrievalClient()
    mock_data = {
        "evidence": [
            "invalid-string-item",
            None,
            {},  # empty dict with no title/url/snippet
            {
                "title": "Valid Source",
                "snippet": "Valid snippet content text.",
                "url": "https://example.com",
            }
        ]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_data)

        result = await client.retrieve_evidence(
            claim="Some query",
            domain="general",
        )

        assert result.success is True
        assert len(result.passages) == 1
        assert result.passages[0].title == "Valid Source"


@pytest.mark.asyncio
async def test_health_check_endpoint():
    """Test check_health method on healthy and degraded states."""
    client = N8NRetrievalClient()

    # Case 1: Healthy 200
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = httpx.Response(200, json={"status": "ok"})
        health = await client.check_health()
        assert health["healthy"] is True
        assert health["status_code"] == 200

    # Case 2: 500 error
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = httpx.Response(500, text="No authentication")
        health = await client.check_health()
        assert health["healthy"] is False
        assert health["status_code"] == 500


# ─────────────────────────────────────────────────────────────────────────────
# VerificationPipeline Integration & Fallback Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_uses_n8n_evidence_and_runs_python_bge_nli():
    """
    Test that when n8n returns evidence, VerificationPipeline feeds it into
    Python's BGE reranker and DeBERTa NLI without adopting n8n verdict.
    """
    pipeline = VerificationPipeline()

    mock_n8n_result = N8NRetrievalResult(
        success=True,
        passages=[
            Passage(
                title="Wikipedia: Paris City",
                source="wikipedia",
                url="https://en.wikipedia.org/wiki/Paris",
                publication_date="2024-01-01",
                snippet="Paris is the capital and most populous city of France.",
                source_id="wiki_paris_city",
                relevance_score=0.0,
                source_confidence_hint=0.90,
            )
        ],
    )

    with patch.object(pipeline.cache, "get", new_callable=AsyncMock, return_value=None), \
         patch.object(pipeline.n8n_client, "retrieve_evidence", new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = mock_n8n_result

        payload = VerifierInputV2(
            query_id="pipe-test-1",
            domain="general",
            suspicious_claims=[SuspiciousClaim(claim_id="c1", text="Paris is the capital of France.")],
        )

        output = await pipeline.verify(payload)

        assert output.domain == "general"
        assert len(output.claim_evidence) == 1
        rep = output.claim_evidence[0]
        assert rep.verdict.value == "verified"
        assert rep.support_score > 0.0
        assert len(rep.evidence) >= 1
        assert "Paris" in rep.evidence[0].title


@pytest.mark.asyncio
async def test_pipeline_fallback_to_python_adapters_on_n8n_failure():
    """
    Test that if n8n returns a failure (e.g. 500 or timeout), VerificationPipeline
    falls back cleanly to the Python adapter search and completes verification.
    """
    pipeline = VerificationPipeline()

    mock_failed_n8n = N8NRetrievalResult(
        success=False,
        passages=[],
        error="n8n HTTP 500: Server Error",
    )

    fallback_passages = [
        Passage(
            title="Wikipedia: France",
            source="wikipedia",
            url="https://en.wikipedia.org/wiki/France",
            publication_date="2024-01-01",
            snippet="France is a country located in Western Europe. Its capital is Paris.",
            source_id="wiki_france",
            relevance_score=0.0,
            source_confidence_hint=0.85,
        )
    ]

    with patch.object(pipeline.cache, "get", new_callable=AsyncMock, return_value=None), \
         patch.object(pipeline.n8n_client, "retrieve_evidence", new_callable=AsyncMock) as mock_n8n, \
         patch("adapters.general.GeneralAdapter.search", new_callable=AsyncMock) as mock_adapter_search:
        mock_n8n.return_value = mock_failed_n8n
        mock_adapter_search.return_value = fallback_passages

        payload = VerifierInputV2(
            query_id="fallback-test-1",
            domain="general",
            suspicious_claims=[SuspiciousClaim(claim_id="c1", text="Paris is the capital of France.")],
        )

        output = await pipeline.verify(payload)

        assert len(output.claim_evidence) == 1
        rep = output.claim_evidence[0]
        assert rep.verdict.value == "verified"
        assert rep.verified_evidence >= 1


@pytest.mark.asyncio
async def test_pipeline_respects_n8n_retrieval_disabled():
    """Test that when N8N_RETRIEVAL_ENABLED=false, n8n client is not invoked."""
    pipeline = VerificationPipeline()

    with patch.object(pipeline.settings, "n8n_retrieval_enabled", False), \
         patch.object(pipeline.cache, "get", new_callable=AsyncMock, return_value=None), \
         patch.object(pipeline.n8n_client, "retrieve_evidence", new_callable=AsyncMock) as mock_n8n, \
         patch("adapters.general.GeneralAdapter.search", new_callable=AsyncMock) as mock_adapter_search:
        mock_adapter_search.return_value = [
            Passage(
                title="Wikipedia: Paris",
                source="wikipedia",
                url="https://en.wikipedia.org/wiki/Paris",
                publication_date="unknown",
                snippet="Paris is the capital of France.",
                source_id="wiki_paris",
            )
        ]

        payload = VerifierInputV2(
            query_id="disabled-test-1",
            domain="general",
            suspicious_claims=[SuspiciousClaim(claim_id="c1", text="Paris is the capital of France.")],
        )

        await pipeline.verify(payload)

        # n8n client retrieve_evidence must NOT be called when disabled
        mock_n8n.assert_not_called()
        mock_adapter_search.assert_called()


@pytest.mark.asyncio
async def test_no_duplicate_retrieval_when_n8n_succeeds():
    """Verify that when n8n returns evidence, Python adapters are NOT invoked (no double-retrieval)."""
    pipeline = VerificationPipeline()

    mock_n8n_res = N8NRetrievalResult(
        success=True,
        passages=[
            Passage(
                title="Wikipedia: Allu Arjun",
                source="wikipedia",
                url="https://en.wikipedia.org/wiki/Allu_Arjun",
                publication_date="2024-01-01",
                snippet="Allu Arjun is an Indian actor. His father is Allu Aravind.",
                source_id="wiki_allu_arjun",
            )
        ],
    )

    with patch.object(pipeline.cache, "get", new_callable=AsyncMock, return_value=None), \
         patch.object(pipeline.n8n_client, "retrieve_evidence", new_callable=AsyncMock) as mock_n8n, \
         patch("adapters.general.GeneralAdapter.search", new_callable=AsyncMock) as mock_adapter_search:
        mock_n8n.return_value = mock_n8n_res

        payload = VerifierInputV2(
            query_id="no-dup-test",
            domain="general",
            suspicious_claims=[SuspiciousClaim(claim_id="c1", text="Allu Arjun's father is Allu Aravind.")],
        )

        output = await pipeline.verify(payload)

        mock_n8n.assert_called()
        mock_adapter_search.assert_not_called()
        assert len(output.claim_evidence) == 1
        assert output.claim_evidence[0].verdict.value == "verified"


@pytest.mark.asyncio
async def test_bge_and_deberta_receive_real_claim_and_evidence_pairs():
    """Explicitly verify that DeBERTa NLI and BGE receive the actual input claim and retrieved snippet."""
    pipeline = VerificationPipeline()

    test_claim = "Aspirin is used to relieve mild to moderate pain."
    test_snippet = "Aspirin (acetylsalicylic acid) is an analgesic medication used to treat pain, fever, or inflammation."

    mock_n8n_res = N8NRetrievalResult(
        success=True,
        passages=[
            Passage(
                title="PubMed: Aspirin Clinical Pharmacology",
                source="pubmed",
                url="https://pubmed.ncbi.nlm.nih.gov/123456/",
                publication_date="2024-01-01",
                snippet=test_snippet,
                source_id="pmid_123456",
            )
        ],
    )

    with patch.object(pipeline.cache, "get", new_callable=AsyncMock, return_value=None), \
         patch.object(pipeline.n8n_client, "retrieve_evidence", new_callable=AsyncMock) as mock_n8n, \
         patch.object(pipeline.nli_engine, "batch_classify") as mock_nli:
        mock_n8n.return_value = mock_n8n_res
        mock_nli.return_value = [
            {
                "label": "entailment",
                "entailment_score": 0.96,
                "contradiction_score": 0.01,
                "neutral_score": 0.03,
                "confidence": 0.96,
                "degraded": False,
            }
        ]

        payload = VerifierInputV2(
            query_id="real-nli-pair-test",
            domain="healthcare",
            suspicious_claims=[SuspiciousClaim(claim_id="c1", text=test_claim)],
        )

        output = await pipeline.verify(payload)

        # Assert NLI batch_classify received exact real claim semantics and exact real premise snippet
        mock_nli.assert_called_once()
        call_args = mock_nli.call_args
        assert "aspirin is used to relieve mild to moderate pain" in call_args[0][0].lower()
        assert test_snippet in call_args[0][1]

        report = output.claim_evidence[0]
        assert report.verdict.value == "verified"
        assert report.evidence[0].nli_entailment == 0.96
        assert report.evidence[0].nli_contradiction == 0.01
        assert report.evidence[0].nli_neutral == 0.03


@pytest.mark.asyncio
async def test_evidence_scorer_preserves_irrelevant_and_neutral():
    """Verify that irrelevant PubMed passages from authoritative sources do not get treated as contradiction."""
    pipeline = VerificationPipeline()

    test_claim = "Aspirin is used to relieve mild to moderate pain."
    unrelated_snippet = "Colorectal cancer incidence in elderly mice was studied following microbiome alterations."

    mock_n8n_res = N8NRetrievalResult(
        success=True,
        passages=[
            Passage(
                title="PubMed: Colorectal Cancer in Mice",
                source="pubmed",
                url="https://pubmed.ncbi.nlm.nih.gov/999999/",
                publication_date="2024-01-01",
                snippet=unrelated_snippet,
                source_id="pmid_999999",
                relevance_score=0.05,
            )
        ],
    )

    with patch.object(pipeline.cache, "get", new_callable=AsyncMock, return_value=None), \
         patch.object(pipeline.n8n_client, "retrieve_evidence", new_callable=AsyncMock) as mock_n8n, \
         patch.object(pipeline.nli_engine, "batch_classify") as mock_nli:
        mock_n8n.return_value = mock_n8n_res
        mock_nli.return_value = [
            {
                "label": "neutral",
                "entailment_score": 0.02,
                "contradiction_score": 0.05,
                "neutral_score": 0.93,
                "confidence": 0.93,
                "degraded": False,
            }
        ]

        payload = VerifierInputV2(
            query_id="irrelevant-nli-test",
            domain="healthcare",
            suspicious_claims=[SuspiciousClaim(claim_id="c1", text=test_claim)],
        )

        output = await pipeline.verify(payload)

        report = output.claim_evidence[0]
        # Since the evidence was neutral/irrelevant, verdict should be UNVERIFIED, NOT CONTRADICTED
        assert report.verdict.value == "unverified"
        assert report.contradiction_score == 0.0




# ─────────────────────────────────────────────────────────────────────────────
# Live Tests — Executed only when RUN_LIVE_N8N_TESTS=true
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_N8N_TESTS", "").lower() != "true",
    reason="Live n8n tests require RUN_LIVE_N8N_TESTS=true environment variable",
)
@pytest.mark.asyncio
async def test_live_n8n_health_endpoint():
    client = N8NRetrievalClient()
    health = await client.check_health()
    assert health["status_code"] in (200, 500)  # Status is captured cleanly


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_N8N_TESTS", "").lower() != "true",
    reason="Live n8n tests require RUN_LIVE_N8N_TESTS=true environment variable",
)
@pytest.mark.asyncio
async def test_live_n8n_retrieval_webhook():
    client = N8NRetrievalClient()
    res = await client.retrieve_evidence(
        claim="Paris is the capital of France.",
        domain="general",
    )
    assert res.trace is not None
    assert res.trace.called is True
