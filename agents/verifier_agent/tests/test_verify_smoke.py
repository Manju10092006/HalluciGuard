from __future__ import annotations
import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app

@pytest.fixture
def sample_payload():
    return {
        "query_id": "test_query_001",
        "domain": "healthcare",
        "suspicious_claims": [
            {
                "claim_id": "claim_1",
                "text": "XYZ drug completely cures diabetes"
            }
        ]
    }

@pytest.mark.anyio
async def test_verify_basic(sample_payload):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/verify", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "query_id" in data
    assert data["domain"] == "healthcare"
    assert "claim_evidence" in data
    assert isinstance(data["claim_evidence"], list)

@pytest.mark.anyio
async def test_verify_response_contract(sample_payload):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/verify", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "query_id" in data
    assert "domain" in data
    assert "domain_validated" in data
    assert "claim_evidence" in data
    assert "overall_evidence_confidence" in data
    assert "latency_ms" in data
    assert "pipeline_stages" in data

@pytest.mark.anyio
async def test_verify_caching(sample_payload):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # First request
        response1 = await ac.post("/verify", json=sample_payload)
        assert response1.status_code == 200
        
        # Second request
        response2 = await ac.post("/verify", json=sample_payload)
        assert response2.status_code == 200
        
        assert response1.json()["claim_evidence"][0]["verdict"] == response2.json()["claim_evidence"][0]["verdict"]
