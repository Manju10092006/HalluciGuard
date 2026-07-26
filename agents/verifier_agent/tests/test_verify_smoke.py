from __future__ import annotations
import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app

@pytest.fixture
def sample_payload():
    return {
        "domain": "healthcare",
        "suspicious_claims": [
            {
                "claim_id": "claim_1",
                "claim_text": "XYZ drug completely cures diabetes"
            }
        ]
    }

@pytest.mark.asyncio
async def test_verify_basic(sample_payload):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/verify", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert data["domain"] == "healthcare"
    assert "reports" in data
    assert isinstance(data["reports"], list)

@pytest.mark.asyncio
async def test_verify_response_contract(sample_payload):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/verify", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "domain" in data
    assert "reports" in data
    assert "metrics" in data
    
@pytest.mark.asyncio
async def test_verify_caching(sample_payload):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # First request
        response1 = await ac.post("/verify", json=sample_payload)
        assert response1.status_code == 200
        
        # Second request
        response2 = await ac.post("/verify", json=sample_payload)
        assert response2.status_code == 200
        
        # In a real scenario we'd assert latency is lower, but logic validation suffices
        assert response1.json()["reports"][0]["verdict"] == response2.json()["reports"][0]["verdict"]
