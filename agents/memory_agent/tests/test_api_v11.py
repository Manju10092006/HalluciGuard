import pytest
from httpx import AsyncClient, ASGITransport

from agents.memory_agent.api.main import app
from agents.memory_agent.config.settings import Settings
from agents.memory_agent.knowledge_graph.graph import KnowledgeGraph
from agents.memory_agent.cache.verification_cache import VerificationCache
from agents.memory_agent.patterns.pattern_learner import PatternLearner
from agents.memory_agent.trust.source_trust import SourceTrustManager
from agents.memory_agent.vector_store.faiss_store import VectorStore
from agents.memory_agent.memory.memory_agent import MemoryAgent


@pytest.fixture
async def client(tmp_path):
    settings = Settings(
        kg_persistence_path=str(tmp_path / "kg.json"),
        cache_db_path=str(tmp_path / "cache.db"),
        pattern_db_path=str(tmp_path / "patterns.db"),
        trust_db_path=str(tmp_path / "trust.db"),
        vector_store_path=str(tmp_path / "vectors"),
    )
    kg = KnowledgeGraph(persistence_path=settings.kg_persistence_path)
    cache = VerificationCache(db_path=settings.cache_db_path, ttl=3600)
    await cache.initialize()
    patterns = PatternLearner(db_path=settings.pattern_db_path)
    await patterns.initialize()
    trust = SourceTrustManager(db_path=settings.trust_db_path)
    await trust.initialize()
    vectors = VectorStore(
        store_path=settings.vector_store_path,
        embedding_model="all-MiniLM-L6-v2",
    )
    vectors.initialize()

    agent = MemoryAgent(
        settings=settings,
        knowledge_graph=kg,
        cache=cache,
        pattern_learner=patterns,
        source_trust=trust,
        vector_store=vectors,
    )
    await agent.initialize()
    app.state.memory_agent = agent

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    await agent.close()


class TestBatchEndpoint:
    @pytest.mark.asyncio
    async def test_store_batch(self, client):
        payload = [
            {
                "claim_text": "Batch API one",
                "domain": "test",
                "verdict": "verified",
                "confidence": 0.9,
            },
            {
                "claim_text": "Batch API two",
                "domain": "test",
                "verdict": "likely_hallucinated",
                "confidence": 0.1,
            },
        ]
        resp = await client.post("/store/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["stored"] == 2


class TestAnalyticsEndpoint:
    @pytest.mark.asyncio
    async def test_kg_analytics(self, client):
        await client.post("/store", json={
            "claim_text": "Analytics claim",
            "domain": "test",
            "verdict": "verified",
            "evidence": [{"source_id": "s1"}],
            "source_ids": ["s1"],
            "confidence": 0.9,
        })
        resp = await client.get("/knowledge-graph/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert "most_important" in data
        assert "connected_components" in data


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        assert "memory_store_total" in resp.text


class TestDuplicateEndpoint:
    @pytest.mark.asyncio
    async def test_duplicate_store_via_api(self, client):
        payload = {
            "claim_text": "Duplicate API claim",
            "domain": "test",
            "verdict": "verified",
            "confidence": 0.9,
        }
        r1 = await client.post("/store", json=payload)
        r2 = await client.post("/store", json=payload)
        assert r1.status_code == 200
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["stored"] is False
        assert d2["duplicate_of"] == r1.json()["fact_id"]


class TestApiKeyAuth:
    @pytest.mark.asyncio
    async def test_requires_api_key_when_configured(self, client, monkeypatch):
        from agents.memory_agent import api as api_module

        monkeypatch.setattr(api_module.main.settings, "memory_agent_api_key", "secret-key")

        resp = await client.post("/store", json={
            "claim_text": "Unauthorized claim",
            "domain": "test",
            "verdict": "verified",
            "confidence": 0.9,
        })
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_allows_valid_api_key(self, client, monkeypatch):
        from agents.memory_agent import api as api_module

        monkeypatch.setattr(api_module.main.settings, "memory_agent_api_key", "secret-key")

        resp = await client.post(
            "/store",
            json={
                "claim_text": "Authorized claim",
                "domain": "test",
                "verdict": "verified",
                "confidence": 0.9,
            },
            headers={"X-API-Key": "secret-key"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_requires_auth_when_configured(self, client, monkeypatch):
        from agents.memory_agent import api as api_module

        monkeypatch.setattr(api_module.main.settings, "memory_agent_api_key", "secret-key")

        resp = await client.get("/metrics")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_no_key_configured_allows_access(self, client):
        resp = await client.post("/store", json={
            "claim_text": "Open claim",
            "domain": "test",
            "verdict": "verified",
            "confidence": 0.9,
        })
        assert resp.status_code == 200
