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


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"]


class TestDomainsEndpoint:
    @pytest.mark.asyncio
    async def test_list_domains(self, client):
        resp = await client.get("/domains")
        assert resp.status_code == 200
        data = resp.json()
        assert "domains" in data
        assert "healthcare" in data["domains"]


class TestStoreEndpoint:
    @pytest.mark.asyncio
    async def test_store_fact(self, client):
        payload = {
            "claim_text": "Test API claim",
            "domain": "test",
            "verdict": "verified",
            "confidence": 0.9,
        }
        resp = await client.post("/store", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "fact_id" in data
        assert data["entities_created"] >= 2

    @pytest.mark.asyncio
    async def test_store_with_evidence(self, client):
        payload = {
            "claim_text": "API claim with evidence",
            "domain": "healthcare",
            "verdict": "likely_hallucinated",
            "evidence": [{"source_id": "pubmed", "title": "Study"}],
            "source_ids": ["pubmed"],
            "confidence": 0.2,
        }
        resp = await client.post("/store", json=payload)
        assert resp.status_code == 200


class TestRecallEndpoint:
    @pytest.mark.asyncio
    async def test_recall(self, client):
        await client.post("/store", json={
            "claim_text": "Recall test claim",
            "domain": "test",
            "verdict": "verified",
            "confidence": 0.85,
        })
        payload = {
            "query": "Recall test claim",
            "domain": "test",
            "top_k": 5,
        }
        resp = await client.post("/recall", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "similar_facts" in data
        assert "cached_verification" in data


class TestCacheEndpoints:
    @pytest.mark.asyncio
    async def test_check_cache_miss(self, client):
        resp = await client.get("/cache/check", params={
            "domain": "test", "claim_text": "nonexistent"
        })
        assert resp.status_code == 200
        assert resp.json()["cached"] is False

    @pytest.mark.asyncio
    async def test_check_cache_hit(self, client):
        await client.post("/store", json={
            "claim_text": "Cache hit test",
            "domain": "test",
            "verdict": "verified",
            "confidence": 0.9,
        })
        resp = await client.get("/cache/check", params={
            "domain": "test", "claim_text": "Cache hit test"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["cached"] is True

    @pytest.mark.asyncio
    async def test_invalidate_cache(self, client):
        resp = await client.delete("/cache/invalidate", params={
            "domain": "test", "claim_text": "something"
        })
        assert resp.status_code == 200


class TestTrustEndpoints:
    @pytest.mark.asyncio
    async def test_update_trust(self, client):
        payload = {
            "source_id": "test_source",
            "source_name": "Test Source",
            "domain": "test",
            "reason": "verified_correct",
        }
        resp = await client.post("/trust/update", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_id"] == "test_source"

    @pytest.mark.asyncio
    async def test_get_trust(self, client):
        await client.post("/trust/update", json={
            "source_id": "s1", "source_name": "S1", "domain": "d1",
            "reason": "verified_correct",
        })
        resp = await client.get("/trust/s1")
        assert resp.status_code == 200
        assert resp.json()["source_name"] == "S1"

    @pytest.mark.asyncio
    async def test_get_trust_not_found(self, client):
        resp = await client.get("/trust/nonexistent")
        assert resp.status_code == 404


class TestPatternEndpoints:
    @pytest.mark.asyncio
    async def test_query_patterns(self, client):
        resp = await client.post("/patterns/query", json={})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_domain_patterns(self, client):
        resp = await client.get("/patterns/domain/healthcare")
        assert resp.status_code == 200


class TestKgEndpoints:
    @pytest.mark.asyncio
    async def test_kg_stats(self, client):
        resp = await client.get("/knowledge-graph/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_nodes" in data

    @pytest.mark.asyncio
    async def test_entity_not_found(self, client):
        resp = await client.get("/knowledge-graph/entity/nonexistent")
        assert resp.status_code == 404


class TestVectorEndpoints:
    @pytest.mark.asyncio
    async def test_vector_search(self, client):
        resp = await client.get("/vectors/search", params={"q": "test query"})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestStatsEndpoint:
    @pytest.mark.asyncio
    async def test_stats(self, client):
        resp = await client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "knowledge_graph" in data
        assert "cache" in data
