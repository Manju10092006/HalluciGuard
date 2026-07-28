import pytest

from agents.memory_agent.config.settings import Settings
from agents.memory_agent.knowledge_graph.graph import KnowledgeGraph
from agents.memory_agent.cache.verification_cache import VerificationCache
from agents.memory_agent.patterns.pattern_learner import PatternLearner
from agents.memory_agent.trust.source_trust import SourceTrustManager
from agents.memory_agent.vector_store.faiss_store import VectorStore
from agents.memory_agent.memory.memory_agent import MemoryAgent
from agents.memory_agent.schemas.models import (
    StoreFactRequest,
    RecallRequest,
    TrustChangeReason,
)


@pytest.fixture
async def agent(tmp_path):
    settings = Settings(
        kg_persistence_path=str(tmp_path / "kg.json"),
        cache_db_path=str(tmp_path / "cache.db"),
        pattern_db_path=str(tmp_path / "patterns.db"),
        trust_db_path=str(tmp_path / "trust.db"),
        vector_store_path=str(tmp_path / "vectors"),
        mock_mode=True,
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

    mem = MemoryAgent(
        settings=settings,
        knowledge_graph=kg,
        cache=cache,
        pattern_learner=patterns,
        source_trust=trust,
        vector_store=vectors,
    )
    await mem.initialize()
    yield mem
    await mem.close()


class TestStoreFact:
    @pytest.mark.asyncio
    async def test_store_basic_fact(self, agent):
        req = StoreFactRequest(
            claim_text="Aspirin cures cancer",
            domain="healthcare",
            verdict="likely_hallucinated",
            evidence=[],
            source_ids=[],
            confidence=0.1,
        )
        resp = await agent.store_fact(req)
        assert resp.fact_id
        assert resp.entities_created >= 2
        assert resp.edges_created >= 1

    @pytest.mark.asyncio
    async def test_store_fact_with_evidence(self, agent):
        req = StoreFactRequest(
            claim_text="Vitamin C prevents colds",
            domain="healthcare",
            verdict="verified",
            evidence=[
                {"source_id": "pubmed", "title": "Study on Vit C"},
                {"source_id": "nih", "title": "NIH Report"},
            ],
            source_ids=["pubmed", "nih"],
            confidence=0.85,
        )
        resp = await agent.store_fact(req)
        assert resp.fact_id
        assert resp.entities_created >= 2

    @pytest.mark.asyncio
    async def test_store_updates_cache(self, agent):
        req = StoreFactRequest(
            claim_text="Test claim for cache",
            domain="test",
            verdict="verified",
            confidence=0.9,
        )
        await agent.store_fact(req)
        cached = await agent.check_cache("test", "Test claim for cache")
        assert cached is not None
        assert cached["verdict"] == "verified"

    @pytest.mark.asyncio
    async def test_store_hallucination_updates_patterns(self, agent):
        req = StoreFactRequest(
            claim_text="The earth was invented in 1999",
            domain="science",
            verdict="likely_hallucinated",
            confidence=0.05,
        )
        resp = await agent.store_fact(req)
        assert resp.pattern_updated is True


class TestRecall:
    @pytest.mark.asyncio
    async def test_recall_returns_similar_facts(self, agent):
        store_req = StoreFactRequest(
            claim_text="Aspirin is used for pain relief",
            domain="healthcare",
            verdict="verified",
            confidence=0.95,
        )
        await agent.store_fact(store_req)

        recall_req = RecallRequest(
            query="Aspirin pain relief",
            domain="healthcare",
            top_k=5,
        )
        resp = await agent.recall(recall_req)
        assert len(resp.similar_facts) > 0

    @pytest.mark.asyncio
    async def test_recall_includes_cache(self, agent):
        req = StoreFactRequest(
            claim_text="Cached claim test",
            domain="test",
            verdict="verified",
            confidence=0.8,
        )
        await agent.store_fact(req)

        recall_req = RecallRequest(
            query="Cached claim test",
            domain="test",
            include_cache=True,
        )
        resp = await agent.recall(recall_req)
        assert resp.cached_verification is not None

    @pytest.mark.asyncio
    async def test_recall_includes_patterns(self, agent):
        req = StoreFactRequest(
            claim_text="Aspirin cures cancer completely",
            domain="healthcare",
            verdict="likely_hallucinated",
            confidence=0.05,
        )
        await agent.store_fact(req)

        recall_req = RecallRequest(
            query="Aspirin cures cancer",
            domain="healthcare",
            include_patterns=True,
        )
        resp = await agent.recall(recall_req)
        assert len(resp.relevant_patterns) > 0


class TestStats:
    @pytest.mark.asyncio
    async def test_get_stats(self, agent):
        stats = await agent.get_stats()
        assert stats.knowledge_graph.total_nodes >= 0
        assert stats.vector_store_size >= 0
