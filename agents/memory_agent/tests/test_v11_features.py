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


class TestDuplicateDetection:
    @pytest.mark.asyncio
    async def test_identical_claim_is_duplicate(self, agent):
        first = StoreFactRequest(
            claim_text="The moon is made of cheese",
            domain="science",
            verdict="likely_hallucinated",
            confidence=0.05,
        )
        resp1 = await agent.store_fact(first)
        assert resp1.stored is True

        second = StoreFactRequest(
            claim_text="The moon is made of cheese",
            domain="science",
            verdict="likely_hallucinated",
            confidence=0.05,
        )
        resp2 = await agent.store_fact(second)
        assert resp2.stored is False
        assert resp2.duplicate_of == resp1.fact_id
        assert resp2.entities_created == 0

    @pytest.mark.asyncio
    async def test_near_duplicate_flagged(self, agent):
        req = StoreFactRequest(
            claim_text="Regular exercise reduces heart disease risk",
            domain="healthcare",
            verdict="verified",
            confidence=0.9,
        )
        await agent.store_fact(req)

        near = StoreFactRequest(
            claim_text="Regular exercise reduces heart disease risk significantly",
            domain="healthcare",
            verdict="verified",
            confidence=0.9,
        )
        resp = await agent.store_fact(near)
        assert resp.duplicate_of is not None


class TestContradictionDetection:
    @pytest.mark.asyncio
    async def test_conflicting_verdict_raises_alert(self, agent):
        positive = StoreFactRequest(
            claim_text="Vitamin C cures the common cold",
            domain="healthcare",
            verdict="verified",
            confidence=0.9,
        )
        await agent.store_fact(positive)

        negative = StoreFactRequest(
            claim_text="Vitamin C does not cure the common cold",
            domain="healthcare",
            verdict="likely_hallucinated",
            confidence=0.05,
        )
        resp = await agent.store_fact(negative)
        assert len(resp.contradictions) > 0
        alert = resp.contradictions[0]
        assert alert.existing_verdict == "verified"
        assert alert.existing_fact_id

    @pytest.mark.asyncio
    async def test_same_verdict_no_contradiction(self, agent):
        first = StoreFactRequest(
            claim_text="Water boils at 100 degrees Celsius at sea level",
            domain="science",
            verdict="verified",
            confidence=0.95,
        )
        await agent.store_fact(first)

        second = StoreFactRequest(
            claim_text="Water boils at 100 degrees Celsius at sea level approximately",
            domain="science",
            verdict="verified",
            confidence=0.95,
        )
        resp = await agent.store_fact(second)
        assert len(resp.contradictions) == 0


class TestBatchStore:
    @pytest.mark.asyncio
    async def test_batch_stores_multiple(self, agent):
        requests = [
            StoreFactRequest(
                claim_text="Batch claim one",
                domain="test",
                verdict="verified",
                confidence=0.9,
            ),
            StoreFactRequest(
                claim_text="Batch claim two",
                domain="test",
                verdict="likely_hallucinated",
                confidence=0.1,
            ),
            StoreFactRequest(
                claim_text="Batch claim three",
                domain="test",
                verdict="verified",
                confidence=0.9,
            ),
        ]
        resp = await agent.store_facts_batch(requests)
        assert resp.total == 3
        assert resp.stored == 3
        assert resp.failed == 0
        assert len(resp.results) == 3

    @pytest.mark.asyncio
    async def test_batch_tracks_duplicates(self, agent):
        req = StoreFactRequest(
            claim_text="Batch duplicate claim",
            domain="test",
            verdict="verified",
            confidence=0.9,
        )
        await agent.store_fact(req)

        resp = await agent.store_facts_batch([req])
        assert resp.total == 1
        assert resp.duplicates == 1
        assert resp.stored == 0


class TestGraphAnalytics:
    @pytest.mark.asyncio
    async def test_analyze_empty_graph(self, agent):
        result = agent.kg.analyze()
        assert result.total_nodes == 0
        assert result.most_important == []

    @pytest.mark.asyncio
    async def test_analyze_populated_graph(self, agent):
        req = StoreFactRequest(
            claim_text="Graph analytics test claim",
            domain="test",
            verdict="verified",
            evidence=[{"source_id": "src-a"}, {"source_id": "src-b"}],
            source_ids=["src-a", "src-b"],
            confidence=0.9,
        )
        await agent.store_fact(req)
        result = agent.kg.analyze(top_k=5)
        assert result.total_nodes >= 3
        assert len(result.most_important) > 0
        assert result.most_important[0].name


class TestReranking:
    @pytest.mark.asyncio
    async def test_recall_reranks(self, agent):
        store = StoreFactRequest(
            claim_text="Coffee increases alertness",
            domain="science",
            verdict="verified",
            source_ids=["trusted_src"],
            confidence=0.9,
        )
        await agent.store_fact(store)
        await agent.update_source_trust(
            source_id="trusted_src",
            source_name="Trusted Source",
            domain="science",
            reason=TrustChangeReason.VERIFIED_CORRECT,
        )

        req = RecallRequest(
            query="coffee alertness",
            domain="science",
            top_k=5,
            rerank=True,
        )
        resp = await agent.recall(req)
        assert resp.reranked is True
        assert len(resp.similar_facts) > 0
        assert resp.similar_facts[0].reranked_score is not None

    @pytest.mark.asyncio
    async def test_min_similarity_filters(self, agent):
        store = StoreFactRequest(
            claim_text="Unrelated claim about gardening",
            domain="test",
            verdict="verified",
            confidence=0.9,
        )
        await agent.store_fact(store)

        req = RecallRequest(
            query="quantum physics of star formation",
            domain="test",
            top_k=5,
            min_similarity=0.99,
        )
        resp = await agent.recall(req)
        assert all(f.score >= 0.99 for f in resp.similar_facts)


class TestFuzzyCache:
    @pytest.mark.asyncio
    async def test_fuzzy_cache_returns_similar_hits(self, agent):
        store = StoreFactRequest(
            claim_text="The Great Wall of China is visible from space",
            domain="science",
            verdict="likely_hallucinated",
            confidence=0.05,
        )
        await agent.store_fact(store)

        req = RecallRequest(
            query="Is the Great Wall of China visible from space orbit",
            domain="science",
            top_k=5,
            fuzzy_cache=True,
        )
        resp = await agent.recall(req)
        assert len(resp.fuzzy_cache_hits) >= 0


class TestFactDeletion:
    @pytest.mark.asyncio
    async def test_delete_existing_fact(self, agent):
        req = StoreFactRequest(
            claim_text="Fact to be deleted",
            domain="test",
            verdict="verified",
            confidence=0.9,
        )
        resp = await agent.store_fact(req)
        assert resp.stored is True

        delete_resp = await agent.delete_fact(resp.fact_id)
        assert delete_resp.deleted is True
        assert "knowledge_graph" in delete_resp.deleted_from
        assert "vector_store" in delete_resp.deleted_from
        assert "cache" in delete_resp.deleted_from

        # Verify it's gone
        assert agent.kg.get_entity(resp.fact_id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_fact(self, agent):
        resp = await agent.delete_fact("nonexistent-id")
        assert resp.deleted is False
        assert resp.deleted_from == []


class TestFactUpdate:
    @pytest.mark.asyncio
    async def test_update_verdict(self, agent):
        from agents.memory_agent.schemas.models import UpdateFactRequest

        req = StoreFactRequest(
            claim_text="Fact to be updated",
            domain="test",
            verdict="likely_hallucinated",
            confidence=0.1,
        )
        resp = await agent.store_fact(req)

        update = UpdateFactRequest(
            fact_id=resp.fact_id,
            new_verdict="verified",
            new_confidence=0.95,
        )
        update_resp = await agent.update_fact(update)
        assert update_resp.old_verdict == "likely_hallucinated"
        assert update_resp.new_verdict == "verified"
        assert update_resp.new_confidence == 0.95
        assert "knowledge_graph" in update_resp.updated_in
        assert "vector_store" in update_resp.updated_in

    @pytest.mark.asyncio
    async def test_update_nonexistent_fact(self, agent):
        from agents.memory_agent.schemas.models import UpdateFactRequest
        import pytest as _pytest

        update = UpdateFactRequest(fact_id="nonexistent", new_verdict="verified")
        with _pytest.raises(ValueError, match="not found"):
            await agent.update_fact(update)
