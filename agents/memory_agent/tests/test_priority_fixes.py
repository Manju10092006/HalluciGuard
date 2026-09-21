import pytest

from agents.memory_agent.config.settings import Settings
from agents.memory_agent.contradiction.detector import ContradictionDetector
from agents.memory_agent.knowledge_graph.graph import KnowledgeGraph
from agents.memory_agent.cache.verification_cache import VerificationCache
from agents.memory_agent.patterns.pattern_learner import PatternLearner
from agents.memory_agent.trust.source_trust import SourceTrustManager
from agents.memory_agent.vector_store.faiss_store import VectorStore
from agents.memory_agent.memory.memory_agent import MemoryAgent
from agents.memory_agent.schemas.models import StoreFactRequest


@pytest.fixture
async def agent(tmp_path):
    settings = Settings(
        kg_persistence_path=str(tmp_path / "kg.json"),
        cache_db_path=str(tmp_path / "cache.db"),
        pattern_db_path=str(tmp_path / "patterns.db"),
        trust_db_path=str(tmp_path / "trust.db"),
        vector_store_path=str(tmp_path / "vectors"),
        storage_journal_path=str(tmp_path / "journal.db"),
        mock_mode=True,
    )
    kg = KnowledgeGraph(persistence_path=settings.kg_persistence_path)
    cache = VerificationCache(db_path=settings.cache_db_path, ttl=3600)
    await cache.initialize()
    patterns = PatternLearner(
        db_path=settings.pattern_db_path,
        min_support=settings.pattern_min_support,
        confidence_threshold=settings.pattern_confidence_threshold,
    )
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


class TestContradictionDetectorStructured:
    """Stage-2 must confirm contradictions instead of trusting similarity."""

    def setup_method(self):
        self.detector = ContradictionDetector(use_nli=False)

    def test_negation_is_contradiction(self):
        confirmed, method = self.detector.confirm(
            "Vitamin C cures the common cold",
            "Vitamin C does not cure the common cold",
        )
        assert confirmed is True
        assert method == "structured"

    def test_conflicting_temporal_values_is_contradiction(self):
        confirmed, _ = self.detector.confirm(
            "The company was founded in 1998",
            "The company was founded in 1999",
        )
        assert confirmed is True

    def test_operated_since_is_not_contradiction(self):
        # High lexical/semantic overlap but logically compatible.
        confirmed, _ = self.detector.confirm(
            "The company was founded in 1998",
            "The company has operated since 1998",
        )
        assert confirmed is False

    def test_subsumption_is_not_contradiction(self):
        confirmed, _ = self.detector.confirm(
            "Python was created by Guido",
            "Python was created by Guido in 1991",
        )
        assert confirmed is False

    def test_negation_of_unrelated_predicate_not_contradiction(self):
        # Both mention smoking & lung cancer, but B negates "cure" which A
        # never asserts; these are logically compatible.
        confirmed, _ = self.detector.confirm(
            "Smoking increases risk of lung cancer",
            "Smoking does not cure lung cancer",
        )
        assert confirmed is False


class TestStage2ContradictionViaAgent:
    @pytest.mark.asyncio
    async def test_confirmed_contradiction_carries_method(self, agent):
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
        assert resp.contradictions[0].confirmation_method == "structured"

    @pytest.mark.asyncio
    async def test_high_similarity_without_logic_contradiction_not_flagged(
        self, agent
    ):
        # Verdicts conflict on semantically similar claims, but "founded in
        # 2001" is compatible with "has operated since 2001" — stage-2 must
        # NOT raise a contradiction purely from similarity.
        confirmed, method, detail = await agent._confirm_contradiction(
            existing_text="The company has operated since 2001",
            new_text="The company was founded in 2001",
        )
        assert confirmed is False
        assert method is None
        assert detail == "unconfirmed"

    @pytest.mark.asyncio
    async def test_stage2_confirms_real_contradiction(self, agent):
        confirmed, method, _ = await agent._confirm_contradiction(
            existing_text="The company was founded in 1998",
            new_text="The company was founded in 1999",
        )
        assert confirmed is True
        assert method == "structured"


class TestStorageJournal:
    @pytest.mark.asyncio
    async def test_store_records_subsystem_outcomes(self, agent):
        req = StoreFactRequest(
            claim_text="Journal test claim",
            domain="test",
            verdict="verified",
            confidence=0.9,
        )
        resp = await agent.store_fact(req)
        assert resp.stored is True

        stats = await agent.get_storage_journal_stats()
        by_subsystem = stats["by_subsystem"]
        assert by_subsystem["knowledge_graph"].get("done", 0) >= 1
        assert by_subsystem["vector_store"].get("done", 0) >= 1
        assert by_subsystem["cache"].get("done", 0) >= 1

    @pytest.mark.asyncio
    async def test_reconcile_rebuilds_derived_vector_index(self, agent):
        op_id = await agent.journal.record(
            op_type="store",
            fact_id="sim-fact-1",
            subsystem="vector_store",
            status="failed",
            payload={
                "claim_text": "Reconcile me",
                "metadata": {
                    "domain": "test",
                    "verdict": "verified",
                    "confidence": 0.9,
                    "fact_id": "sim-fact-1",
                },
            },
            claim_text="Reconcile me",
            domain="test",
            error="simulated failure",
        )
        assert op_id

        result = await agent.reconcile()
        assert result["reconciled"] >= 1

        entry = agent.vectors.get("sim-fact-1")
        assert entry is not None
        assert entry.text == "Reconcile me"

    @pytest.mark.asyncio
    async def test_reconcile_repairs_cache_from_journal_payload(self, agent):
        await agent.journal.record(
            op_type="store",
            fact_id="sim-fact-2",
            subsystem="cache",
            status="failed",
            payload={
                "domain": "test",
                "claim_text": "Cached reconcile claim",
                "verdict": "verified",
                "confidence": 0.9,
                "evidence_summary": "recovered",
                "source_count": 1,
            },
            claim_text="Cached reconcile claim",
            domain="test",
            error="simulated failure",
        )
        result = await agent.reconcile()
        assert result["reconciled"] >= 1

        cached = await agent.cache.get("test", "Cached reconcile claim")
        assert cached is not None
        assert cached.verdict == "verified"

    @pytest.mark.asyncio
    async def test_gated_store_is_not_journaled_as_fact_write(self, agent):
        req = StoreFactRequest(
            claim_text="Nothing to journal",
            domain="test",
            verdict="likely_hallucinated",
            confidence=0.05,
        )
        resp = await agent.store_fact(req)
        assert resp.stored is False

        stats = await agent.get_storage_journal_stats()
        assert stats["total_operations"] == 0