import pytest

from agents.memory_agent.cache.verification_cache import VerificationCache


@pytest.fixture
async def cache(tmp_path):
    c = VerificationCache(db_path=str(tmp_path / "test_cache.db"), ttl=3600)
    await c.initialize()
    yield c
    await c.close()


class TestCacheOperations:
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        await cache.set(
            domain="healthcare",
            claim_text="Aspirin cures cancer",
            verdict="likely_hallucinated",
            evidence_summary="No evidence found",
            confidence=0.1,
            source_count=0,
        )
        result = await cache.get("healthcare", "Aspirin cures cancer")
        assert result is not None
        assert result.verdict == "likely_hallucinated"
        assert result.domain == "healthcare"

    @pytest.mark.asyncio
    async def test_get_miss(self, cache):
        result = await cache.get("finance", "nonexistent claim")
        assert result is None

    @pytest.mark.asyncio
    async def test_upsert(self, cache):
        await cache.set(
            domain="test", claim_text="claim1",
            verdict="v1", evidence_summary="e1", confidence=0.5, source_count=1,
        )
        await cache.set(
            domain="test", claim_text="claim1",
            verdict="v2", evidence_summary="e2", confidence=0.8, source_count=2,
        )
        result = await cache.get("test", "claim1")
        assert result.verdict == "v2"
        assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_invalidate(self, cache):
        await cache.set(
            domain="test", claim_text="claim",
            verdict="v", evidence_summary="e", confidence=0.5, source_count=1,
        )
        removed = await cache.invalidate("test", "claim")
        assert removed is True
        result = await cache.get("test", "claim")
        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate_domain(self, cache):
        await cache.set(
            domain="d1", claim_text="c1",
            verdict="v", evidence_summary="e", confidence=0.5, source_count=1,
        )
        await cache.set(
            domain="d2", claim_text="c2",
            verdict="v", evidence_summary="e", confidence=0.5, source_count=1,
        )
        removed = await cache.invalidate_domain("d1")
        assert removed == 1
        assert await cache.get("d2", "c2") is not None

    @pytest.mark.asyncio
    async def test_stats(self, cache):
        await cache.set(
            domain="test", claim_text="c1",
            verdict="v", evidence_summary="e", confidence=0.5, source_count=1,
        )
        stats = await cache.get_stats()
        assert stats.total_entries == 1
        assert "test" in stats.domain_breakdown


class TestKeyGeneration:
    def test_key_consistency(self):
        k1 = VerificationCache._make_key("healthcare", "Aspirin cures cancer")
        k2 = VerificationCache._make_key("healthcare", "aspirin  cures  cancer")
        assert k1 == k2

    def test_key_different_domains(self):
        k1 = VerificationCache._make_key("healthcare", "claim")
        k2 = VerificationCache._make_key("finance", "claim")
        assert k1 != k2
