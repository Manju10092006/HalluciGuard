import pytest

from agents.memory_agent.trust.source_trust import SourceTrustManager
from agents.memory_agent.schemas.models import TrustChangeReason


@pytest.fixture
async def trust(tmp_path):
    mgr = SourceTrustManager(
        db_path=str(tmp_path / "test_trust.db"),
        prior=0.5,
        learning_rate=0.1,
        decay_rate=0.01,
    )
    await mgr.initialize()
    yield mgr
    await mgr.close()


class TestTrustUpdate:
    @pytest.mark.asyncio
    async def test_new_source_gets_prior(self, trust):
        update = await trust.update_trust(
            source_id="pubmed",
            source_name="PubMed",
            domain="healthcare",
            reason=TrustChangeReason.VERIFIED_CORRECT,
        )
        assert update.source_id == "pubmed"
        assert update.old_score == 0.5
        assert update.new_score > 0.5

    @pytest.mark.asyncio
    async def test_correct_increases_trust(self, trust):
        u1 = await trust.update_trust(
            source_id="s1", source_name="S1", domain="test",
            reason=TrustChangeReason.VERIFIED_CORRECT,
        )
        u2 = await trust.update_trust(
            source_id="s1", source_name="S1", domain="test",
            reason=TrustChangeReason.VERIFIED_CORRECT,
        )
        assert u2.new_score > u1.new_score

    @pytest.mark.asyncio
    async def test_incorrect_decreases_trust(self, trust):
        u1 = await trust.update_trust(
            source_id="s1", source_name="S1", domain="test",
            reason=TrustChangeReason.VERIFIED_INCORRECT,
        )
        assert u1.new_score < 0.5
        u2 = await trust.update_trust(
            source_id="s1", source_name="S1", domain="test",
            reason=TrustChangeReason.VERIFIED_INCORRECT,
        )
        assert u2.new_score < u1.new_score

    @pytest.mark.asyncio
    async def test_trust_bounded_0_1(self, trust):
        for _ in range(100):
            await trust.update_trust(
                source_id="s1", source_name="S1", domain="test",
                reason=TrustChangeReason.VERIFIED_CORRECT,
            )
        record = await trust.get_trust("s1")
        assert record.trust_score <= 1.0

    @pytest.mark.asyncio
    async def test_trust_bounded_floor(self, trust):
        for _ in range(100):
            await trust.update_trust(
                source_id="s1", source_name="S1", domain="test",
                reason=TrustChangeReason.VERIFIED_INCORRECT,
            )
        record = await trust.get_trust("s1")
        assert record.trust_score >= 0.0


class TestTrustRetrieval:
    @pytest.mark.asyncio
    async def test_get_trust(self, trust):
        await trust.update_trust(
            source_id="s1", source_name="S1", domain="test",
            reason=TrustChangeReason.VERIFIED_CORRECT,
        )
        record = await trust.get_trust("s1")
        assert record is not None
        assert record.source_name == "S1"

    @pytest.mark.asyncio
    async def test_get_trust_nonexistent(self, trust):
        record = await trust.get_trust("nope")
        assert record is None

    @pytest.mark.asyncio
    async def test_get_domain_sources(self, trust):
        await trust.update_trust(
            source_id="s1", source_name="S1", domain="d1",
            reason=TrustChangeReason.VERIFIED_CORRECT,
        )
        await trust.update_trust(
            source_id="s2", source_name="S2", domain="d2",
            reason=TrustChangeReason.VERIFIED_CORRECT,
        )
        sources = await trust.get_domain_sources("d1")
        assert len(sources) == 1
        assert sources[0].source_id == "s1"

    @pytest.mark.asyncio
    async def test_get_top_sources(self, trust):
        await trust.update_trust(
            source_id="s1", source_name="S1", domain="d1",
            reason=TrustChangeReason.VERIFIED_CORRECT,
        )
        await trust.update_trust(
            source_id="s2", source_name="S2", domain="d1",
            reason=TrustChangeReason.VERIFIED_INCORRECT,
        )
        top = await trust.get_top_sources(domain="d1")
        assert len(top) == 2
        assert top[0].trust_score >= top[1].trust_score


class TestTrustHistory:
    @pytest.mark.asyncio
    async def test_history_recorded(self, trust):
        await trust.update_trust(
            source_id="s1", source_name="S1", domain="d1",
            reason=TrustChangeReason.VERIFIED_CORRECT,
        )
        history = await trust.get_trust_history("s1")
        assert len(history) == 1
        assert history[0]["reason"] == "verified_correct"

    @pytest.mark.asyncio
    async def test_multiple_history_entries(self, trust):
        await trust.update_trust(
            source_id="s1", source_name="S1", domain="d1",
            reason=TrustChangeReason.VERIFIED_CORRECT,
        )
        await trust.update_trust(
            source_id="s1", source_name="S1", domain="d1",
            reason=TrustChangeReason.VERIFIED_INCORRECT,
        )
        history = await trust.get_trust_history("s1")
        assert len(history) == 2


class TestDecay:
    @pytest.mark.asyncio
    async def test_global_decay(self, trust):
        await trust.update_trust(
            source_id="s1", source_name="S1", domain="d1",
            reason=TrustChangeReason.VERIFIED_CORRECT,
        )
        before = await trust.get_trust("s1")
        decayed = await trust.apply_global_decay()
        assert decayed >= 1
        after = await trust.get_trust("s1")
        assert after.trust_score < before.trust_score

    @pytest.mark.asyncio
    async def test_total_sources(self, trust):
        assert await trust.get_total_sources() == 0
        await trust.update_trust(
            source_id="s1", source_name="S1", domain="d1",
            reason=TrustChangeReason.VERIFIED_CORRECT,
        )
        assert await trust.get_total_sources() == 1
