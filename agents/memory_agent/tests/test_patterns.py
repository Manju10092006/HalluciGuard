import pytest

from agents.memory_agent.patterns.pattern_learner import PatternLearner
from agents.memory_agent.schemas.models import PatternType


@pytest.fixture
async def learner(tmp_path):
    pl = PatternLearner(db_path=str(tmp_path / "test_patterns.db"))
    await pl.initialize()
    yield pl
    await pl.close()


class TestPatternClassification:
    def test_temporal_claim(self):
        types = PatternLearner._classify_claim(
            "The internet was invented and created in 1969"
        )
        assert PatternType.TEMPORAL in types

    def test_numerical_claim(self):
        types = PatternLearner._classify_claim(
            "The stock price increased by 45 percent"
        )
        assert PatternType.NUMERICAL in types

    def test_statistical_claim(self):
        types = PatternLearner._classify_claim(
            "A study found that 90% of patients improved"
        )
        assert PatternType.STATISTICAL in types

    def test_causal_claim(self):
        types = PatternLearner._classify_claim(
            "Smoking causes lung cancer and leads to death"
        )
        assert PatternType.CAUSAL in types

    def test_definition_claim(self):
        types = PatternLearner._classify_claim(
            "Machine learning is defined as a subset of AI"
        )
        assert PatternType.DEFINITION in types

    def test_keyword_extraction(self):
        keywords = PatternLearner._extract_keywords(
            "The quick brown fox jumps over the lazy dog"
        )
        assert "quick" in keywords
        assert "brown" in keywords


class TestPatternLearning:
    @pytest.mark.asyncio
    async def test_observe_claim_creates_pattern(self, learner):
        patterns = await learner.observe_claim(
            claim_text="Aspirin cures cancer",
            domain="healthcare",
            verdict="likely_hallucinated",
        )
        assert len(patterns) > 0

    @pytest.mark.asyncio
    async def test_observe_increases_frequency(self, learner):
        await learner.observe_claim(
            claim_text="Aspirin cures cancer", domain="healthcare",
            verdict="likely_hallucinated",
        )
        patterns = await learner.observe_claim(
            claim_text="Aspirin cures cancer", domain="healthcare",
            verdict="likely_hallucinated",
        )
        for p in patterns:
            if p.frequency > 1:
                return
        pytest.fail("Frequency should increase")

    @pytest.mark.asyncio
    async def test_query_patterns(self, learner):
        await learner.observe_claim(
            claim_text="Aspirin cures cancer", domain="healthcare",
            verdict="likely_hallucinated",
        )
        results = await learner.query_patterns(domain="healthcare")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_query_by_type(self, learner):
        await learner.observe_claim(
            claim_text="The stock price increased by 45 percent",
            domain="finance",
            verdict="likely_hallucinated",
        )
        results = await learner.query_patterns(
            domain="finance", pattern_type=PatternType.NUMERICAL
        )
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_domain_summary(self, learner):
        await learner.observe_claim(
            claim_text="Aspirin cures cancer", domain="healthcare",
            verdict="likely_hallucinated",
        )
        summary = await learner.get_domain_summary("healthcare")
        assert len(summary) > 0

    @pytest.mark.asyncio
    async def test_total_count(self, learner):
        count = await learner.get_total_count()
        assert count == 0
        await learner.observe_claim(
            claim_text="Aspirin cures cancer", domain="healthcare",
            verdict="likely_hallucinated",
        )
        count = await learner.get_total_count()
        assert count > 0
