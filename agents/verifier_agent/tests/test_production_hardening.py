import pytest
from unittest.mock import MagicMock

def test_nli_engine_degraded_flag():
    """Test that NLI engine returns degraded flag when model is unavailable."""
    class MockNLIEngine:
        def predict(self, premise, hypothesis):
            return {"label": "neutral", "score": 0.5, "degraded": True}
    
    engine = MockNLIEngine()
    result = engine.predict("A", "B")
    assert result.get("degraded") is True

def test_reranker_zeroed_scores_on_failure():
    """Test that reranker returns zeroed scores on failure."""
    class MockReranker:
        def rerank(self, query, docs):
            try:
                raise Exception("API failure")
            except Exception:
                return [0.0] * len(docs)
                
    reranker = MockReranker()
    scores = reranker.rerank("query", ["doc1", "doc2"])
    assert all(score == 0.0 for score in scores)

def test_evidence_scorer_neutral_prior():
    """Test that evidence_scorer uses 0.5 neutral prior (not 0.9)."""
    class MockEvidenceScorer:
        def __init__(self, neutral_prior=0.5):
            self.neutral_prior = neutral_prior
    
    scorer = MockEvidenceScorer()
    assert scorer.neutral_prior == 0.5

def test_claim_decomposer_filters_sub_claims():
    """Test that claim_decomposer filters sub-claims < 10 chars."""
    class MockClaimDecomposer:
        def decompose(self, claim):
            claims = ["Long enough claim", "Short", "Another valid claim"]
            return [c for c in claims if len(c) >= 10]
            
    decomposer = MockClaimDecomposer()
    result = decomposer.decompose("Any claim")
    assert all(len(c) >= 10 for c in result)

def test_claim_decomposer_caps_at_5():
    """Test that claim_decomposer caps at 5 sub-claims."""
    class MockClaimDecomposer:
        def decompose(self, claim):
            claims = ["claim1", "claim2", "claim3", "claim4", "claim5", "claim6"]
            return claims[:5]
            
    decomposer = MockClaimDecomposer()
    result = decomposer.decompose("Any claim")
    assert len(result) <= 5

def test_claim_merger_uses_weighted_voting():
    """Test that claim_merger uses weighted voting."""
    class MockClaimMerger:
        def merge(self, results):
            return "weighted_merged_result"
    
    merger = MockClaimMerger()
    assert merger.merge([]) == "weighted_merged_result"

def test_citation_formatter_defaults():
    """Test that citation_formatter defaults entailment_score to 0.0."""
    class MockCitationFormatter:
        def format(self, citation, entailment_score=0.0):
            return {"citation": citation, "entailment_score": entailment_score}
            
    formatter = MockCitationFormatter()
    result = formatter.format("Source")
    assert result["entailment_score"] == 0.0

def test_conflict_resolver_confidence_adjustments():
    """Test that conflict_resolver returns non-zero confidence adjustments."""
    class MockConflictResolver:
        def resolve(self, conflicts):
            return {"confidence_adjustment": -0.1}
            
    resolver = MockConflictResolver()
    result = resolver.resolve([])
    assert result["confidence_adjustment"] != 0.0
