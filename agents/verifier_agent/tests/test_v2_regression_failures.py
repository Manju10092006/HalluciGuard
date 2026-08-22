"""
HalluciGuard Verifier Agent — V2 Regression Failure Unit Tests.

Deterministic unit tests validating fixes for baseline failure modes:
  1. False Verification of Subnational Capitals (Hyderabad / Telangana vs India)
  2. Missed Contradiction on Location (Eiffel Tower in London vs Paris)
  3. Missed Contradiction on Kinship (Chiranjeevi father of Allu Arjun vs Allu Aravind)
  4. Correct Match on Creators (Java created by James Gosling)
  5. Contradiction-Suppression Bypass on Object Mismatch
  6. Dynamic Gate Relevance Assessment (No Hardcoded 0.85 Priors)
"""
import sys
import os
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
VERIFIER_DIR = os.path.join(PROJECT_ROOT, "agents", "verifier_agent")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if VERIFIER_DIR not in sys.path:
    sys.path.insert(0, VERIFIER_DIR)

from schemas.models import Passage, VerdictLabel
from scorers.relation_verifier import RelationVerifier
from scorers.evidence_scorer import EvidenceScorer
from adapters.web_enhanced import WebEnhancedAdapter
from unittest.mock import MagicMock


def _make_passage(
    title: str = "Test",
    snippet: str = "Test snippet content.",
    relevance_score: float = 0.0,
    source_confidence_hint: float = 0.85,
    url: str = "https://en.wikipedia.org/wiki/Test",
    source: str = "wikipedia",
    source_id: str = "wiki_test",
) -> Passage:
    return Passage(
        title=title,
        source=source,
        url=url,
        publication_date="2024-01-01",
        snippet=snippet,
        source_id=source_id,
        relevance_score=relevance_score,
        source_confidence_hint=source_confidence_hint,
    )


class TestV2RegressionFailures:
    """Test suite for Verifier V2 core truth and relation fixes."""

    def setup_method(self):
        self.rv = RelationVerifier()
        self.scorer = EvidenceScorer()

    def test_hyderabad_capital_relation_check(self):
        """Failure 1: Hyderabad is capital of India -> OBJECT_MISMATCH -> CONTRADICTED."""
        claim = "Hyderabad is the capital of India."
        passage = _make_passage(
            title="Hyderabad",
            snippet="Hyderabad is the capital and largest city of the Indian state of Telangana.",
            relevance_score=0.85,
        )
        res = self.rv.verify_relation(claim, [passage])
        assert res.status == "OBJECT_MISMATCH"
        assert "telangana" in res.mismatch_detail.lower()

        nli = {"label": "entailment", "entailment_score": 0.93, "contradiction_score": 0.05, "neutral_score": 0.02}
        ev_class = self.scorer.classify_evidence(claim, passage, nli)
        assert ev_class == "CONTRADICTING"

        scores = self.scorer.score_evidence(claim, [passage], [nli], domain="general")
        assert scores["verdict"] == VerdictLabel.CONTRADICTED

    def test_eiffel_tower_location_relation_check(self):
        """Failure 2: Eiffel Tower located in London -> OBJECT_MISMATCH -> CONTRADICTED."""
        claim = "The Eiffel Tower is located in London."
        passage = _make_passage(
            title="Eiffel Tower",
            snippet="The Eiffel Tower is a lattice tower on the Champ de Mars in Paris, France.",
            relevance_score=0.80,
        )
        res = self.rv.verify_relation(claim, [passage])
        assert res.status == "OBJECT_MISMATCH"
        assert "paris" in res.mismatch_detail.lower()

        # Notice NLI gives 0.00 contradiction, neutral label
        nli = {"label": "neutral", "entailment_score": 0.01, "contradiction_score": 0.02, "neutral_score": 0.97}
        ev_class = self.scorer.classify_evidence(claim, passage, nli)
        assert ev_class == "CONTRADICTING"

        scores = self.scorer.score_evidence(claim, [passage], [nli], domain="general")
        assert scores["verdict"] == VerdictLabel.CONTRADICTED

    def test_allu_arjun_father_relation_check(self):
        """Failure 3: Chiranjeevi is father of Allu Arjun -> OBJECT_MISMATCH -> CONTRADICTED."""
        claim = "Chiranjeevi is the father of Allu Arjun."
        passage = _make_passage(
            title="Allu Arjun",
            snippet="Allu Arjun was born on 8 April 1982 in a Telugu family to film producer Allu Aravind and Nirmala in Madras",
            relevance_score=0.75,
        )
        res = self.rv.verify_relation(claim, [passage])
        assert res.status == "OBJECT_MISMATCH"
        assert "allu aravind" in res.mismatch_detail.lower()

        nli = {"label": "neutral", "entailment_score": 0.02, "contradiction_score": 0.10, "neutral_score": 0.88}
        ev_class = self.scorer.classify_evidence(claim, passage, nli)
        assert ev_class == "CONTRADICTING"

        scores = self.scorer.score_evidence(claim, [passage], [nli], domain="general")
        assert scores["verdict"] == VerdictLabel.CONTRADICTED

    def test_java_creator_relation_check(self):
        """Failure 4: Java created by James Gosling -> MATCH -> VERIFIED."""
        claim = "Java was created by James Gosling."
        passage = _make_passage(
            title="Java (programming language)",
            snippet="Java was originally developed by James Gosling at Sun Microsystems and released in May 1995.",
            relevance_score=0.90,
        )
        res = self.rv.verify_relation(claim, [passage])
        assert res.status == "MATCH"

        nli = {"label": "entailment", "entailment_score": 0.98, "contradiction_score": 0.01, "neutral_score": 0.01}
        ev_class = self.scorer.classify_evidence(claim, passage, nli)
        assert ev_class == "SUPPORTING"

        scores = self.scorer.score_evidence(claim, [passage], [nli], domain="general")
        assert scores["verdict"] == VerdictLabel.VERIFIED

    def test_coverage_suppression_bypass(self):
        """Verify that object mismatch bypasses word-coverage suppression rule."""
        claim = "The Eiffel Tower is located in London."
        # Snippet contains 'eiffel tower' and 'paris', but NOT 'london'
        passage = _make_passage(
            title="Eiffel Tower",
            snippet="The Eiffel Tower is a landmark situated in Paris.",
            relevance_score=0.80,
        )
        nli = {"label": "contradiction", "entailment_score": 0.0, "contradiction_score": 0.90, "neutral_score": 0.10}
        # Without relation check, covered words = 3/4 ("eiffel", "tower", "located"), "london" missing -> would be NEUTRAL
        # With relation check bypass -> must be CONTRADICTING
        ev_class = self.scorer.classify_evidence(claim, passage, nli)
        assert ev_class == "CONTRADICTING"

    def test_gate_relevance_dynamic_assessment(self):
        """Verify that web_enhanced._assess_primary_quality computes overlap, not a constant 0.85."""
        mock_primary = MagicMock()
        mock_primary.name = "general"
        adapter = WebEnhancedAdapter(primary_adapter=mock_primary)

        # 1. Zero overlap passage -> low relevance
        p_irrelevant = _make_passage(
            title="Quantum Physics",
            snippet="Quantum electrodynamics describes relativistic quantum field theory.",
            relevance_score=0.0,
            source_confidence_hint=0.85,
        )
        trace_irrel = adapter._assess_primary_quality([p_irrelevant], "The Eiffel Tower is located in London.")
        assert trace_irrel.top_relevance < 0.30

        # 2. High overlap passage -> high relevance
        p_relevant = _make_passage(
            title="Eiffel Tower",
            snippet="The Eiffel Tower is located in Paris France on the Champ de Mars.",
            relevance_score=0.0,
            source_confidence_hint=0.85,
        )
        trace_rel = adapter._assess_primary_quality([p_relevant], "The Eiffel Tower is located in London.")
        assert trace_rel.top_relevance > 0.40