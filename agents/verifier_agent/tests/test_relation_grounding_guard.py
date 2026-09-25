"""
HalluciGuard Verifier Agent — Relation-Grounding Guard Regression Tests.

Locks the Phase 1 certification fixes (see CERTIFICATION_AUDIT_REPORT.md):

  F-2  `_names_match` no longer treats a single token swallowed by a longer,
       DIFFERENT entity as the same entity. The prior substring / single-token
       subset logic made "Paris" match "Paris, Texas" and "India" match
       "Indiana", forcing a spurious 0.95 entailment and a FALSE VERIFIED.

  F-1  A VERIFIED verdict that NO structured relation check could ground rests
       purely on raw NLI + lexical heuristics. Its confidence is now capped at
       `ungrounded_confidence_ceiling` so the Judge does not treat an
       un-grounded accept as high-certainty. The verdict itself is unchanged,
       and CONTRADICTED confidence is left intact (fail-closed keeps refutations
       strong).

These are deterministic, network-free tests (no model download required for the
relation layer; the evidence-scorer tests drive `score_evidence` with explicit
NLI dicts).
"""
import os
import sys

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


class TestNamesMatchGuard:
    """F-2 unit tests: `_names_match` returns True only for the SAME entity."""

    def setup_method(self):
        self.rv = RelationVerifier()

    def test_exact_equality_matches(self):
        assert self.rv._names_match("Paris", "Paris") is True

    def test_single_token_swallowed_by_longer_name_rejected_paris_texas(self):
        # "Paris" (France, implied) must NOT be equated with "Paris, Texas".
        # This was the headline false-VERIFIED vector before the fix.
        assert self.rv._names_match("Paris", "Paris, Texas") is False
        assert self.rv._names_match("Paris, Texas", "Paris") is False

    def test_single_token_prefix_rejected_india_indiana(self):
        # "India" is a substring of "Indiana" — the old substring test matched.
        assert self.rv._names_match("India", "Indiana") is False
        assert self.rv._names_match("Indiana", "India") is False

    def test_distinct_single_tokens_rejected(self):
        assert self.rv._names_match("London", "Paris") is False

    def test_multi_token_subset_matches(self):
        # Legitimate qualifier expansion: the smaller name is a >=2-token subset.
        assert self.rv._names_match("Eiffel Tower", "Eiffel Tower landmark") is True

    def test_honorific_stripped_then_equal(self):
        # "Sir" is stripped by _normalize_name, so this resolves via exact equality.
        assert self.rv._names_match("James Gosling", "Sir James Gosling") is True

    def test_distinct_multi_token_names_rejected(self):
        assert self.rv._names_match("James Gosling", "Guido van Rossum") is False

    def test_empty_name_rejected(self):
        assert self.rv._names_match("", "Paris") is False
        assert self.rv._names_match("Paris", "") is False


class TestRelationGroundingEndToEnd:
    """F-2 end-to-end: the substring vector no longer manufactures a MATCH,
    while the legitimate multi-token containment path still matches."""

    def setup_method(self):
        self.rv = RelationVerifier()

    def test_india_indiana_capital_is_object_mismatch_not_false_match(self):
        # "Indianapolis is the capital of India" is FALSE (New Delhi is India's
        # capital; Indianapolis is Indiana's). Before the fix, _names_match(
        # "india","indiana") returned True via substring => false MATCH =>
        # false VERIFIED. It must now surface as an OBJECT_MISMATCH.
        claim = "Indianapolis is the capital of India."
        evidence = _make_passage(
            title="Indianapolis",
            snippet="Indianapolis is the capital of Indiana.",
            relevance_score=0.85,
        )
        res = self.rv.verify_relation(claim, [evidence])
        assert res.status == "OBJECT_MISMATCH"
        assert "indiana" in res.mismatch_detail.lower()

    def test_multi_token_subject_subset_still_matches(self):
        # The guarded multi-token containment path must still confirm a true
        # relation: claim subject "Eiffel Tower" is a 2-token subset of the
        # evidence subject "Eiffel Tower landmark", same object "Paris".
        claim = "The Eiffel Tower is located in Paris."
        evidence = _make_passage(
            title="Eiffel Tower",
            snippet="The Eiffel Tower landmark is located in Paris.",
            relevance_score=0.85,
        )
        res = self.rv.verify_relation(claim, [evidence])
        assert res.status == "MATCH"


class TestUngroundedConfidenceCap:
    """F-1: a VERIFIED verdict with no decisive relation grounding is
    confidence-capped; CONTRADICTED and grounded VERIFIED are not."""

    def setup_method(self):
        self.scorer = EvidenceScorer()

    def test_ungrounded_verified_confidence_is_capped(self, monkeypatch):
        # Force the ceiling to 0.0 so the cap is unambiguously observable: any
        # positive VERIFIED confidence must be driven to exactly 0.0, while the
        # VERDICT stays VERIFIED (the cap never changes the verdict).
        monkeypatch.setattr(
            EvidenceScorer, "_get_ungrounded_confidence_ceiling", lambda self: 0.0
        )
        claim = "Water boils at 100 degrees Celsius at sea level."  # no relation triple
        passage = _make_passage(
            title="Boiling point of water",
            snippet="At standard sea-level pressure, water boils at 100 degrees Celsius.",
            relevance_score=0.9,
        )
        nli = {
            "label": "entailment",
            "entailment_score": 0.97,
            "contradiction_score": 0.01,
            "neutral_score": 0.02,
        }
        result = self.scorer.score_evidence(claim, [passage], [nli], domain="general")
        assert result["grounded"] is False
        assert result["verdict"] == VerdictLabel.VERIFIED
        # Positive pre-cap confidence (VERIFIED implies support>=0.30) -> capped to 0.0.
        assert result["confidence_score"] == 0.0

    def test_grounded_verified_confidence_is_not_capped(self, monkeypatch):
        # Same ceiling of 0.0, but a GROUNDED claim (relation MATCH) must skip
        # the cap entirely: confidence stays strictly positive.
        monkeypatch.setattr(
            EvidenceScorer, "_get_ungrounded_confidence_ceiling", lambda self: 0.0
        )
        claim = "Java was created by James Gosling."
        passage = _make_passage(
            title="Java (programming language)",
            snippet="Java was originally developed by James Gosling at Sun Microsystems and released in May 1995.",
            relevance_score=0.9,
        )
        nli = {
            "label": "entailment",
            "entailment_score": 0.98,
            "contradiction_score": 0.01,
            "neutral_score": 0.01,
        }
        result = self.scorer.score_evidence(claim, [passage], [nli], domain="general")
        assert result["grounded"] is True
        assert result["verdict"] == VerdictLabel.VERIFIED
        assert result["confidence_score"] > 0.0

    def test_ungrounded_contradicted_confidence_is_not_capped(self, monkeypatch):
        # Fail-closed: refutations keep their confidence. The cap targets
        # VERIFIED only, so an ungrounded CONTRADICTED stays strong even at a
        # 0.0 ceiling.
        monkeypatch.setattr(
            EvidenceScorer, "_get_ungrounded_confidence_ceiling", lambda self: 0.0
        )
        claim = "Water boils at 100 degrees Celsius at sea level."
        passage = _make_passage(
            title="Boiling point",
            snippet="Water does not boil at 100 degrees Celsius at sea level; that claim is false.",
            relevance_score=0.9,
        )
        nli = {
            "label": "contradiction",
            "entailment_score": 0.02,
            "contradiction_score": 0.90,
            "neutral_score": 0.08,
        }
        result = self.scorer.score_evidence(claim, [passage], [nli], domain="general")
        assert result["grounded"] is False
        assert result["verdict"] == VerdictLabel.CONTRADICTED
        assert result["confidence_score"] > 0.0

    def test_default_ceiling_binds_on_multisource_ungrounded_verified(self):
        # With the production default ceiling (0.70), a strong multi-source
        # ungrounded VERIFIED must not report confidence above the ceiling.
        claim = "Water boils at 100 degrees Celsius at sea level."
        passages = [
            _make_passage(
                title=f"Source {i}",
                snippet=f"Reference {i}: at sea level, water boils at 100 degrees Celsius.",
                relevance_score=0.9,
                url=f"https://example{i}.org/water",
                source_id=f"src_{i}",
            )
            for i in range(3)
        ]
        nlis = [
            {
                "label": "entailment",
                "entailment_score": 0.97,
                "contradiction_score": 0.01,
                "neutral_score": 0.02,
            }
            for _ in passages
        ]
        result = self.scorer.score_evidence(claim, passages, nlis, domain="general")
        assert result["grounded"] is False
        assert result["verdict"] == VerdictLabel.VERIFIED
        assert result["confidence_score"] <= 0.70

    def test_empty_evidence_reports_ungrounded(self):
        result = self.scorer.score_evidence("Some claim.", [], [], domain="general")
        assert result["grounded"] is False
        assert result["verdict"] == VerdictLabel.UNVERIFIED
