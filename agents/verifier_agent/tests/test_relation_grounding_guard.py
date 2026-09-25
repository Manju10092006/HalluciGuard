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
from api.pipeline import VerificationPipeline


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


class TestPageTitleCleaning:
    """`_clean_page_title` must reduce a decorated retrieval title to the bare
    entity, so title-anchored passive facts align with a single-token claim
    subject. Real adapters emit these exact decorations (see
    scripts/probe_retrieval.py)."""

    def setup_method(self):
        self.rv = RelationVerifier()

    def test_strips_wikipedia_prefix(self):
        assert self.rv._clean_page_title("Wikipedia: Microsoft") == "Microsoft"

    def test_strips_web_prefix_and_wikipedia_suffix(self):
        # Tavily form. Before the fix this normalized to "web microsoft
        # wikipedia" and never matched the claim subject "microsoft".
        assert self.rv._clean_page_title("Web: Microsoft - Wikipedia") == "Microsoft"

    def test_strips_section_parenthetical(self):
        assert self.rv._clean_page_title("Wikipedia: Microsoft (Overview)") == "Microsoft"

    def test_preserves_city_state_comma(self):
        # A comma is NOT a separator we strip — "Albuquerque, New Mexico" is one
        # entity and must survive intact (the object matcher needs the state).
        assert self.rv._clean_page_title("Wikipedia: Albuquerque, New Mexico") == "Albuquerque, New Mexico"


class TestFounderTitleGrounding:
    """A true (co-)founder claim must MATCH even when the only decisive evidence
    is a title-anchored passive lead on a Tavily-decorated page — the false
    CONFLICTED vector for 'Microsoft was founded by Bill Gates/Paul Allen'."""

    def setup_method(self):
        self.rv = RelationVerifier()

    def test_founder_grounded_via_tavily_title_only(self):
        # The founding fact is recoverable ONLY through the page title: the lead
        # is passive ("Founded in 1975 by ...") with no inline org subject, and
        # the title is Tavily-decorated. This isolates the title-cleaning fix.
        claim = "Microsoft was founded by Bill Gates."
        evidence = _make_passage(
            title="Web: Microsoft - Wikipedia",
            snippet="American multinational technology company. Founded in 1975 by Bill Gates and Paul Allen.",
            source="web",
            relevance_score=0.85,
        )
        res = self.rv.verify_relation(claim, [evidence])
        assert res.status == "MATCH"

    def test_cofounder_not_contradicted_by_single_founder_context(self):
        # "Microsoft was founded by Paul Allen" is TRUE. A Bill-Gates biography
        # ("He co-founded Microsoft") must not out-vote the multi-founder lead
        # and force OBJECT_MISMATCH — the match-first pre-scan confirms Paul Allen.
        claim = "Microsoft was founded by Paul Allen."
        gates_bio = _make_passage(
            title="Wikipedia: Bill Gates",
            snippet="Bill Gates is an American businessman who co-founded Microsoft.",
            relevance_score=0.85,
        )
        ms_lead = _make_passage(
            title="Wikipedia: Microsoft",
            snippet="Microsoft Corporation is a technology company. Founded in 1975 by Bill Gates and Paul Allen.",
            relevance_score=0.85,
        )
        res = self.rv.verify_relation(claim, [gates_bio, ms_lead])
        assert res.status == "MATCH"

    def test_founder_grounded_when_full_date_precedes_by(self):
        claim = "Microsoft was founded by Bill Gates in April 1975."
        evidence = _make_passage(
            title="Wikipedia: Microsoft (Overview)",
            snippet=(
                "Microsoft is a computer technology corporation founded on "
                "April 4, 1975, by Bill Gates and Paul Allen in Albuquerque, New Mexico."
            ),
            relevance_score=0.85,
        )
        res = self.rv.verify_relation(claim, [evidence])
        assert res.status == "MATCH"


class TestOrgLocationGrounding:
    """Organization location / headquarters claims must ground deterministically
    (the false CONTRADICTED vector for 'started in Albuquerque' / 'HQ in
    Redmond'), stay multi-location tolerant, yet still catch a real wrong city."""

    def setup_method(self):
        self.rv = RelationVerifier()

    def _ms_founding(self):
        return _make_passage(
            title="Wikipedia: Microsoft (History)",
            snippet=(
                "Microsoft is a computer technology corporation founded on April 4, 1975, "
                "by Bill Gates and Paul Allen in Albuquerque, New Mexico. The company later "
                "moved its headquarters to Redmond, Washington."
            ),
            relevance_score=0.85,
        )

    def _ms_hq(self):
        return _make_passage(
            title="Wikipedia: Microsoft (Overview)",
            snippet="The company is based at Microsoft's headquarters in Redmond, Washington.",
            relevance_score=0.85,
        )

    def test_started_in_city_matches_despite_later_hq(self):
        # #4: TRUE. Evidence proves BOTH Albuquerque (founding) and Redmond (HQ);
        # multi-location tolerance must confirm Albuquerque rather than let the
        # Redmond triple manufacture an OBJECT_MISMATCH.
        claim = "Microsoft was started in Albuquerque, New Mexico."
        res = self.rv.verify_relation(claim, [self._ms_founding(), self._ms_hq()])
        assert res.status == "MATCH"

    def test_headquarters_city_matches(self):
        # #6: TRUE. The HQ claim must ground against the supporting HQ passage
        # (raw NLI mislabels this true statement as a contradiction).
        claim = "Microsoft's headquarters remain in Redmond, Washington today."
        res = self.rv.verify_relation(claim, [self._ms_hq(), self._ms_founding()])
        assert res.status == "MATCH"

    def test_wrong_city_is_object_mismatch(self):
        # Tolerance must NOT hide a genuine contradiction: a subject-aligned
        # location claim that NO evidence confirms is still OBJECT_MISMATCH.
        claim = "Microsoft was started in Boston, Massachusetts."
        res = self.rv.verify_relation(claim, [self._ms_founding(), self._ms_hq()])
        assert res.status == "OBJECT_MISMATCH"
        # The detail must cite a real proven location, not the unconfirmed claim city.
        detail = (res.mismatch_detail or "").lower()
        assert "albuquerque" in detail or "redmond" in detail

    def test_city_state_disambiguation_preserved(self):
        # F-2 protection carried into org location: "Paris, Texas" must not be
        # confirmed by evidence about "Paris, France".
        claim = "Foobar Inc is headquartered in Paris, Texas."
        evidence = _make_passage(
            title="Wikipedia: Foobar Inc",
            snippet="Foobar Inc is headquartered in Paris, France.",
            relevance_score=0.85,
        )
        res = self.rv.verify_relation(claim, [evidence])
        assert res.status == "OBJECT_MISMATCH"

    def test_founder_date_is_not_misread_as_location(self):
        claim = "Microsoft was founded by Bill Gates in April 1975."
        triples = self.rv.extract_triples(claim)
        assert any(t.relation == "created_by" for t in triples)
        assert all(
            not (t.relation == "location_of" and t.object == "april")
            for t in triples
        )


class TestDecisionGradeTopicality:
    """NLI cannot turn an entity-only lexical hit into a refutation."""

    def setup_method(self):
        self.rv = RelationVerifier()

    def test_off_topic_city_page_cannot_contradict_org_location_claim(self):
        claim = "Microsoft was started in Albuquerque, New Mexico."
        city_page = _make_passage(
            title="Wikipedia: Albuquerque, New Mexico",
            snippet=(
                "Albuquerque is the most populous city in New Mexico and was "
                "founded in 1706 as La Villa de Alburquerque."
            ),
            relevance_score=0.91,
        )
        nli = [{
            "label": "contradiction",
            "entailment_score": 0.01,
            "contradiction_score": 0.98,
            "neutral_score": 0.01,
        }]

        passages, results = VerificationPipeline._select_decision_grade_evidence(
            [city_page], nli, claim=claim, relation_verifier=self.rv
        )

        assert passages == []
        assert results == []
        assert nli[0]["contradiction_score"] == 0.0
        assert nli[0]["label"] == "neutral"

    def test_subject_aligned_wrong_location_remains_contradiction(self):
        claim = "Microsoft was started in Boston, Massachusetts."
        microsoft_page = _make_passage(
            title="Wikipedia: Microsoft",
            snippet=(
                "Microsoft was founded by Bill Gates and Paul Allen in "
                "Albuquerque, New Mexico."
            ),
            relevance_score=0.91,
        )
        nli = [{
            "label": "neutral",
            "entailment_score": 0.02,
            "contradiction_score": 0.08,
            "neutral_score": 0.90,
        }]

        passages, results = VerificationPipeline._select_decision_grade_evidence(
            [microsoft_page], nli, claim=claim, relation_verifier=self.rv
        )

        assert passages == [microsoft_page]
        assert results[0]["label"] == "contradiction"
        assert results[0]["contradiction_score"] >= 0.95
