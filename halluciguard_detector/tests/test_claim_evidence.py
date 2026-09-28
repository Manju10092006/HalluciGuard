"""Hermetic regression tests for claim-level evidence verification.

These tests never load the DeBERTa model: ``Detector`` is built with
``__new__``, given a stub evidence engine, and its inference seams are
monkeypatched. They lock in the behavioral guarantees the feature shipped with.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from halluciguard_detector.detector import Detector
from halluciguard_detector.evidence import select_evidence
from halluciguard_detector.schemas import ClaimLabel
from halluciguard_detector.text import numeric_consistency


class EvidenceStub:
    def __init__(self, decomposed=None, checkable=True, snippets=None):
        self._decomposed = decomposed
        self._checkable = checkable
        self._snippets = snippets or {}

    def decompose(self, text):
        return list(self._decomposed or [])

    def checkable(self, text):
        return bool(self._checkable)

    def select(self, claim, evidence_texts):
        if claim in self._snippets:
            return list(self._snippets[claim])
        return list(evidence_texts)


def make_detector(monkeypatch, classify=None, claims=None, evidence_stub=None):
    detector = Detector.__new__(Detector)
    detector.temperature = 1.0
    detector.threshold = 0.5
    detector.max_length = 384
    detector.version = "test"
    detector.evidence = evidence_stub or EvidenceStub()
    if claims is not None:
        monkeypatch.setattr(detector, "_atomic_claims", lambda answer: list(claims))
    if classify is not None:
        monkeypatch.setattr(detector, "_classify", classify)
    return detector


CLAIM = "Java was created by Snehith in 1995."
EVIDENCE = ["Java was created by James Gosling in 1995."]


def mapping(supported, contradicted, unknown):
    return {
        ClaimLabel.SUPPORTED: supported,
        ClaimLabel.CONTRADICTED: contradicted,
        ClaimLabel.NOT_ENOUGH_INFO: unknown,
    }


# ---------------------------------------------------------------------------
# Test 1: evidence is mandatory, including empty/whitespace-only evidence.
# ---------------------------------------------------------------------------
def test_evidence_is_mandatory_at_detector_level(monkeypatch):
    detector = make_detector(monkeypatch, classify=lambda c, e: mapping(0.9, 0.05, 0.05))
    with pytest.raises(ValueError):
        detector.detect(draft_answer=CLAIM, evidence=[])
    with pytest.raises(ValueError):
        detector.detect(draft_answer=CLAIM, evidence=["   "])


# ---------------------------------------------------------------------------
# Test 2: NOT_ENOUGH_INFO is its own class, not CONTRADICTED.
# ---------------------------------------------------------------------------
def test_no_evidence_claim_is_not_contradiction(monkeypatch):
    detector = make_detector(
        monkeypatch,
        claims=[(CLAIM, (0, len(CLAIM)))],
        evidence_stub=EvidenceStub(snippets={CLAIM: []}),
    )
    result = detector.detect(draft_answer=CLAIM, evidence=["irrelevant context"], user_query="q")
    item = result.sentences[0]
    assert item.label == ClaimLabel.NOT_ENOUGH_INFO
    assert item.contradicted_probability == 0.0
    assert item.unknown_probability == 1.0
    assert result.contradicted_count == 0
    assert result.unknown_count == 1


# ---------------------------------------------------------------------------
# Test 3: NOT_ENOUGH_INFO must never be folded into the false rate.
# ---------------------------------------------------------------------------
def test_unknown_is_not_false(monkeypatch):
    detector = make_detector(monkeypatch, classify=lambda c, e: mapping(0.35, 0.15, 0.50))
    detector._atomic_claims = lambda a: [(CLAIM, (0, len(CLAIM)))]
    result = detector.detect(draft_answer=CLAIM, evidence=EVIDENCE)
    item = result.sentences[0]
    assert item.label == ClaimLabel.NOT_ENOUGH_INFO
    assert item.contradicted_probability < 0.2
    assert result.contradicted_count == 0
    assert result.unknown_count == 1
    assert any("NOT_ENOUGH_INFO is not proof" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Test 4: the three calibrated probabilities are reported separately.
# ---------------------------------------------------------------------------
def test_separate_class_probabilities_round_trip(monkeypatch):
    neutral = "Apples are a common fruit."
    detector = make_detector(monkeypatch, classify=lambda c, e: mapping(0.80, 0.10, 0.10))
    detector._atomic_claims = lambda a: [(neutral, (0, len(neutral)))]
    result = detector.detect(
        draft_answer=neutral,
        evidence=["Apples are a widely eaten fruit."],
    )
    item = result.sentences[0]
    assert item.label == ClaimLabel.SUPPORTED
    assert item.supported_probability == pytest.approx(0.80)
    assert item.contradicted_probability == pytest.approx(0.10)
    assert item.unknown_probability == pytest.approx(0.10)
    assert item.requires_verification is False
    assert item.hallucination_probability == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Test 5: numeric/date/percent mismatch is a warning + secondary signal, never
# an invented 0.05/0.90/0.05.
# ---------------------------------------------------------------------------
def test_numeric_mismatch_primes_warning_without_inventing_probs(monkeypatch):
    detector = make_detector(monkeypatch, classify=lambda c, e: mapping(0.85, 0.05, 0.10))
    detector._atomic_claims = lambda a: [("Released in 1995.", (0, 17))]
    result = detector.detect(
        draft_answer="Released in 1995.",
        evidence=["The product was released in 1996."],
    )
    item = result.sentences[0]
    assert any("Number/date mismatch" in w for w in result.warnings)
    assert item.supported_probability == pytest.approx(0.85)
    assert item.label == ClaimLabel.SUPPORTED


def test_numeric_consistency_detects_year_conflict():
    conflicts = numeric_consistency("released in 1995", "released in 1996")
    assert conflicts and "conflicting years" in conflicts[0]
    assert numeric_consistency("released in 1995", "released in 1995") == []


# ---------------------------------------------------------------------------
# Test 6: entity-conflict guard is a modest secondary signal, not an override.
# ---------------------------------------------------------------------------
def test_entity_conflict_guard_renormalizes_without_extreme_values(monkeypatch):
    detector = make_detector(monkeypatch, classify=lambda c, e: mapping(0.75, 0.10, 0.15))
    detector._atomic_claims = lambda a: [(CLAIM, (0, len(CLAIM)))]
    result = detector.detect(
        draft_answer=CLAIM,
        evidence=["Java was created by James Gosling in 1995."],
    )
    item = result.sentences[0]
    assert any("Named-entity conflict" in w for w in result.warnings)
    assert item.contradicted_probability > 0.10
    assert item.contradicted_probability < 0.90
    total = item.supported_probability + item.contradicted_probability + item.unknown_probability
    assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 7: opinion / non-factual claims never become hallucination.
# ---------------------------------------------------------------------------
def test_opinion_claim_is_not_auto_hallucination(monkeypatch):
    stub = EvidenceStub(checkable=False)
    detector = make_detector(monkeypatch, evidence_stub=stub)
    detector._atomic_claims = lambda a: [("I believe Java is the best language.", (0, 33))]
    result = detector.detect(
        draft_answer="I believe Java is the best language.",
        evidence=EVIDENCE,
    )
    item = result.sentences[0]
    assert item.non_factual is True
    assert item.hallucination_probability == 0.0
    assert item.risk.value == "LOW"
    assert item.requires_verification is False
    assert result.non_factual_count == 1
    assert result.contradicted_count == 0
    assert result.unknown_count == 0


# ---------------------------------------------------------------------------
# Test 8: answer-level aggregation counts + backward-compatible max semantics.
# ---------------------------------------------------------------------------
def test_answer_level_aggregation_counts_and_max_preserved(monkeypatch):
    claims = [
        (CLAIM, (0, len(CLAIM))),
        ("Java is free.", (30, 42)),
        ("Java is a drink.", (43, 59)),
    ]
    detector = make_detector(
        monkeypatch,
        claims=claims,
        classify=lambda c, e: {
            CLAIM: mapping(0.05, 0.90, 0.05),
            "Java is free.": mapping(0.90, 0.05, 0.05),
            "Java is a drink.": mapping(0.10, 0.10, 0.80),
        }[c],
        evidence_stub=EvidenceStub(
            snippets={
                CLAIM: ["Java was created by James Gosling."],
                "Java is free.": ["Java is free forever."],
                "Java is a drink.": ["An article about coffee."],
            }
        ),
    )
    result = detector.detect(draft_answer="Java is a drink. Java is free.", evidence=EVIDENCE)
    assert result.claim_count == 3
    assert result.supported_count == 1
    assert result.contradicted_count == 1
    assert result.unknown_count == 1
    assert result.probability == pytest.approx(0.95)
    assert result.label == "HALLUCINATION"
    assert result.requires_verification is True
    assert len(result.sentences) == 3


# ---------------------------------------------------------------------------
# Evidence module: sentence fallback when shared decomposition is unavailable
# and lexical fallback when the shared stack cannot be loaded.
# ---------------------------------------------------------------------------
def test_atomic_claims_fall_back_to_sentence_spans(monkeypatch):
    stub = EvidenceStub(decomposed=[], snippets={})
    detector = make_detector(monkeypatch, evidence_stub=stub)
    spans = detector._atomic_claims("First claim here. Second claim there.")
    assert len(spans) == 2
    assert "First claim here." in spans[0][0]
    assert spans[0][1] == (0, len("First claim here."))

def test_select_evidence_falls_back_to_lexical_selection(monkeypatch):
    import halluciguard_detector.evidence as evidence_module

    monkeypatch.setattr(evidence_module, "_ensure_verifier", lambda: False)
    snippets = select_evidence(
        "Java was created by James Gosling",
        ["The Java language was created by James Gosling in 1995.", "Irrelevant text."],
        rerank_top=1,
    )
    assert snippets and "James Gosling" in snippets[0]