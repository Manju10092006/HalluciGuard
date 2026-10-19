"""Regression tests for detector semantic consistency and evidence selection.

These lock in the fixes for the "remaining issues" pass:

- UNKNOWN is separated from CONTRADICTED semantics at the answer level: an
  all-NOT_ENOUGH_INFO answer is never labeled HALLUCINATION.
- ``verification_risk`` is present and equals the legacy operational score; the
  three class probabilities are kept separate and never folded together.
- Hybrid retrieval + reranking (shared Verifier stack) is used when available,
  reranked on the REAL claim, with a safe deterministic lexical fallback that
  is traceable and never fabricates confidence.
- The entity guard distinguishes a direct contradiction (same relation) from a
  mere entity mismatch (different relation).
- Numeric/date mismatches reinforce contradiction but stay bounded.
- Compound claims are split and each sub-claim is verified independently.
- Non-factual/opinion claims are never treated as factual hallucination.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from helpers import EvidenceStub, make_detector

import halluciguard_detector.evidence as evidence_module
from halluciguard_detector.detector import Detector
from halluciguard_detector.evidence import ClaimEvidenceEngine, select_evidence
from halluciguard_detector.schemas import ClaimLabel
from halluciguard_detector.text import (
    has_entity_conflict,
    numeric_consistency,
    shared_relation,
)

S = ClaimLabel.SUPPORTED
C = ClaimLabel.CONTRADICTED
N = ClaimLabel.NOT_ENOUGH_INFO


# ---------------------------------------------------------------------------
# Canonical supported / contradicted / unknown cases.
# ---------------------------------------------------------------------------
def test_supported_claim(monkeypatch):
    claim = "The Earth orbits the Sun."
    detector = make_detector(
        monkeypatch,
        claims=[(claim, (0, len(claim)))],
        classify=lambda c, e: {S: 0.92, C: 0.04, N: 0.04},
        evidence_stub=EvidenceStub(snippets={claim: ["The Earth revolves around the Sun."]}),
    )
    result = detector.detect(
        draft_answer=claim,
        evidence=["The Earth revolves around the Sun."],
    )
    assert result.sentences[0].label == S
    assert result.supported_count == 1
    assert result.contradicted_count == 0
    assert result.unknown_count == 0
    assert result.contradiction_mass == pytest.approx(0.04)
    assert result.requires_verification is False
    assert result.label == "NO_HALLUCINATION"


def test_contradicted_claim(monkeypatch):
    claim = "The Earth is the center of the solar system."
    detector = make_detector(
        monkeypatch,
        claims=[(claim, (0, len(claim)))],
        classify=lambda c, e: {S: 0.03, C: 0.94, N: 0.03},
        evidence_stub=EvidenceStub(snippets={claim: ["The Earth orbits the Sun."]}),
    )
    result = detector.detect(
        draft_answer=claim,
        evidence=["The Earth orbits the Sun."],
    )
    assert result.sentences[0].label == C
    assert result.contradicted_count == 1
    assert result.contradiction_mass == pytest.approx(0.94)
    assert result.label == "HALLUCINATION"


def test_unknown_claim_with_only_related_evidence(monkeypatch):
    claim = "The company generated $5 billion revenue."
    detector = make_detector(
        monkeypatch,
        claims=[(claim, (0, len(claim)))],
        classify=lambda c, e: {S: 0.06, C: 0.05, N: 0.89},
        evidence_stub=EvidenceStub(snippets={claim: ["The company operates globally."]}),
    )
    result = detector.detect(
        draft_answer=claim,
        evidence=["The company operates globally."],
    )
    assert result.sentences[0].label == N
    assert result.unknown_count == 1
    # Insufficient evidence carries NO refutation signal at all.
    assert result.contradiction_mass == pytest.approx(0.05)
    assert result.contradicted_count == 0
    assert result.requires_verification is True
    assert result.label == "NO_HALLUCINATION"


def test_class_mapping_normalization_keeps_the_sum_to_one_invariant():
    mapping = Detector._normalize_class_mapping({S: 0.2, C: 0.3, N: 0.5})
    assert sum(mapping.values()) == pytest.approx(1.0)
    assert mapping[C] == pytest.approx(0.3)


@pytest.mark.parametrize("raw", [
    {S: 0.2, N: 0.8}, {}, {S: 5.0, C: "nope", N: -2.0},
    {S: float("nan"), C: .5, N: .5}, {S: .1, C: .1, N: .1},
])
def test_invalid_classifier_probabilities_raise_instead_of_implying_support(raw):
    with pytest.raises(ValueError):
        Detector._normalize_class_mapping(raw)


def test_absent_contradiction_probability_cannot_authorize_inference(monkeypatch):
    claim = "The company operates globally."
    detector = make_detector(
        monkeypatch, claims=[(claim, (0, len(claim)))],
        classify=lambda c, e: {S: 0.10, N: 0.90},
        evidence_stub=EvidenceStub(snippets={claim: ["The company operates in 50 countries."]}),
    )
    with pytest.raises(ValueError, match="incomplete_detector"):
        detector.detect(draft_answer=claim, evidence=["The company operates in 50 countries."])


# ---------------------------------------------------------------------------
# Answer level: UNKNOWN is never collapsed into HALLUCINATION.
# ---------------------------------------------------------------------------
def test_unknown_only_answer_is_not_labeled_hallucination(monkeypatch):
    # A single claim the model cannot decide: high unknown, low contradiction.
    claim = "The company generated $5 billion revenue in 2025."
    detector = make_detector(
        monkeypatch,
        claims=[(claim, (0, len(claim)))],
        classify=lambda c, e: {S: 0.10, C: 0.05, N: 0.85},
        evidence_stub=EvidenceStub(snippets={claim: ["The company operates in 50+ countries."]}),
    )
    result = detector.detect(
        draft_answer=claim,
        evidence=["The company operates in more than 50 countries."],
    )
    assert result.sentences[0].label == N
    assert result.sentences[0].unknown_probability == pytest.approx(0.85)
    assert result.sentences[0].contradicted_probability < 0.10
    # Answer level: NOT_ENOUGH_INFO must not trip the HALLUCINATION label.
    assert result.label == "NO_HALLUCINATION"
    # It still needs verification (evidence insufficient), routed to the Judge.
    assert result.requires_verification is True
    assert result.unknown_count == 1
    assert result.contradicted_count == 0
    # An unknown-dominated answer carries no refutation signal whatsoever.
    assert result.contradiction_mass < 0.10
    # The operational triage score is nevertheless high, because the claim is
    # unverified. This is the exact reason it must not be read as falsity.
    assert result.verification_risk > 0.85


def test_verification_risk_present_and_separate_from_contradiction(monkeypatch):
    claim = "Java was created by Snehith in 1995."
    detector = make_detector(
        monkeypatch,
        claims=[(claim, (0, len(claim)))],
        classify=lambda c, e: {S: 0.05, C: 0.90, N: 0.05},
        evidence_stub=EvidenceStub(snippets={claim: ["Java was created by James Gosling in 1995."]}),
    )
    result = detector.detect(
        draft_answer=claim,
        evidence=["Java was created by James Gosling in 1995."],
    )
    # verification_risk is the operational triage score, equal to the legacy
    # probability alias, and equals contradicted+unknown for the top claim.
    assert result.verification_risk == pytest.approx(result.probability)
    top = result.sentences[0]
    assert top.verification_risk == pytest.approx(top.hallucination_probability)
    assert top.verification_risk == pytest.approx(
        top.contradicted_probability + top.unknown_probability
    )
    # A genuinely contradicted claim is labeled HALLUCINATION.
    assert result.label == "HALLUCINATION"
    assert result.contradicted_count == 1


# ---------------------------------------------------------------------------
# Semantic paraphrase should stay SUPPORTED; related-but-insufficient unknown.
# ---------------------------------------------------------------------------
def test_semantic_paraphrase_supported_and_related_insufficient(monkeypatch):
    paraphrase = "Microsoft acquired the company in 2016."
    detector = make_detector(
        monkeypatch,
        claims=[(paraphrase, (0, len(paraphrase)))],
        classify=lambda c, e: {S: 0.88, C: 0.05, N: 0.07},
        evidence_stub=EvidenceStub(
            snippets={paraphrase: ["The firm was purchased by Microsoft during 2016."]}
        ),
    )
    result = detector.detect(
        draft_answer=paraphrase,
        evidence=["The firm was purchased by Microsoft during 2016."],
    )
    assert result.sentences[0].label == S
    assert result.sentences[0].supported_probability == pytest.approx(0.88)


def test_related_but_insufficient_is_unknown_not_supported(monkeypatch):
    claim = "Virat Kohli scored 100 runs."
    detector = make_detector(
        monkeypatch,
        claims=[(claim, (0, len(claim)))],
        classify=lambda c, e: {S: 0.05, C: 0.05, N: 0.90},
        evidence_stub=EvidenceStub(snippets={claim: ["Virat Kohli played in the match."]}),
    )
    result = detector.detect(
        draft_answer=claim,
        evidence=["Virat Kohli played in the match."],
    )
    assert result.sentences[0].label == N
    assert result.contradicted_count == 0
    assert result.unknown_count == 1


# ---------------------------------------------------------------------------
# Issue #7: compound claims are split and each is verified independently.
# ---------------------------------------------------------------------------
def test_compound_claim_is_split_and_verified_independently(monkeypatch):
    answer = "Tesla was founded in 2003 and acquired by Google in 2018."
    decomposed = ["Tesla was founded in 2003.", "Tesla was acquired by Google in 2018."]
    # Evidence supports only the first sub-claim.
    snippets = {
        "Tesla was founded in 2003.": ["Tesla was founded in 2003 by Martin Eberhard."],
        "Tesla was acquired by Google in 2018.": ["Tesla operates electric vehicles."],
    }
    responses = {
        "Tesla was founded in 2003.": {S: 0.9, C: 0.05, N: 0.05},
        "Tesla was acquired by Google in 2018.": {S: 0.05, C: 0.05, N: 0.90},
    }
    seen_claims = []
    detector = make_detector(
        monkeypatch,
        evidence_stub=EvidenceStub(decomposed=decomposed, snippets=snippets),
    )

    def fake_classify(claim, evidence):
        seen_claims.append(claim)
        return dict(responses[claim])

    detector._classify = fake_classify
    result = detector.detect(draft_answer=answer, evidence=["Some evidence."])
    # Both atomic claims were independently routed through NLI.
    assert set(seen_claims) == set(decomposed)
    assert result.claim_count == 2
    labels = {s.text: s.label for s in result.sentences}
    assert labels["Tesla was founded in 2003."] == S
    assert labels["Tesla was acquired by Google in 2018."] == N
    assert result.supported_count == 1
    assert result.unknown_count == 1
    # Mixed supported/unknown is not an automatic HALLUCINATION.
    assert result.label == "NO_HALLUCINATION"
    assert result.requires_verification is True


# ---------------------------------------------------------------------------
# Issue #9: entity guard — direct contradiction vs mere mismatch.
# ---------------------------------------------------------------------------
def test_entity_guard_direct_contradiction_reinforced(monkeypatch):
    claim = "Apple acquired Company A."
    detector = make_detector(
        monkeypatch,
        claims=[(claim, (0, len(claim)))],
        classify=lambda c, e: {S: 0.70, C: 0.10, N: 0.20},
        evidence_stub=EvidenceStub(snippets={claim: ["Apple acquired Company B."]}),
    )
    result = detector.detect(
        draft_answer=claim,
        evidence=["Apple acquired Company B."],
    )
    item = result.sentences[0]
    # Same relation ("acquired") -> contradiction reinforced beyond its floor.
    assert has_entity_conflict(claim, "Apple acquired Company B.") is True
    assert "acquired" in shared_relation(claim, "Apple acquired Company B.")
    assert item.contradicted_probability == pytest.approx(0.10)
    assert item.requires_verification is True
    assert item.contradicted_probability < 0.90
    assert any("shared relation" in w for w in result.warnings)


def test_entity_guard_mere_mismatch_not_auto_contradicted(monkeypatch):
    claim = "Apple works with Company A."
    detector = make_detector(
        monkeypatch,
        claims=[(claim, (0, len(claim)))],
        # Model is mildly uncertain but does not see a contradiction.
        classify=lambda c, e: {S: 0.60, C: 0.15, N: 0.25},
        evidence_stub=EvidenceStub(snippets={claim: ["Apple acquired Company B."]}),
    )
    result = detector.detect(
        draft_answer=claim,
        evidence=["Apple acquired Company B."],
    )
    item = result.sentences[0]
    # Different relation ("works with" vs "acquired"): the entity mismatch alone
    # must not drive a contradiction, and must not become HALLUCINATION.
    assert has_entity_conflict(claim, "Apple acquired Company B.") is True
    assert shared_relation(claim, "Apple acquired Company B.") == set()
    assert item.contradicted_probability == pytest.approx(0.15)  # unchanged
    assert item.label == S  # still SUPPORTED; not auto-contradicted
    assert result.label == "NO_HALLUCINATION"
    assert any("not treated as an automatic contradiction" in w for w in result.warnings)


def test_entity_guard_on_shared_relation_leaves_model_preferring_unknown(monkeypatch):
    # A swapped entity on an identical relation, but the model already says the
    # evidence does not settle the claim. The guard must not manufacture
    # contradiction out of genuine uncertainty.
    claim = "Apple acquired Company A."
    detector = make_detector(
        monkeypatch,
        claims=[(claim, (0, len(claim)))],
        classify=lambda c, e: {S: 0.15, C: 0.15, N: 0.70},
        evidence_stub=EvidenceStub(snippets={claim: ["Apple acquired Company B."]}),
    )
    result = detector.detect(
        draft_answer=claim,
        evidence=["Apple acquired Company B."],
    )
    item = result.sentences[0]
    assert item.contradicted_probability == pytest.approx(0.15)
    assert item.unknown_probability == pytest.approx(0.70)
    assert item.label == N
    assert any("shared relation noted" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Issue #10: numeric and temporal consistency reinforce contradiction.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "claim,evidence,conflicts_with",
    [
        ("The population is 10 million.", "The population is 12 million.", "million"),
        ("The company was founded in 2010.", "The company was founded in 2012.", "year"),
    ],
)
def test_numeric_and_temporal_mismatch_reinforced(monkeypatch, claim, evidence, conflicts_with):
    detector = make_detector(
        monkeypatch,
        claims=[(claim, (0, len(claim)))],
        classify=lambda c, e: {S: 0.80, C: 0.05, N: 0.15},
        evidence_stub=EvidenceStub(snippets={claim: [evidence]}),
    )
    result = detector.detect(draft_answer=claim, evidence=[evidence])
    item = result.sentences[0]
    assert any(conflicts_with in w.lower() for w in result.warnings)
    assert item.contradicted_probability == pytest.approx(0.05)
    assert item.requires_verification is True
    assert item.contradicted_probability < 0.90
    # Remains normalized.
    total = item.supported_probability + item.contradicted_probability + item.unknown_probability
    assert total == pytest.approx(1.0)


def test_numeric_consistency_unit_and_year_rules():
    assert numeric_consistency("population is 10 million", "population is 12 million")
    assert numeric_consistency("founded in 2010", "founded in 2012")
    assert numeric_consistency("founded in 2010", "founded in 2010") == []
    # Unrelated numbers across different units are not a clash.
    assert numeric_consistency("raised 5 million", "grew 20 percent") == []


def test_numeric_mismatch_end_to_end_reaches_contradicted(monkeypatch):
    # A soft NLI model that is only mildly unsure must still be corrected into
    # CONTRADICTED when the evidence states a different, concrete number.
    claim = "The city has a population of 10 million."
    detector = make_detector(
        monkeypatch,
        claims=[(claim, (0, len(claim)))],
        classify=lambda c, e: {S: 0.55, C: 0.30, N: 0.15},
        evidence_stub=EvidenceStub(snippets={claim: ["The city has a population of 12 million."]}),
    )
    result = detector.detect(
        draft_answer=claim,
        evidence=["The city has a population of 12 million."],
    )
    assert result.sentences[0].label == S
    assert result.sentences[0].contradicted_probability == pytest.approx(0.30)
    assert result.contradicted_count == 0
    assert result.label == "NO_HALLUCINATION"
    assert result.requires_verification is True


# ---------------------------------------------------------------------------
# Issue #6: a fallback is traceable, never silently hidden.
# ---------------------------------------------------------------------------
def test_lexical_fallback_is_reported_as_degraded(monkeypatch):
    monkeypatch.setattr(evidence_module, "_ensure_verifier", lambda: False)
    claim = "The Earth orbits the Sun."
    detector = make_detector(
        monkeypatch,
        claims=[(claim, (0, len(claim)))],
        classify=lambda c, e: {S: 0.9, C: 0.05, N: 0.05},
    )
    detector.evidence = ClaimEvidenceEngine(rerank_top=1)
    result = detector.detect(
        draft_answer=claim,
        evidence=["The Earth revolves around the Sun."],
    )
    item = result.sentences[0]
    # Real lexical evidence was used...
    assert item.evidence_snippets
    # ...and the degradation is reported instead of hidden.
    assert item.evidence_degraded is True
    assert item.evidence_route == "lexical"
    assert result.evidence_degraded is True
    assert any("Evidence selection degraded" in w for w in result.warnings)
    # Crucially, no confidence was fabricated by the fallback itself: the label
    # still comes from the model.
    assert item.label == S
    assert item.supported_probability == pytest.approx(0.9)


def test_hybrid_route_is_not_marked_degraded(monkeypatch):
    trace: dict = {}
    _install_fake_shared_stack(monkeypatch, _FakeRetriever(), _FakeReranker())
    ClaimEvidenceEngine(rerank_top=1).select(
        "The Earth orbits the Sun.", ["The Earth revolves around the Sun."], trace=trace
    )
    assert trace["route"] == "hybrid"
    assert trace["degraded"] is False


def test_reranker_empty_output_keeps_hybrid_order_as_traceable_fallback(monkeypatch):
    class _EmptyReranker:
        def rerank(self, query, passages, k=3):
            return []

    _install_fake_shared_stack(monkeypatch, _FakeRetriever(), _EmptyReranker())
    trace: dict = {}
    snippets = ClaimEvidenceEngine(rerank_top=1).select(
        "The Earth orbits the Sun.", ["The Earth revolves around the Sun."], trace=trace
    )
    assert snippets == ["The Earth revolves around the Sun."]
    assert trace["route"] == "hybrid_pre_rerank"
    assert trace["degraded"] is True
    assert "reranker" in trace["reason"].lower()


# ---------------------------------------------------------------------------
# Issue #11: non-factual / opinion claims are never factual hallucination.
# ---------------------------------------------------------------------------
def test_opinion_claim_is_non_factual_not_hallucination(monkeypatch):
    claim = "This is the best programming language."
    detector = make_detector(
        monkeypatch,
        claims=[(claim, (0, len(claim)))],
        evidence_stub=EvidenceStub(checkable=False, snippets={}),
    )
    result = detector.detect(draft_answer=claim, evidence=["Some evidence."])
    item = result.sentences[0]
    assert item.non_factual is True
    assert item.label == N
    assert item.contradicted_probability == 0.0
    assert item.verification_risk == 0.0
    assert result.non_factual_count == 1
    assert result.contradicted_count == 0
    # An all-opinion answer is not labeled hallucinated.
    assert result.label == "NO_HALLUCINATION"


# ---------------------------------------------------------------------------
# Issue #5/6: hybrid retrieval + rerank on the real claim, with lexical fallback.
# ---------------------------------------------------------------------------
class _FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, passages, k=5, dense_model=None):
        self.calls.append(("retrieve", query, k, dense_model))
        return list(passages)[:k]

    def diagnostics(self):
        return {"route": "hybrid", "degraded": False}


class _FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, passages, k=3):
        # Record the exact query the reranker received to prove it is the claim.
        self.calls.append(("rerank", query, k))
        return list(passages)[:k]

    def diagnostics(self):
        return {"inference_executed": True, "degraded": False}


def _install_fake_shared_stack(monkeypatch, retriever, reranker):
    monkeypatch.setattr(evidence_module, "_ensure_verifier", lambda: True)
    monkeypatch.setattr(evidence_module, "_hybrid_retriever", retriever)
    monkeypatch.setattr(evidence_module, "_reranker", reranker)
    monkeypatch.setattr(
        evidence_module, "_passage_cls", lambda **kwargs: SimpleNamespace(**kwargs)
    )


def test_hybrid_path_uses_shared_stack_and_reranks_on_real_claim(monkeypatch):
    retriever = _FakeRetriever()
    reranker = _FakeReranker()
    _install_fake_shared_stack(monkeypatch, retriever, reranker)
    claim = "The Earth orbits the Sun."
    engine = ClaimEvidenceEngine(pool_k=4, rerank_top=2)
    snippets = engine.select(claim, ["The Earth revolves around the Sun.", "Unrelated."])
    # Hybrid retrieval was invoked.
    assert any(c[0] == "retrieve" for c in retriever.calls)
    # Reranker received the REAL claim (not a rewritten query).
    rerank_calls = [c for c in reranker.calls if c[0] == "rerank"]
    assert rerank_calls, "reranker must be called"
    assert rerank_calls[0][1] == claim
    # Returned evidence is capped by rerank_top.
    assert len(snippets) <= 2


def test_retrieval_config_is_honest(monkeypatch):
    """No dead knobs: pool_k is the retriever's k, rerank_top caps the output.

    The shared HybridRetriever exposes a single fused top-k cap and derives its
    own per-backend candidate window, so a separate ``candidate_k`` would be
    configuration that silently does nothing. Guard against it returning.
    """
    assert not hasattr(ClaimEvidenceEngine, "candidate_k")
    assert "candidate_k" not in inspect.signature(ClaimEvidenceEngine).parameters
    assert "candidate_k" not in inspect.signature(select_evidence).parameters

    retriever = _FakeRetriever()
    reranker = _FakeReranker()
    _install_fake_shared_stack(monkeypatch, retriever, reranker)
    engine = ClaimEvidenceEngine(pool_k=7, rerank_top=2)
    engine.select("The Earth orbits the Sun.", ["a", "b", "c", "d", "e", "f", "g", "h"])
    # pool_k is actually forwarded as the retriever's k, not silently ignored.
    assert retriever.calls[0][2] == 7
    assert reranker.calls[0][2] == 2


def test_lexical_fallback_when_shared_stack_unavailable(monkeypatch):
    monkeypatch.setattr(evidence_module, "_ensure_verifier", lambda: False)
    claim = "The Earth orbits the Sun."
    engine = ClaimEvidenceEngine(rerank_top=1)
    snippets = engine.select(claim, ["The Earth revolves around the Sun.", "Unrelated."])
    # Falls back to deterministic lexical selection; no crash, real snippets.
    assert snippets and "revolves" in snippets[0]


def test_untraced_legacy_adapter_is_reported_as_degraded(monkeypatch):
    class _LegacyEvidence:
        """An evidence adapter predating the trace seam (no ``trace`` kwarg)."""

        def decompose(self, text):
            return []

        def checkable(self, text):
            return True

        def select(self, claim, evidence_texts):
            return list(evidence_texts)

    claim = "The Earth orbits the Sun."
    detector = make_detector(
        monkeypatch,
        claims=[(claim, (0, len(claim)))],
        classify=lambda c, e: {S: 0.9, C: 0.05, N: 0.05},
        evidence_stub=_LegacyEvidence(),
    )
    result = detector.detect(
        draft_answer=claim,
        evidence=["The Earth revolves around the Sun."],
    )
    item = result.sentences[0]
    # Selection still worked, but it is not silently treated as a clean run.
    assert item.evidence_snippets
    assert item.evidence_degraded is True
    assert item.evidence_route == "untraced"
    assert item.label == S


def test_select_evidence_returns_empty_for_no_claim_or_no_evidence():
    assert select_evidence("", ["something"]) == []
    assert select_evidence("a claim", []) == []


def test_fallback_evidence_still_drives_nli_not_fake_confidence(monkeypatch):
    # When the shared stack is unavailable, the detector must still classify
    # using lexical evidence (real, traceable snippets), not fabricate numbers.
    monkeypatch.setattr(evidence_module, "_ensure_verifier", lambda: False)
    engine = ClaimEvidenceEngine()
    snippet = engine.select(
        "The Earth orbits the Sun.", ["The Earth revolves around the Sun."]
    )[0]
    claim = "The Earth orbits the Sun."
    detector = make_detector(
        monkeypatch,
        claims=[(claim, (0, len(claim)))],
        classify=lambda c, e: {S: 0.9, C: 0.05, N: 0.05},
        evidence_stub=EvidenceStub(snippets={claim: [snippet]}),
    )
    result = detector.detect(
        draft_answer=claim, evidence=["The Earth revolves around the Sun."]
    )
    assert result.sentences[0].evidence_snippets == [snippet]
    assert result.sentences[0].label == S
