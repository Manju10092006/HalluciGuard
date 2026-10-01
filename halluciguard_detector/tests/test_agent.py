from __future__ import annotations

from types import SimpleNamespace

import pytest

from halluciguard_detector.agent import DetectorAgent
from halluciguard_detector.schemas import ClaimLabel, RiskLevel


def test_pre_verification_triage_extracts_claims_without_loading_model(monkeypatch):
    agent = DetectorAgent(model_path="unused")
    monkeypatch.setattr(agent, "_get_detector", lambda: (_ for _ in ()).throw(AssertionError("must not load")))
    result = agent.detect("Who created Java?", "Java was created by Snehith in 1995.")
    assert result["next_action"] == "Verify"
    assert result["hallucination_probability"] is None
    assert result["grounded"] is False
    assert result["claims"][0]["label"] == "UNVERIFIED"


def test_pre_verification_does_not_fabricate_any_probability(monkeypatch):
    """Pre-verification is triage only: no NLI output may be invented."""
    agent = DetectorAgent(model_path="unused")
    monkeypatch.setattr(agent, "_get_detector", lambda: (_ for _ in ()).throw(AssertionError("must not load")))
    result = agent.detect("Who created Java?", "Java was created by Snehith in 1995.")
    assert result["grounded"] is False
    assert result["inference_executed"] is False
    assert result["model_loaded"] is False
    assert result["calibration_applied"] is False
    # Both the canonical and the deprecated score stay absent, not zero.
    assert result["verification_risk"] is None
    assert result["hallucination_probability"] is None
    assert result["contradiction_mass"] is None
    assert result["next_action"] == "Verify"
    for claim in result["claims"]:
        assert claim["label"] == "UNVERIFIED"
        assert claim["probabilities"] == {}
        assert claim["claim_risk"] is None
        assert claim["requires_verification"] is True
        assert claim["evidence_snippets"] == []


def test_grounded_adapter_preserves_sentence_predictions(monkeypatch):
    sentence = SimpleNamespace(
        text="Java was created by Snehith.",
        start=0,
        end=30,
        hallucination_probability=0.91,
        label=ClaimLabel.CONTRADICTED,
        probabilities={
            ClaimLabel.SUPPORTED: 0.05,
            ClaimLabel.CONTRADICTED: 0.90,
            ClaimLabel.NOT_ENOUGH_INFO: 0.05,
        },
        risk=RiskLevel.HIGH,
        evidence_snippets=["Java was created by James Gosling."],
        model_input_evidence="Java was created by James Gosling.",
    )
    grounded = SimpleNamespace(
        sentences=[sentence],
        probability=0.91,
        risk=RiskLevel.HIGH,
        requires_verification=True,
        model_version="detector-best",
        warnings=[],
    )
    fake_detector = SimpleNamespace(detect=lambda **kwargs: grounded)
    agent = DetectorAgent(model_path="unused")
    monkeypatch.setattr(agent, "_get_detector", lambda: fake_detector)
    result = agent.detect(
        "Who created Java?",
        "Java was created by Snehith.",
        evidence=["Java was created by James Gosling."],
    )
    assert result["grounded"] is True
    assert result["hallucination_probability"] == 0.91
    assert result["claims"][0]["label"] == "CONTRADICTED"
    assert result["claims"][0]["requires_verification"] is True


def _grounded_response(**overrides):
    """A grounded result whose three classes are deliberately distinguishable."""
    sentence = SimpleNamespace(
        text="Java was created by Snehith.",
        start=0,
        end=30,
        hallucination_probability=0.95,
        verification_risk=0.95,
        label=ClaimLabel.NOT_ENOUGH_INFO,
        probabilities={
            ClaimLabel.SUPPORTED: 0.05,
            ClaimLabel.CONTRADICTED: 0.15,
            ClaimLabel.NOT_ENOUGH_INFO: 0.80,
        },
        supported_probability=0.05,
        contradicted_probability=0.15,
        unknown_probability=0.80,
        non_factual=False,
        risk=RiskLevel.MEDIUM,
        evidence_snippets=["Java was created by James Gosling."],
        model_input_evidence="Java was created by James Gosling.",
    )
    grounded = SimpleNamespace(
        sentences=[sentence],
        probability=0.95,
        verification_risk=0.95,
        contradiction_mass=0.15,
        risk=RiskLevel.HIGH,
        requires_verification=True,
        model_version="detector-best",
        warnings=[],
        claim_count=1,
        supported_count=0,
        contradicted_count=0,
        unknown_count=1,
        non_factual_count=0,
    )
    for key, value in overrides.items():
        setattr(grounded, key, value)
    return grounded


def _grounded_agent(monkeypatch, grounded):
    agent = DetectorAgent(model_path="unused")
    monkeypatch.setattr(
        agent, "_get_detector", lambda: SimpleNamespace(detect=lambda **kwargs: grounded)
    )
    return agent


def test_grounded_adapter_forwards_contradiction_mass(monkeypatch):
    """contradiction_mass is the only refutation signal and must reach consumers."""
    agent = _grounded_agent(monkeypatch, _grounded_response())
    result = agent.detect(
        "Who created Java?",
        "Java was created by Snehith.",
        evidence=["Java was created by James Gosling."],
    )
    assert result["contradiction_mass"] == pytest.approx(0.15)


def test_grounded_adapter_derives_contradiction_mass_when_absent(monkeypatch):
    """An older response object without the field still yields a correct value."""
    grounded = _grounded_response()
    del grounded.contradiction_mass
    agent = _grounded_agent(monkeypatch, grounded)
    result = agent.detect(
        "Who created Java?",
        "Java was created by Snehith.",
        evidence=["Java was created by James Gosling."],
    )
    assert result["contradiction_mass"] == pytest.approx(0.15)


def test_grounded_adapter_keeps_class_probabilities_separate(monkeypatch):
    """unknown_probability must never be folded into contradicted_probability."""
    agent = _grounded_agent(monkeypatch, _grounded_response())
    result = agent.detect(
        "Who created Java?",
        "Java was created by Snehith.",
        evidence=["Java was created by James Gosling."],
    )
    claim = result["claims"][0]
    assert claim["supported_probability"] == pytest.approx(0.05)
    assert claim["contradicted_probability"] == pytest.approx(0.15)
    assert claim["unknown_probability"] == pytest.approx(0.80)
    # The three classes remain distinct, and the triage score is their sum.
    assert claim["verification_risk"] == pytest.approx(0.95)
    # An unverified (unknown) claim still needs verification...
    assert claim["requires_verification"] is True
    # ...but it is not a contradiction.
    assert claim["label"] == "NOT_ENOUGH_INFO"


def test_grounded_probability_semantics_states_the_real_meaning(monkeypatch):
    agent = _grounded_agent(monkeypatch, _grounded_response())
    result = agent.detect(
        "Who created Java?",
        "Java was created by Snehith.",
        evidence=["Java was created by James Gosling."],
    )
    semantics = result["probability_semantics"]
    assert "insufficient evidence" in semantics
    assert "never folded" in semantics
    assert "not a probability the claim is false" in semantics
    assert "Judge" in semantics
    # The deprecated alias is still present for backward compatibility.
    assert result["hallucination_probability"] == result["verification_risk"]


def test_grounded_excludes_non_factual_from_contradiction_mass(monkeypatch):
    """A non-factual claim must not inflate the refutation signal."""
    sentence = SimpleNamespace(
        text="This is the best programming language.",
        start=0,
        end=36,
        hallucination_probability=0.0,
        verification_risk=0.0,
        label=ClaimLabel.NOT_ENOUGH_INFO,
        probabilities={},
        supported_probability=0.0,
        contradicted_probability=0.0,
        unknown_probability=0.0,
        non_factual=True,
        risk=RiskLevel.LOW,
        evidence_snippets=[],
    )
    grounded = _grounded_response()
    grounded.sentences = [sentence]
    grounded.contradiction_mass = 0.0
    grounded.verification_risk = 0.0
    grounded.probability = 0.0
    agent = _grounded_agent(monkeypatch, grounded)
    result = agent.detect(
        "Which language is best?",
        "This is the best programming language.",
        evidence=["Python is a popular language."],
    )
    assert result["claims"][0]["non_factual"] is True
    # No factual classifier call happened; unavailable must differ from zero.
    assert result["contradiction_mass"] is None
    assert result["verification_risk"] is None
    assert result["inference_executed"] is False
