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
    )
    grounded = SimpleNamespace(
        sentences=[sentence],
        probability=0.91,
        risk=RiskLevel.HIGH,
        requires_verification=True,
        model_version="detector-best",
        warnings=[],
    )
    fake_detector = SimpleNamespace(detect=lambda **kwargs: grounded, threshold=0.5)
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


@pytest.mark.parametrize("probabilities,threshold,expected", [
    ([0.45, 0.30, 0.25], 0.5, True),
    ([0.30, 0.40, 0.30], 0.8, False),
])
def test_binary_triage_can_disagree_with_three_class_argmax(monkeypatch, probabilities, threshold, expected):
    labels = [ClaimLabel.SUPPORTED, ClaimLabel.CONTRADICTED, ClaimLabel.NOT_ENOUGH_INFO]
    sentence = SimpleNamespace(text='A claim.', start=0, end=8,
        hallucination_probability=sum(probabilities[1:]),
        label=labels[max(range(3), key=probabilities.__getitem__)],
        probabilities=dict(zip(labels, probabilities)), risk=RiskLevel.HIGH,
        evidence_snippets=['Evidence.'])
    grounded = SimpleNamespace(sentences=[sentence], probability=sentence.hallucination_probability,
        risk=RiskLevel.HIGH, requires_verification=expected, model_version='test', warnings=[])
    agent = DetectorAgent(model_path='unused')
    monkeypatch.setattr(agent, '_get_detector', lambda: SimpleNamespace(detect=lambda **kwargs: grounded, threshold=threshold))
    result = agent.detect('Question', 'A claim.', evidence=['Evidence.'])
    assert result['claims'][0]['requires_verification'] is expected
    assert result['claims'][0]['label'] == sentence.label.value
