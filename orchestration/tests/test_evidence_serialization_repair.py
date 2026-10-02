"""Real enum/Pydantic contracts; no network or factual accuracy claims."""
from dataclasses import dataclass
from enum import Enum
import pytest
from orchestration import graph
from orchestration.schemas import Evidence, ClaimReport, VerifierResult
from agents.judge_agent.judge_agent import JudgeAgent


class ProviderLabel(str, Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


@pytest.mark.parametrize("label", list(ProviderLabel))
def test_python_enum_evidence_survives_canonical_conversion(label):
    raw = {"claim_reports": [{"claim_id": "c1", "claim_text": "A factual claim.",
        "verdict": "verified", "evidence": [{"source": "fixture", "snippet": "Evidence.",
        "entailment_label": label, "entailment_score": .99,
        "nli_entailment": .99, "nli_contradiction": .005, "nli_neutral": .005}]}]}
    result = graph._build_canonical_verifier_result(raw, "q", "general")
    assert result.evidence[0].entailment_label == label.value
    assert result.evidence[0].nli_entailment == .99
    assert result.evidence[0].nli_contradiction == .005
    assert result.evidence[0].nli_neutral == .005
    assert graph._has_supporting_evidence(result.claim_reports[0].model_dump()) == (
        label == ProviderLabel.ENTAILMENT)


def test_recursive_dump_serializes_models_dataclasses_and_enum_values():
    @dataclass
    class Envelope:
        evidence: Evidence
        status: ProviderLabel
    ev = Evidence(evidence_id="s", source="fixture", snippet="Evidence.",
                  entailment_label="entailment", entailment_score=.99)
    dumped = graph._dump({"nested": [Envelope(ev, ProviderLabel.ENTAILMENT)]})
    assert dumped["nested"][0]["status"] == "entailment"
    assert dumped["nested"][0]["evidence"]["entailment_label"] == "entailment"


@pytest.mark.parametrize("label", ["EntailmentLabel.ENTAILMENT", ProviderLabel.ENTAILMENT])
def test_supported_evidence_can_reach_memory_gate(label):
    raw = {"claim_reports": [{"claim_id": "c", "claim_text": "Hanoi is the capital of Vietnam.",
        "verdict": "verified", "evidence": [{"source": "fixture", "snippet": "Hanoi is the capital of Vietnam.",
        "entailment_label": label, "entailment_score": .995}]}]}
    report = graph._build_canonical_verifier_result(raw, "q", "general").claim_reports[0]
    assert graph._has_supporting_evidence(graph._dump(report))


def test_neutral_only_score_dominance_cannot_authorize_judge_acceptance():
    ev = Evidence(evidence_id="s", source="fixture", snippet="Unrelated passage.",
                  entailment_label="neutral", entailment_score=.001)
    claim = ClaimReport(claim_id="c", claim_text="Hanoi is the capital of Vietnam.",
        verdict="verified", support_score=.7, contradiction_score=.0, confidence_score=.7, evidence=[ev])
    result = JudgeAgent().evaluate(
        verifier_result=VerifierResult(query_id="q", domain="general", claim_reports=[claim]),
        user_query="What is the capital of Vietnam?", original_response=claim.claim_text,
        domain="general", retry_count=0)
    assert str(getattr(result.decision, "value", result.decision)) != "ACCEPT"
