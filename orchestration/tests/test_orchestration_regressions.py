from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from orchestration.graph import _memory_node, _reverifier_node, _verifier_node
from orchestration.schemas import ClaimReport, CorrectionRequest, Evidence, VerdictLabel
from services.character_regenerator import CharacterRegenerator


@dataclass
class _RawVerifierOutput:
    query_id: str
    domain: str
    claim_evidence: list[dict]
    overall_evidence_confidence: float


def _state(**updates):
    state = {
        "execution_id": "exec-1",
        "request_id": "req-1",
        "user_query": "Tell me about Java.",
        "llm_response": "Java was created in 1972. Java runs on the JVM.",
        "draft_response": "Java was created in 1972. Java runs on the JVM.",
        "domain": "general",
        "trace": [],
        "errors": [],
        "inter_agent_bus": [],
        "retry_count": 0,
        "max_retries": 2,
    }
    state.update(updates)
    return state


@pytest.mark.asyncio
async def test_verifier_receives_detector_atomic_claims(monkeypatch):
    captured = None

    class Pipeline:
        async def verify(self, payload):
            nonlocal captured
            captured = payload
            return _RawVerifierOutput(
                query_id=payload.query_id,
                domain=payload.domain,
                claim_evidence=[],
                overall_evidence_confidence=0.0,
            )

    class Suspicious:
        def __init__(self, claim_id, text):
            self.claim_id = claim_id
            self.text = text

    class Payload:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(
        "orchestration.graph._get_verifier_imports", lambda: (Pipeline, Suspicious, Payload)
    )
    state = _state(
        detected_claims=[
            {"claim_id": "c1", "text": "Java was created in 1972."},
            {"claim_id": "c2", "text": "Java runs on the JVM."},
        ]
    )

    await _verifier_node(state)

    assert [claim.text for claim in captured.suspicious_claims] == [
        "Java was created in 1972.",
        "Java runs on the JVM.",
    ]


@pytest.mark.asyncio
async def test_reverifier_does_not_pass_unverified_regeneration(monkeypatch):
    class Pipeline:
        async def verify(self, payload):
            return _RawVerifierOutput(
                query_id=payload.query_id,
                domain=payload.domain,
                claim_evidence=[
                    {
                        "claim_id": "rev-1",
                        "claim_text": payload.suspicious_claims[0].text,
                        "verdict": "unverified",
                        "confidence_score": 0.1,
                        "evidence": [],
                    }
                ],
                overall_evidence_confidence=0.1,
            )

    class Suspicious:
        def __init__(self, claim_id, text):
            self.claim_id = claim_id
            self.text = text

    class Payload:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(
        "orchestration.graph._get_verifier_imports", lambda: (Pipeline, Suspicious, Payload)
    )
    result = await _reverifier_node(
        _state(
            correction_result={
                "corrected_text": "Java was developed by a team at Sun Microsystems."
            },
            final_response="Java was developed by a team at Sun Microsystems.",
        )
    )

    assert result["reverification_result"]["passed"] is False
    assert result["reverification_result"]["remaining_contradictions"] == 0


@pytest.mark.asyncio
async def test_memory_rejects_unverified_claim_even_after_accept(monkeypatch):
    calls = []

    class Memory:
        async def initialize(self):
            pass

        async def close(self):
            pass

        async def store_fact(self, request):
            calls.append(request)

    monkeypatch.setattr("agents.memory_agent.memory.memory_agent.MemoryAgent", Memory)
    result = await _memory_node(
        _state(
            judge_decision="ACCEPT",
            verifier_result={
                "claim_reports": [
                    {
                        "claim_id": "c1",
                        "claim_text": "Kushal is the topper of KMIT.",
                        "verdict": "unverified",
                        "evidence": [],
                    }
                ]
            },
        )
    )

    assert calls == []
    assert result["memory_result"]["status"] == "skipped"


@pytest.mark.asyncio
async def test_character_agent_regenerates_but_marks_candidate_unvalidated():
    class Service:
        async def generate(self, **kwargs):
            assert "claims_requiring_correction" in kwargs["user_query"]
            assert "Dennis Ritchie" in kwargs["user_query"]
            return SimpleNamespace(
                status="success",
                draft_response=(
                    "Java was developed by James Gosling and his team at Sun Microsystems."
                ),
                error=None,
                error_code=None,
            )

    evidence = Evidence(
        evidence_id="e1",
        source="Oracle",
        snippet="Java was developed by James Gosling and his team at Sun Microsystems.",
        entailment_label="contradiction",
        entailment_score=0.99,
    )
    claim = ClaimReport(
        claim_id="c1",
        claim_text="Java was created by Dennis Ritchie in 1972.",
        verdict=VerdictLabel.CONTRADICTED,
        contradiction_score=0.99,
        confidence_score=0.99,
        evidence=[evidence],
    )
    request = CorrectionRequest(
        execution_id="exec-1",
        user_query="Who created Java?",
        original_response="Java was created by Dennis Ritchie in 1972.",
        claims_to_correct=[claim],
        contradictory_evidence=[evidence],
        correction_instructions="Regenerate using the evidence.",
    )

    result = await CharacterRegenerator(Service()).regenerate(request)

    assert "James Gosling" in result.corrected_text
    assert str(result.validation_status) == "unvalidated"
    assert result.corrected_text != result.original_text
