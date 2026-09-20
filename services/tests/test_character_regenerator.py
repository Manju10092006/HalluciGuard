"""Regression tests for the Corrector Agent (CharacterRegenerator).

The Corrector is the correction-and-regeneration controller: given the Judge's
structured feedback (false claims + evidence + facts to preserve) it prompts the
base LLM to regenerate a COMPLETE corrected answer and hands that back as an
UNVALIDATED candidate for the Re-Verifier.

These tests pin the fail-closed contract the agent used to violate by raising:
  1. A good regeneration returns COMPLETED / UNVALIDATED with the new text.
  2. A transient empty response is RETRIED, not crashed, and can still succeed.
  3. An unchanged echo is RETRIED, and if every attempt echoes -> FAILED (closed),
     original preserved, NO fabricated correction, and NO exception raised.
  4. A raising LLM service is caught and fails closed, never propagating.
"""
from __future__ import annotations

import os
import sys

import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from services.character_regenerator import CharacterRegenerator
from orchestration.schemas import (
    CorrectionRequest,
    ClaimReport,
    Evidence,
    VerdictLabel,
    EntailmentLabel,
    ExecutionStatus,
    ValidationStatus,
)


class _Gen:
    """Minimal stand-in for GenerationResult (only the fields regenerate reads)."""

    def __init__(self, draft: str, status: str = "success", error: str | None = None):
        self.draft_response = draft
        self.status = status
        self.error = error
        self.error_code = None


class _FakeService:
    """Injectable BaseLLMService: yields a scripted sequence of outcomes."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = 0

    async def generate(self, **kwargs):
        item = self._scripted[min(self.calls, len(self._scripted) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


def _request() -> CorrectionRequest:
    ev = Evidence(
        evidence_id="E1",
        title="Java history",
        source="Sun Microsystems",
        snippet="Java was developed by James Gosling and his team at Sun Microsystems.",
        entailment_label=EntailmentLabel.CONTRADICTION,
        entailment_score=0.95,
        credibility_score=0.95,
    )
    claim = ClaimReport(
        claim_id="c1",
        claim_text="Java was created by Dennis Ritchie in 1972.",
        verdict=VerdictLabel.CONTRADICTED,
        support_score=0.05,
        contradiction_score=0.95,
        confidence_score=0.95,
        evidence=[ev],
    )
    return CorrectionRequest(
        execution_id="exec-1",
        user_query="Who created Java?",
        original_response="Java was created by Dennis Ritchie in 1972.",
        claims_to_correct=[claim],
        claims_to_preserve=[],
        trusted_evidence=[],
        contradictory_evidence=[ev],
        correction_instructions="Repair the contradicted claim using evidence.",
    )


CORRECTED = "Java was developed by James Gosling and his team at Sun Microsystems in the early 1990s."


@pytest.mark.asyncio
async def test_good_regeneration_returns_unvalidated_candidate():
    svc = _FakeService([_Gen(CORRECTED)])
    res = await CharacterRegenerator(service=svc).regenerate(_request())
    assert res.status == ExecutionStatus.COMPLETED
    assert res.validation_status == ValidationStatus.UNVALIDATED
    assert res.corrected_text == CORRECTED
    assert svc.calls == 1
    assert res.changed_claims[0]["action"] == "regenerated"


@pytest.mark.asyncio
async def test_transient_empty_is_retried_then_succeeds():
    svc = _FakeService([_Gen("", status="failed", error="empty"), _Gen(CORRECTED)])
    res = await CharacterRegenerator(service=svc).regenerate(_request())
    assert res.status == ExecutionStatus.COMPLETED
    assert res.corrected_text == CORRECTED
    assert svc.calls == 2  # first attempt retried, second succeeded


@pytest.mark.asyncio
async def test_persistent_echo_fails_closed_without_raising(monkeypatch):
    monkeypatch.setenv("HG_CORRECTOR_MAX_ATTEMPTS", "2")
    original = "Java was created by Dennis Ritchie in 1972."
    svc = _FakeService([_Gen(original), _Gen(original)])
    res = await CharacterRegenerator(service=svc).regenerate(_request())
    # Failed closed: no fabricated correction, original preserved, no exception.
    assert res.status == ExecutionStatus.FAILED
    assert res.corrected_text == ""
    assert res.original_text == original
    assert svc.calls == 2
    assert res.changed_claims[0]["action"] == "correction_failed"


@pytest.mark.asyncio
async def test_raising_service_is_caught_and_fails_closed():
    svc = _FakeService([RuntimeError("network down"), RuntimeError("still down")])
    res = await CharacterRegenerator(service=svc).regenerate(_request())
    assert res.status == ExecutionStatus.FAILED
    assert res.corrected_text == ""
    assert "generation_error" in res.changed_claims[0]["reason"]
