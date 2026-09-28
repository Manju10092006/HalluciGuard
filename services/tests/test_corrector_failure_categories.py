"""Tests for the Corrector's diagnostic failure categories.

These assert the ``failure_category`` recorded on a fail-closed CorrectionResult
so operators can measure whether correction failures are caused by LLM provider
reliability (LLM_PROVIDER_FAILURE) or by the model echoing the original answer
(MODEL_ECHO). NO_LOCATABLE_CLAIM / NO_OP_MATCH are defined for the sentence-level
targeting path and are not reachable by the whole-answer regenerator on this
branch (see the implementation report).
"""

from __future__ import annotations

import os
import sys

import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from services.character_regenerator import (
    CharacterRegenerator,
    FAIL_LLM_PROVIDER,
    FAIL_MODEL_ECHO,
)
from orchestration.schemas import (
    CorrectionRequest,
    ClaimReport,
    Evidence,
    VerdictLabel,
    EntailmentLabel,
    ExecutionStatus,
)


class _Gen:
    def __init__(self, draft, status="success", error=None, error_code=None, provider_used=None):
        self.draft_response = draft
        self.status = status
        self.error = error
        self.error_code = error_code
        self.provider_used = provider_used


class _FakeService:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = 0

    async def generate(self, **kwargs):
        item = self._scripted[min(self.calls, len(self._scripted) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


ORIGINAL = "Java was created by Dennis Ritchie in 1972."


def _request() -> CorrectionRequest:
    ev = Evidence(
        evidence_id="E1",
        title="Java history",
        source="Sun Microsystems",
        snippet="Java was developed by James Gosling at Sun Microsystems.",
        entailment_label=EntailmentLabel.CONTRADICTION,
        entailment_score=0.95,
        credibility_score=0.95,
    )
    claim = ClaimReport(
        claim_id="c1",
        claim_text=ORIGINAL,
        verdict=VerdictLabel.CONTRADICTED,
        support_score=0.05,
        contradiction_score=0.95,
        confidence_score=0.95,
        evidence=[ev],
    )
    return CorrectionRequest(
        execution_id="exec-1",
        user_query="Who created Java?",
        original_response=ORIGINAL,
        claims_to_correct=[claim],
        claims_to_preserve=[],
        trusted_evidence=[],
        contradictory_evidence=[ev],
        correction_instructions="Repair the contradicted claim using evidence.",
    )


@pytest.mark.asyncio
async def test_provider_failure_is_categorized(monkeypatch):
    monkeypatch.setenv("HG_CORRECTOR_MAX_ATTEMPTS", "2")
    svc = _FakeService(
        [
            _Gen("", status="failed", error="rate limited", error_code="HTTP_429", provider_used=None),
            _Gen("", status="failed", error="server error", error_code="HTTP_500", provider_used=None),
        ]
    )
    res = await CharacterRegenerator(service=svc).regenerate(_request())
    assert res.status == ExecutionStatus.FAILED
    assert res.failure_category == FAIL_LLM_PROVIDER


@pytest.mark.asyncio
async def test_raising_service_is_provider_failure(monkeypatch):
    monkeypatch.setenv("HG_CORRECTOR_MAX_ATTEMPTS", "2")
    svc = _FakeService([RuntimeError("network down"), RuntimeError("still down")])
    res = await CharacterRegenerator(service=svc).regenerate(_request())
    assert res.status == ExecutionStatus.FAILED
    assert res.failure_category == FAIL_LLM_PROVIDER


@pytest.mark.asyncio
async def test_persistent_echo_is_categorized(monkeypatch):
    monkeypatch.setenv("HG_CORRECTOR_MAX_ATTEMPTS", "2")
    svc = _FakeService([_Gen(ORIGINAL, provider_used="groq"), _Gen(ORIGINAL, provider_used="groq")])
    res = await CharacterRegenerator(service=svc).regenerate(_request())
    assert res.status == ExecutionStatus.FAILED
    assert res.failure_category == FAIL_MODEL_ECHO
    assert res.provider_used == "groq"


@pytest.mark.asyncio
async def test_success_has_no_failure_category():
    corrected = "Java was created by James Gosling at Sun Microsystems in the 1990s."
    svc = _FakeService([_Gen(corrected, provider_used="groq")])
    res = await CharacterRegenerator(service=svc).regenerate(_request())
    assert res.status == ExecutionStatus.COMPLETED
    assert res.failure_category is None
    assert res.provider_used == "groq"
