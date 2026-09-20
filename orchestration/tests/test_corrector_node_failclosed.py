"""Orchestration-level tests for the Corrector node's fail-closed routing.

When the Corrector genuinely cannot regenerate a usable answer it must NOT feed
the unchanged contradicted answer back into the Re-Verifier on a loop it cannot
win. Instead the node applies the configured fail policy:

  * default            -> route "human_escalation"
  * HG_CORRECTOR_FAIL_MODE=reject -> route "reject"

A successful regeneration still routes to "reverifier".
"""
from __future__ import annotations

import os
import sys

import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import orchestration.graph as graph
from orchestration.schemas import CorrectionResult, ExecutionStatus, ValidationStatus


def _base_state() -> dict:
    return {
        "execution_id": "exec-1",
        "request_id": "req-1",
        "user_query": "Who created Java?",
        "llm_response": "Java was created by Dennis Ritchie in 1972.",
        "domain": "general",
        "correction_attempt_count": 0,
        "verifier_result": {
            "claim_reports": [
                {
                    "claim_id": "c1",
                    "claim_text": "Java was created by Dennis Ritchie in 1972.",
                    "verdict": "contradicted",
                    "support_score": 0.05,
                    "contradiction_score": 0.95,
                    "confidence_score": 0.9,
                    "evidence": [],
                }
            ]
        },
        "inter_agent_bus": [],
        "trace": [],
    }


class _StubRegen:
    def __init__(self, result):
        self._result = result

    async def regenerate(self, request):
        return self._result


def _failed_correction() -> CorrectionResult:
    return CorrectionResult(
        original_text="Java was created by Dennis Ritchie in 1972.",
        corrected_text="",
        changed_claims=[{"claim_id": "c1", "action": "correction_failed", "reason": "echo"}],
        validation_status=ValidationStatus.UNVALIDATED,
        attempt_count=2,
        status=ExecutionStatus.FAILED,
    )


def _good_correction() -> CorrectionResult:
    return CorrectionResult(
        original_text="Java was created by Dennis Ritchie in 1972.",
        corrected_text="Java was developed by James Gosling at Sun Microsystems.",
        changed_claims=[{"claim_id": "c1", "action": "regenerated"}],
        validation_status=ValidationStatus.UNVALIDATED,
        attempt_count=1,
        status=ExecutionStatus.COMPLETED,
    )


@pytest.mark.asyncio
async def test_failed_correction_defaults_to_human_escalation(monkeypatch):
    monkeypatch.delenv("HG_CORRECTOR_FAIL_MODE", raising=False)
    monkeypatch.setattr(graph, "CharacterRegenerator", None, raising=False)
    monkeypatch.setattr(
        "services.character_regenerator.CharacterRegenerator",
        lambda: _StubRegen(_failed_correction()),
    )
    out = await graph._corrector_node(_base_state())
    assert out["route"] == "human_escalation"
    assert out["verification_status"] == "correction_failed"
    assert graph._corrector_route({**out}) == "human_escalation"


@pytest.mark.asyncio
async def test_failed_correction_reject_mode(monkeypatch):
    monkeypatch.setenv("HG_CORRECTOR_FAIL_MODE", "reject")
    monkeypatch.setattr(
        "services.character_regenerator.CharacterRegenerator",
        lambda: _StubRegen(_failed_correction()),
    )
    out = await graph._corrector_node(_base_state())
    assert out["route"] == "reject"
    assert graph._corrector_route({**out}) == "reject"


@pytest.mark.asyncio
async def test_successful_correction_routes_to_reverifier(monkeypatch):
    monkeypatch.delenv("HG_CORRECTOR_FAIL_MODE", raising=False)
    monkeypatch.setattr(
        "services.character_regenerator.CharacterRegenerator",
        lambda: _StubRegen(_good_correction()),
    )
    out = await graph._corrector_node(_base_state())
    assert out["route"] == "reverifier"
    assert out["final_response"] == "Java was developed by James Gosling at Sun Microsystems."
    assert graph._corrector_route({**out}) == "reverifier"
