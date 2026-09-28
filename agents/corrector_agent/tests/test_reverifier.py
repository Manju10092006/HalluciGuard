"""Tests for the ReVerifier stage (independent post-correction verification).

Covers the contract:
  * RE-EXTRACTS claims from the CORRECTED answer (not just the flagged claim).
  * Catches a secondary hallucination introduced by the Corrector (the Java-year
    case) even when the originally-flagged claim is now correct.
  * SUPPORTED / CONTRADICTED / UNKNOWN verdict model; UNKNOWN is never a failure.
  * Output shape: VERIFIED{claims, overall_confidence} / FAILED{failed_claims}.
  * The reverification event does NOT overwrite the original verifier verdicts.
  * Orchestrator routes a FAILED reverification back to correction.

Everything is deterministic and offline — no model, no network, no API key. The
LLM (model_client) and the pre-correction Judge are replaced with doubles.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path

# The repo root carries a Gradio ``app.py`` that shadows ``corrector_agent/app``
# whenever the root is on sys.path (which the repo-root pytest config requires).
# Bind the ``app`` name to the corrector's app package explicitly, before any
# ``from app.* import``, so this test collects under ``python -m pytest`` from the
# repo root without altering the app package on disk.
AGENT_DIR = str(Path(__file__).resolve().parents[1])
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)
_APP_DIR = os.path.join(AGENT_DIR, "app")
if "app" not in sys.modules or not hasattr(sys.modules["app"], "__path__"):
    _pkg = types.ModuleType("app")
    _pkg.__path__ = [_APP_DIR]  # make ``app`` a package rooted at corrector_agent/app
    sys.modules["app"] = _pkg

import pytest

from app.models import (
    AtomicClaim,
    ClaimStatus,
    CorrectionPlan,
    EvidencePassage,
    JudgeVerificationPayload,
    JudgeVerificationResult,
)
from app.planner import CorrectionPlanner
from app.reverifier import (
    CONTRADICTED,
    SUPPORTED,
    UNKNOWN,
    ReVerifier,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _java_payload() -> JudgeVerificationPayload:
    """The contract's motivating case: evidence says James Gosling, 1995."""
    return JudgeVerificationPayload(
        query="Who created Java and when?",
        originalResponse="Java was created by Snehith.",
        claims=[
            AtomicClaim(
                id="c1",
                text="Java was created by Snehith.",
                status=ClaimStatus.HALLUCINATED,
                confidenceScore=0.9,
                evidenceIds=["ev1"],
            )
        ],
        supportingEvidence=[
            EvidencePassage(
                id="ev1",
                sourceTitle="Java (programming language)",
                passageText="Java was created by James Gosling and first released in 1995.",
            )
        ],
        trustScore=0.4,
        correctionInstructions="",
    )


def _plan(payload: JudgeVerificationPayload) -> CorrectionPlan:
    return CorrectionPlanner().planCorrection(payload)


# ---------------------------------------------------------------------------
# Re-extraction
# ---------------------------------------------------------------------------

def test_reextracts_all_sentences_from_corrected_answer():
    """The ReVerifier inspects EVERY sentence of the corrected answer, not just
    the one originally-flagged claim."""
    rv = ReVerifier()
    payload = _java_payload()
    corrected = "Java was created by James Gosling. It was first released in 1995."
    result = rv.reverify(payload, _plan(payload), corrected)
    assert len(result.claims) == 2  # both sentences re-extracted


# ---------------------------------------------------------------------------
# The secondary-hallucination catch (the whole point)
# ---------------------------------------------------------------------------

def test_catches_new_wrong_year_even_though_creator_is_fixed():
    """Corrector fixed the creator (SUPPORTED) but invented '1991' (evidence says
    1995). The ReVerifier must FAIL on the year."""
    rv = ReVerifier()
    payload = _java_payload()
    corrected = "Java was created by James Gosling in 1991."
    result = rv.reverify(payload, _plan(payload), corrected)

    assert result.status == "FAILED"
    contract = result.to_contract()
    assert contract["status"] == "FAILED"
    assert len(contract["failed_claims"]) == 1
    assert "1991" in contract["failed_claims"][0]["text"]
    assert contract["failed_claims"][0]["verdict"] == CONTRADICTED


def test_clean_correction_with_correct_year_is_verified():
    rv = ReVerifier()
    payload = _java_payload()
    corrected = "Java was created by James Gosling in 1995."
    result = rv.reverify(payload, _plan(payload), corrected)

    assert result.status == "VERIFIED"
    contract = result.to_contract()
    assert contract["status"] == "VERIFIED"
    assert contract["overall_confidence"] > 0.0
    assert all(c["verdict"] == SUPPORTED for c in contract["claims"])


def test_correction_with_no_added_number_is_supported():
    rv = ReVerifier()
    payload = _java_payload()
    corrected = "Java was created by James Gosling."
    result = rv.reverify(payload, _plan(payload), corrected)
    assert result.status == "VERIFIED"
    assert result.claims[0].verdict == SUPPORTED


# ---------------------------------------------------------------------------
# UNKNOWN is not falsity
# ---------------------------------------------------------------------------

def test_unknown_claim_is_not_treated_as_failure():
    """A claim with no evidence either way is UNKNOWN, and UNKNOWN must not fail
    the answer."""
    rv = ReVerifier()
    payload = JudgeVerificationPayload(
        query="What is the meaning of life?",
        originalResponse="The meaning of life is 42.",
        claims=[
            AtomicClaim(
                id="c1",
                text="The meaning of life is 42.",
                status=ClaimStatus.INSUFFICIENT_EVIDENCE,
                confidenceScore=0.5,
                evidenceIds=[],
            )
        ],
        supportingEvidence=[],
        trustScore=0.3,
        correctionInstructions="",
    )
    corrected = "Current evidence is insufficient to support this claim."
    result = rv.reverify(payload, _plan(payload), corrected)

    assert result.status == "VERIFIED"  # UNKNOWN != false
    assert all(c.verdict == UNKNOWN for c in result.claims)


def test_contradiction_evidence_triggers_failure():
    rv = ReVerifier()
    payload = JudgeVerificationPayload(
        query="Is the earth flat?",
        originalResponse="The earth is flat.",
        claims=[
            AtomicClaim(
                id="c1",
                text="The earth is flat.",
                status=ClaimStatus.CONTRADICTED,
                confidenceScore=0.9,
                evidenceIds=["cev1"],
            )
        ],
        supportingEvidence=[],
        contradictionEvidence=[
            EvidencePassage(
                id="cev1",
                sourceTitle="Geodesy",
                passageText="The earth is an oblate spheroid, not flat.",
            )
        ],
        trustScore=0.3,
        correctionInstructions="",
    )
    # The Corrector "repair" that stupidly re-asserts the flat-earth claim.
    corrected = "The earth is flat and not round."
    result = rv.reverify(payload, _plan(payload), corrected)
    assert result.status == "FAILED"
    assert any(c.verdict == CONTRADICTED for c in result.claims)


# ---------------------------------------------------------------------------
# Separate event: original verdicts are not overwritten
# ---------------------------------------------------------------------------

def test_reverification_does_not_mutate_original_verdicts():
    rv = ReVerifier()
    payload = _java_payload()
    original_claims_snapshot = [(c.id, c.status, c.text) for c in payload.claims]
    judge_result = JudgeVerificationResult(
        isApproved=True,
        trustScore=0.98,
        verifiedClaimsCount=1,
        remainingHallucinationsCount=0,
        feedback="approved",
    )
    rv.reverify(payload, _plan(payload), "Java was created by James Gosling in 1991.", judge_result)

    # payload.claims (the pre-correction verifier verdicts) survive untouched.
    assert [(c.id, c.status, c.text) for c in payload.claims] == original_claims_snapshot
    # The judge result object is likewise untouched.
    assert judge_result.isApproved is True


def test_injected_extractor_is_used():
    """A custom claim extractor can be injected (e.g. an LLM splitter) without
    requiring network — proves the seam works."""
    calls = {}

    def fake_extractor(text: str):
        calls["text"] = text
        return ["Java was created by James Gosling in 1991."]

    rv = ReVerifier(claim_extractor=fake_extractor)
    payload = _java_payload()
    result = rv.reverify(payload, _plan(payload), "irrelevant multi. sentence. text.")
    assert calls["text"] == "irrelevant multi. sentence. text."
    assert result.status == "FAILED"


# ---------------------------------------------------------------------------
# Orchestrator integration — FAILED reverification routes back to correction
# ---------------------------------------------------------------------------

class _StubModelClient:
    """Returns a scripted corrected text per attempt; no model/network."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    def generate_correction(self, prompt):
        idx = min(self.calls, len(self._outputs) - 1)
        self.calls += 1
        return self._outputs[idx]


class _AlwaysApproveJudge:
    """Pre-correction Judge that always approves — isolates the ReVerifier as the
    sole gate, so a caught secondary hallucination is unambiguously its doing."""

    def verifyCorrectedResponse(self, payload, plan, candidateResponse, attemptNumber):
        return JudgeVerificationResult(
            isApproved=True,
            trustScore=0.98,
            verifiedClaimsCount=len(payload.claims),
            remainingHallucinationsCount=0,
            feedback="Approved by stub judge.",
        )


def _run_orchestrator(payload, model_outputs, max_retries=3):
    from app.orchestrator import CorrectorOrchestrator

    orch = CorrectorOrchestrator()
    orch.judge = _AlwaysApproveJudge()
    orch.model_client = _StubModelClient(model_outputs)
    return asyncio.run(orch.executeCorrectionPipeline(payload, maxRetries=max_retries))


def test_orchestrator_rejects_judge_approved_answer_with_secondary_hallucination():
    """Judge approves, but the ReVerifier catches the wrong year and blocks it;
    with no better attempt available the pipeline terminates unresolved rather
    than emitting the bad answer."""
    payload = _java_payload()
    result = _run_orchestrator(
        payload,
        model_outputs=["Java was created by James Gosling in 1991."],
        max_retries=2,
    )
    # Bad answer must NOT be emitted verbatim as an approved result.
    assert result.isFullyApproved is False
    assert result.isTerminatedUnresolved is True
    # The reverification event is recorded in the trace, separate from the judge.
    reverify_logs = [t for t in result.traceLogs if t.stage == "REVERIFY"]
    assert reverify_logs
    assert any("FAILED" in t.title for t in reverify_logs)


def test_orchestrator_accepts_when_retry_fixes_the_secondary_hallucination():
    """Attempt 1 introduces the wrong year (ReVerifier FAILS -> retry); attempt 2
    is clean (ReVerifier VERIFIES -> accepted)."""
    payload = _java_payload()
    result = _run_orchestrator(
        payload,
        model_outputs=[
            "Java was created by James Gosling in 1991.",   # bad -> reverify fails
            "Java was created by James Gosling in 1995.",   # good -> reverify passes
        ],
        max_retries=3,
    )
    assert result.isFullyApproved is True
    assert result.isTerminatedUnresolved is False
    assert "1995" in result.finalResponse
    assert "1991" not in result.finalResponse
    reverify_logs = [t for t in result.traceLogs if t.stage == "REVERIFY"]
    assert any("FAILED" in t.title for t in reverify_logs)
    assert any("VERIFIED" in t.title for t in reverify_logs)


def test_orchestrator_accepts_clean_correction_first_try():
    payload = _java_payload()
    result = _run_orchestrator(
        payload,
        model_outputs=["Java was created by James Gosling in 1995."],
        max_retries=2,
    )
    assert result.isFullyApproved is True
    assert result.isTerminatedUnresolved is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
