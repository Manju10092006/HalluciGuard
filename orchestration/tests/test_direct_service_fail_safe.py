"""Offline direct-service regressions assert verification and grounded proof calls."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import pytest

from services import llm_detector_verifier_service as direct_service
from services.base_llm_service import GenerationResult


QUERY = "Who founded Microsoft?"
DRAFT = "Microsoft was founded by Bill Gates and Paul Allen."
EVIDENCE = "Microsoft was founded in 1975 by Bill Gates and Paul Allen."
SECRET = "dummy-private-detector-key-do-not-expose"


def _triage(**changes):
    result = {
        "status": "completed",
        "risk_level": "HIGH",
        "next_action": "Verify",
        "hallucination_probability": None,
        "confidence_score": None,
        "model_source": "halluciguard_detector_pre_verification_triage",
        "model_loaded": False,
        "inference_executed": False,
        "detector_degraded": False,
        "grounded": False,
        "calibrated": False,
        "claims": [{"claim_id": 1, "text": DRAFT}],
    }
    result.update(changes)
    return result


def _low_result(**changes):
    return _triage(risk_level="LOW", next_action="Accept", **changes)


def _phase1_result():
    return _low_result(
        phase1={
            "status": "scored",
            "mode": "fast_path",
            "bypass_eligible": True,
            "head_id": "offline-calibrated-head",
            "calibration_id": "offline-calibration",
            "calibrated_risks": [0.01],
            "response_risk": 0.01,
        }
    )


def _grounded_result(**changes):
    result = _triage(
        risk_level="LOW",
        next_action="Accept",
        hallucination_probability=0.01,
        confidence_score=0.99,
        model_source="halluciguard_detector_ragtruth_deberta",
        model_loaded=True,
        inference_executed=True,
        grounded=True,
        calibrated=True,
    )
    result.update(changes)
    return result


def _verifier_output(evidence_present=True):
    evidence = [
        SimpleNamespace(
            title="Microsoft company history",
            source="microsoft",
            url="https://www.microsoft.com/en-us/about",
            snippet=EVIDENCE,
            entailment_label="entailment",
            entailment_score=0.99,
            credibility_score=0.95,
        )
    ] if evidence_present else []
    return SimpleNamespace(
        domain="general",
        domain_validated=True,
        retrieved_sources=len(evidence),
        verified_sources=len(evidence),
        claim_evidence=[
            SimpleNamespace(
                claim_id="c1",
                claim_text=DRAFT,
                verdict="verified" if evidence else "not_enough_info",
                evidence=evidence,
            )
        ],
    )


class StubLLM:
    async def generate(self, user_query, **kwargs):
        return GenerationResult(
            user_query=user_query,
            draft_response=DRAFT,
            model="offline-generator",
            provider="offline",
            generation_mode="normal",
            mode="normal",
            temperature=0.0,
            latency_ms=0,
            finish_reason="stop",
            request_id="direct-service-offline",
            status="success",
        )


class StubDetector:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def detect(self, user_query, llm_response, evidence=None, **kwargs):
        self.calls.append(
            {"user_query": user_query, "llm_response": llm_response, "evidence": evidence}
        )
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return deepcopy(self.outcome)


class StubVerifier:
    def __init__(self, evidence_present=True, failure=None):
        self.output = _verifier_output(evidence_present)
        self.failure = failure
        self.calls = []

    async def verify(self, payload):
        self.calls.append(payload)
        if self.failure is not None:
            raise self.failure
        return self.output


@pytest.fixture(autouse=True)
def offline_service_dependencies(monkeypatch):
    """Scope import stubs to each test; never replace modules at collection time."""
    for name in ("ALWAYS_VERIFY", "ALLOW_DETECTOR_FAST_PATH", "CERTIFICATION_MODE"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(
        direct_service,
        "_load_verifier_imports",
        lambda: (StubVerifier, SimpleNamespace, SimpleNamespace),
    )

    claims_package = ModuleType("claims")
    claims_package.__path__ = []
    decomposer_module = ModuleType("claims.claim_decomposer")

    class StubClaimDecomposer:
        def decompose(self, answer):
            return [answer]

    decomposer_module.ClaimDecomposer = StubClaimDecomposer
    monkeypatch.setitem(sys.modules, "claims", claims_package)
    monkeypatch.setitem(sys.modules, "claims.claim_decomposer", decomposer_module)

    # pytest.ini already supplies the verifier package path; api.__init__ is light.
    from api.certification import (
        CertificationError,
        certification_enabled_from_env,
        enforce_detector,
    )

    monkeypatch.setattr(
        direct_service,
        "_load_certification",
        lambda: (enforce_detector, certification_enabled_from_env, CertificationError),
    )


def _service(triage, *, evidence_present=True, verifier_failure=None):
    detector = StubDetector(triage)
    verifier = StubVerifier(evidence_present, verifier_failure)
    service = direct_service.BaseLLMDetectorVerifierService(
        llm_service=StubLLM(), detector_agent=detector, verifier_pipeline=verifier
    )
    return service, detector, verifier


def _assert_verified(result, verifier):
    assert len(verifier.calls) == 1
    assert verifier.calls[0].suspicious_claims[0].text == DRAFT
    assert result.verifier["executed"] is True
    assert result.draft_response == DRAFT


def _assert_not_certified(result):
    certification = result.detector["certification"]
    assert certification["status"] == "not_certified"
    assert certification["stage"] == "detector"
    assert certification["reason"]


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["defaults", "always", "disabled-fast-path"])
async def test_default_and_safety_policies_verify_even_valid_phase1(monkeypatch, policy):
    if policy == "always":
        monkeypatch.setenv("ALWAYS_VERIFY", "true")
        monkeypatch.setenv("ALLOW_DETECTOR_FAST_PATH", "true")
    elif policy == "disabled-fast-path":
        monkeypatch.setenv("ALWAYS_VERIFY", "false")
        monkeypatch.setenv("ALLOW_DETECTOR_FAST_PATH", "false")
    service, detector, verifier = _service(_phase1_result())

    result = await service.execute_slice(QUERY)

    _assert_verified(result, verifier)
    assert len(detector.calls) == 1


@pytest.mark.asyncio
async def test_detector_exception_reaches_verifier_and_public_error_is_type_only():
    service, detector, verifier = _service(RuntimeError(SECRET))

    result = await service.execute_slice(QUERY)

    _assert_verified(result, verifier)
    assert len(detector.calls) == 1
    public_output = json.dumps(result.model_dump(), default=str)
    assert "RuntimeError" in public_output
    assert SECRET not in public_output
    assert result.detector["decision"] == "VERIFY"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [None, [], "LOW", {}, {"risk_level": "LOW", "next_action": "Accept"}],
)
async def test_malformed_detector_output_cannot_skip_verifier(monkeypatch, payload):
    monkeypatch.setenv("ALWAYS_VERIFY", "false")
    monkeypatch.setenv("ALLOW_DETECTOR_FAST_PATH", "true")
    service, _, verifier = _service(payload)

    result = await service.execute_slice(QUERY)

    _assert_verified(result, verifier)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["status", "risk_level", "next_action", "claims"])
async def test_incomplete_detector_record_cannot_authorize_phase1_skip(monkeypatch, field):
    monkeypatch.setenv("ALWAYS_VERIFY", "false")
    monkeypatch.setenv("ALLOW_DETECTOR_FAST_PATH", "true")
    payload = _phase1_result()
    del payload[field]
    service, _, verifier = _service(payload)

    result = await service.execute_slice(QUERY)

    _assert_verified(result, verifier)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"status": "failed"},
        {"status": "degraded"},
        {"status": "unavailable"},
        {"status": "failed", "detector_degraded": False},
        {"status": "degraded", "detector_degraded": False},
        {"detector_degraded": True},
    ],
)
async def test_failed_or_degraded_low_accept_record_still_verifies(monkeypatch, changes):
    monkeypatch.setenv("ALWAYS_VERIFY", "false")
    monkeypatch.setenv("ALLOW_DETECTOR_FAST_PATH", "true")
    payload = _phase1_result()
    payload.update(changes)
    service, _, verifier = _service(payload)

    result = await service.execute_slice(QUERY)

    _assert_verified(result, verifier)
    assert result.detector["detector_degraded"] is True


@pytest.mark.asyncio
async def test_uncalibrated_low_accept_is_not_fast_path_authority(monkeypatch):
    monkeypatch.setenv("ALWAYS_VERIFY", "false")
    monkeypatch.setenv("ALLOW_DETECTOR_FAST_PATH", "true")
    service, _, verifier = _service(_low_result())

    result = await service.execute_slice(QUERY)

    _assert_verified(result, verifier)


@pytest.mark.asyncio
async def test_valid_calibrated_phase1_can_skip_only_when_opted_in(monkeypatch):
    monkeypatch.setenv("ALWAYS_VERIFY", "false")
    monkeypatch.setenv("ALLOW_DETECTOR_FAST_PATH", "true")
    service, detector, verifier = _service(_phase1_result())

    result = await service.execute_slice(QUERY)

    assert verifier.calls == []
    assert len(detector.calls) == 1
    assert result.verifier["executed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"status": "unavailable"},
        {"mode": "shadow"},
        {"bypass_eligible": "true"},
        {"head_id": ""},
        {"calibration_id": None},
        {"calibrated_risks": []},
        {"calibrated_risks": [0.01, 0.01]},
        {"calibrated_risks": [float("inf")]},
        {"calibrated_risks": [True]},
        {"response_risk": float("nan")},
        {"response_risk": 0.2},
    ],
)
async def test_invalid_phase1_record_still_reaches_verifier(monkeypatch, changes):
    monkeypatch.setenv("ALWAYS_VERIFY", "false")
    monkeypatch.setenv("ALLOW_DETECTOR_FAST_PATH", "true")
    payload = _phase1_result()
    payload["phase1"].update(changes)
    service, _, verifier = _service(payload)

    result = await service.execute_slice(QUERY)

    _assert_verified(result, verifier)


@pytest.mark.asyncio
async def test_force_verifier_overrides_calibrated_phase1_fast_path(monkeypatch):
    monkeypatch.setenv("ALWAYS_VERIFY", "false")
    monkeypatch.setenv("ALLOW_DETECTOR_FAST_PATH", "true")
    service, _, verifier = _service(_phase1_result())

    result = await service.execute_slice(QUERY, force_verifier=True)

    _assert_verified(result, verifier)


@pytest.mark.asyncio
async def test_certification_cannot_reject_evidencefree_triage_before_verifier(monkeypatch):
    monkeypatch.setenv("CERTIFICATION_MODE", "true")
    service, detector, verifier = _service(_triage())

    result = await service.execute_slice(QUERY)

    _assert_verified(result, verifier)
    assert len(detector.calls) == 1
    assert detector.calls[0]["evidence"] is None
    _assert_not_certified(result)
    assert result.detector["detector_inference_executed"] is False


@pytest.mark.asyncio
async def test_certification_checks_original_grounded_proof_after_verifier(monkeypatch):
    monkeypatch.setenv("CERTIFICATION_MODE", "true")
    service, detector, verifier = _service(_grounded_result())
    enforce, enabled, error_type = direct_service._load_certification()
    observed_verifier_calls = []

    def enforce_after_verification(diagnostics, is_enabled):
        observed_verifier_calls.append(len(verifier.calls))
        enforce(diagnostics, is_enabled)

    monkeypatch.setattr(
        direct_service,
        "_load_certification",
        lambda: (enforce_after_verification, enabled, error_type),
    )

    result = await service.execute_slice(QUERY)

    _assert_verified(result, verifier)
    assert len(detector.calls) == 1
    assert observed_verifier_calls == [1]
    certification = result.detector["certification"]
    assert certification["status"] == "certified"
    assert certification["stage"] == "detector"


@pytest.mark.asyncio
async def test_certification_cannot_abort_force_verifier_on_failed_initial_detector(monkeypatch):
    monkeypatch.setenv("CERTIFICATION_MODE", "true")
    service, detector, verifier = _service(RuntimeError(SECRET))

    result = await service.execute_slice(QUERY, force_verifier=True)

    _assert_verified(result, verifier)
    assert len(detector.calls) == 1
    _assert_not_certified(result)
    assert SECRET not in json.dumps(result.model_dump(), default=str)


@pytest.mark.asyncio
@pytest.mark.parametrize("verifier_failure", [None, RuntimeError("offline verifier failed")])
async def test_no_returned_evidence_never_certifies_detector(monkeypatch, verifier_failure):
    monkeypatch.setenv("CERTIFICATION_MODE", "true")
    service, detector, verifier = _service(
        _triage(), evidence_present=False, verifier_failure=verifier_failure
    )

    result = await service.execute_slice(QUERY)

    _assert_verified(result, verifier)
    assert len(detector.calls) == 1
    _assert_not_certified(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial_detector",
    [
        RuntimeError(SECRET),
        {},
        _grounded_result(inference_executed=False),
        _grounded_result(grounded=False),
        _grounded_result(model_loaded=False),
        _grounded_result(detector_degraded=True),
        _grounded_result(status="failed", detector_degraded=False),
    ],
)
async def test_failed_or_non_grounded_inference_is_explicitly_not_certified(monkeypatch, initial_detector):
    monkeypatch.setenv("CERTIFICATION_MODE", "true")
    service, detector, verifier = _service(initial_detector)

    result = await service.execute_slice(QUERY)

    _assert_verified(result, verifier)
    assert len(detector.calls) == 1
    assert detector.calls[0]["evidence"] is None
    _assert_not_certified(result)
    assert SECRET not in json.dumps(result.model_dump(), default=str)
