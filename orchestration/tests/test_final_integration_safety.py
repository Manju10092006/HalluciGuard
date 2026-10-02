"""Release routing checks with controlled services, not accuracy experiments."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from orchestration import detector_bridge
from orchestration.graph import _detector_node
from services import llm_detector_verifier_service as service_module
from services.base_llm_service import GenerationResult
from services.llm_detector_verifier_service import BaseLLMDetectorVerifierService


def grounded_low():
    return dict(status="completed", detector_degraded=False, grounded=True,
                calibration_applied=True, calibrated=True, model_loaded=True,
                inference_executed=True, hallucination_probability=.02,
                probability_available=True, confidence_score=.98,
                risk_level="LOW", next_action="Accept", claims=[])


@pytest.mark.parametrize("field,value", [
    ("status", None), ("grounded", "false"), ("calibration_applied", "true"),
    ("inference_executed", "true"), ("model_loaded", False),
    ("detector_degraded", "false"),
])
def test_bridge_incomplete_execution_flags_cannot_accept(field, value):
    payload = grounded_low()
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value
    output = detector_bridge._map_result(payload)
    assert output["next_action"] == "Verify"
    assert output["risk_level"] == "HIGH"
    if field == "status":
        assert output["status"] == "unknown"
        assert output["detector_degraded"]


@pytest.mark.asyncio
async def test_graph_cannot_enable_unreleased_phase1_with_opt_in_flags(monkeypatch):
    monkeypatch.setenv("ALWAYS_VERIFY", "false")
    monkeypatch.setenv("ALLOW_DETECTOR_FAST_PATH", "true")
    monkeypatch.setattr(detector_bridge, "run_detection", lambda *a: grounded_low())
    result = await _detector_node(dict(user_query="What is the capital of France?",
                                     llm_response="Paris is the capital of France."))
    assert result["route"] == "verify"
    assert result["verification_status"] == "verification_required"


def generation():
    return GenerationResult(user_query="q", draft_response="Paris is the capital of France.",
                            provider="fixture", model="fixture", mode="normal",
                            generation_mode="normal", temperature=0, latency_ms=0,
                            finish_reason="stop", request_id="fixture", status="success")


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    grounded_low(), {}, None, RuntimeError("dummy-secret"),
    {**grounded_low(), "hallucination_probability": float("nan")},
    {**grounded_low(), "grounded": "false"},
])
@pytest.mark.parametrize("certification", [False, True])
async def test_direct_slice_always_reaches_verifier(monkeypatch, payload, certification):
    monkeypatch.setenv("CERTIFICATION_MODE", str(certification).lower())
    monkeypatch.setenv("ALWAYS_VERIFY", "false")
    monkeypatch.setenv("ALLOW_DETECTOR_FAST_PATH", "true")
    def detect(**kwargs):
        if isinstance(payload, Exception):
            raise payload
        return payload
    verifier = SimpleNamespace(verify=AsyncMock(return_value=SimpleNamespace(
        claim_evidence=[], retrieved_sources=0, verified_sources=0, domain="general")))
    service = BaseLLMDetectorVerifierService(
        llm_service=SimpleNamespace(generate=AsyncMock(return_value=generation())),
        detector_agent=SimpleNamespace(detect=detect), verifier_pipeline=verifier)
    result = await service.execute_slice("q")
    verifier.verify.assert_awaited_once()
    assert result.verifier["executed"]
    assert "dummy-secret" not in str(result.model_dump())
    if certification:
        assert result.detector["certification_completed"] is False
        assert result.detector["certification_status"] == "pending_grounded_detection"


@pytest.mark.asyncio
@pytest.mark.parametrize("verifier_fails", [False, True])
async def test_certification_helper_failure_remains_visible_and_verifies(monkeypatch, verifier_fails):
    monkeypatch.setenv("CERTIFICATION_MODE", "true")
    def fail_helpers():
        raise ImportError("dummy-secret")
    monkeypatch.setattr(service_module, "_load_certification", fail_helpers)
    verifier = SimpleNamespace(verify=AsyncMock(
        side_effect=RuntimeError("dummy-secret") if verifier_fails else None,
        return_value=SimpleNamespace(claim_evidence=[], domain="general")))
    service = BaseLLMDetectorVerifierService(
        llm_service=SimpleNamespace(generate=AsyncMock(return_value=generation())),
        detector_agent=SimpleNamespace(detect=lambda **kw: grounded_low()),
        verifier_pipeline=verifier)
    result = await service.execute_slice("q")
    verifier.verify.assert_awaited_once()
    assert result.detector["certification_completed"] is False
    assert result.detector["certification_reason"] == (
        "verifier_failed_before_grounded_detection" if verifier_fails
        else "certification_helpers_unavailable")
    assert result.verifier["status"] == ("failed" if verifier_fails else "completed")
    assert "dummy-secret" not in str(result.model_dump())
