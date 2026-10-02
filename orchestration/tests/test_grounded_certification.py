"""Controlled contract tests; no external model or retrieval execution."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents.verifier_agent.api.certification import enforce_detector, CertificationError
from services.llm_detector_verifier_service import BaseLLMDetectorVerifierService
from services.llm_detector_verifier_service import _source_identifier
from orchestration.tests.test_final_integration_safety import generation


def grounded():
    return {"status": "completed", "detector_degraded": False, "grounded": True,
            "model_loaded": True, "inference_executed": True,
            "calibration_applied": True, "hallucination_probability": .2,
            "confidence_score": .8, "claims": [{"claim_risk": .2,
                "model_input_evidence": "Paris is the capital of France.",
                "probabilities": {"SUPPORTED": .8, "CONTRADICTED": .1,
                                  "NOT_ENOUGH_INFO": .1}}]}


@pytest.mark.parametrize("value,expected", [(None, None), (object(), None),
    (True, None), ("official-france", "official-france"), (0, "0")])
def test_source_ids_are_serializable_without_fabricating_provenance(value, expected):
    assert _source_identifier(value) == expected


@pytest.mark.parametrize("key,value", [
    ("status", "triage"), ("grounded", False), ("model_loaded", False),
    ("inference_executed", False), ("calibration_applied", False),
    ("detector_degraded", True), ("hallucination_probability", None),
    ("hallucination_probability", float("nan")), ("confidence_score", True),
    ("claims", []), ("claims", [None]),
])
def test_incomplete_record_cannot_be_certified(key, value):
    payload = grounded()
    payload[key] = value
    with pytest.raises(CertificationError):
        enforce_detector(payload, True)


@pytest.mark.parametrize("payload", [None, {}, {"detector_degraded": False,
                                             "detector_inference_executed": True}])
def test_missing_result_is_not_a_certificate(payload):
    with pytest.raises(CertificationError):
        enforce_detector(payload, True)
    enforce_detector(payload, False)


@pytest.mark.parametrize("change", [
    {"probabilities": {"SUPPORTED": 1.}}, {"claim_risk": None},
    {"model_input_evidence": ""}, {"non_factual": True},
    {"probabilities": {"SUPPORTED": .8, "CONTRADICTED": .8, "NOT_ENOUGH_INFO": .1}},
])
def test_claim_requires_real_input_and_valid_probabilities(change):
    payload = grounded()
    payload["claims"][0].update(change)
    with pytest.raises(CertificationError):
        enforce_detector(payload, True)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("case", ["valid", "detector_error", "invalid_grounded", "no_evidence", "verifier_error"])
async def test_verifier_precedes_grounded_certification(monkeypatch, mode, case):
    monkeypatch.setenv("CERTIFICATION_MODE", str(mode).lower())
    calls = []
    def detect(**kwargs):
        if "evidence" not in kwargs:
            calls.append("triage")
            if case == "detector_error":
                raise RuntimeError("dummy-private-key")
            return {"status": "triage", "risk_level": "HIGH", "next_action": "Verify"}
        calls.append("grounded")
        assert calls[:2] == ["triage", "verifier"]
        assert kwargs["evidence"][0]["source_id"] == "official-france"
        result = grounded()
        if case == "invalid_grounded":
            result["hallucination_probability"] = None
        return result
    async def verify(payload):
        calls.append("verifier")
        if case == "verifier_error":
            raise RuntimeError("dummy-private-key")
        evidence = [] if case == "no_evidence" else [SimpleNamespace(
            snippet="Paris is the capital of France.", source_id="official-france",
            source="government", url="https://www.france.fr")]
        return SimpleNamespace(claim_evidence=[SimpleNamespace(evidence=evidence)], domain="general")
    verifier = SimpleNamespace(verify=AsyncMock(side_effect=verify))
    service = BaseLLMDetectorVerifierService(
        llm_service=SimpleNamespace(generate=AsyncMock(return_value=generation())),
        detector_agent=SimpleNamespace(detect=detect), verifier_pipeline=verifier)
    result = await service.execute_slice("What is the capital of France?")
    verifier.verify.assert_awaited_once()
    assert "dummy-private-key" not in str(result.model_dump())
    if not mode:
        assert calls == ["triage", "verifier"]
        assert result.certification is None
    elif case in {"valid", "detector_error"}:
        assert calls == ["triage", "verifier", "grounded"]
        assert result.certification["status"] == "passed"
        assert result.certification["completed"] is True
        assert result.detector["certification_completed"] is False
    else:
        assert result.certification["completed"] is False
        assert result.certification["status"] == "failed"
        assert result.verifier["status"] == ("failed" if case == "verifier_error" else "completed")
