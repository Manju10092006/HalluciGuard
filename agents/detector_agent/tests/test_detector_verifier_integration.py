"""Contract tests for the Detector → Verifier handoff.

These tests mock the network boundary so they prove routing and payload
construction without requiring a running Verifier service or GPU model.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

import agents.detector_agent.app as detector_app_module
from agents.detector_agent.models import DetectionResult, NextAction, RiskLevel
from agents.detector_agent.verifier_client import VerifierUnavailableError


class FakeDetector:
    def __init__(self, result: DetectionResult) -> None:
        self.result = result

    def detect(self, *, user_query: str, llm_response: str) -> DetectionResult:
        return self.result


def _result(action: NextAction, risk: RiskLevel) -> DetectionResult:
    return DetectionResult(
        confidence_score=0.9 if risk != RiskLevel.HIGH else 0.1,
        hallucination_probability=0.1 if risk != RiskLevel.HIGH else 0.9,
        risk_level=risk,
        next_action=action,
    )


def test_low_risk_still_invokes_verifier(monkeypatch):
    calls = []
    detector_app_module.agent = FakeDetector(_result(NextAction.ACCEPT, RiskLevel.LOW))

    async def fake_verify(**kwargs):
        calls.append(kwargs)
        return {"claim_evidence": [{"verdict": "insufficient_evidence", "evidence": []}]}

    monkeypatch.setattr(detector_app_module.verifier_client, "verify", fake_verify)

    with TestClient(detector_app_module.app) as client:
        response = client.post(
            "/analyze",
            json={
                "user_query": "What is the capital of France?",
                "llm_response": "The capital of France is Paris.",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["verifier_invoked"] is True
    assert body["final_status"] == "INSUFFICIENT_EVIDENCE"
    assert len(calls) == 1


def test_medium_risk_still_invokes_verifier(monkeypatch):
    calls = []
    detector_app_module.agent = FakeDetector(_result(NextAction.ACCEPT, RiskLevel.MEDIUM))

    async def fake_verify(**kwargs):
        calls.append(kwargs)
        return {"claim_evidence": []}

    monkeypatch.setattr(detector_app_module.verifier_client, "verify", fake_verify)

    with TestClient(detector_app_module.app) as client:
        response = client.post(
            "/analyze",
            json={
                "user_query": "Explain HTTP.",
                "llm_response": "HTTP stands for HyperText Transfer Programming.",
            },
        )

    assert response.status_code == 200
    assert response.json()["verifier_invoked"] is True
    assert len(calls) == 1


def test_high_risk_invokes_verifier_exactly_once(monkeypatch):
    detector_app_module.agent = FakeDetector(_result(NextAction.VERIFY, RiskLevel.HIGH))
    calls = []

    async def fake_verify(**kwargs):
        calls.append(kwargs)
        return {
            "query_id": kwargs["query_id"],
            "domain": kwargs["domain"],
            "claim_evidence": [
                {
                    "claim_id": "x",
                    "claim_text": kwargs["claim_text"],
                    "verdict": "likely_hallucinated",
                }
            ],
        }

    monkeypatch.setattr(detector_app_module.verifier_client, "verify", fake_verify)

    with TestClient(detector_app_module.app) as client:
        response = client.post(
            "/analyze",
            json={
                "query_id": "integration-1",
                "domain": "general",
                "user_query": "What is the capital of France?",
                "llm_response": "The capital of France is Tokyo.",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["verifier_invoked"] is True
    assert body["final_status"] == "LIKELY_HALLUCINATED"
    assert len(calls) == 1
    assert calls[0]["query_id"] == "integration-1"
    assert calls[0]["domain"] == "general"
    assert calls[0]["claim_text"] == "The capital of France is Tokyo."


def test_high_risk_fails_closed_when_verifier_unavailable(monkeypatch):
    detector_app_module.agent = FakeDetector(_result(NextAction.VERIFY, RiskLevel.HIGH))

    async def fake_verify(**kwargs):
        raise VerifierUnavailableError("connection refused")

    monkeypatch.setattr(detector_app_module.verifier_client, "verify", fake_verify)

    with TestClient(detector_app_module.app) as client:
        response = client.post(
            "/analyze",
            json={
                "user_query": "What is the capital of France?",
                "llm_response": "The capital of France is Tokyo.",
            },
        )

    assert response.status_code == 503
    assert "could not be verified" in response.json()["detail"]["message"]


def test_detector_exception_still_invokes_verifier_and_sanitizes_error(monkeypatch):
    class Broken:
        def detect(self, **kwargs):
            raise RuntimeError("dummy-secret")
    detector_app_module.agent = Broken()
    calls = []
    async def fake_verify(**kwargs):
        calls.append(kwargs)
        return {"claim_evidence": []}
    monkeypatch.setattr(detector_app_module.verifier_client, "verify", fake_verify)
    with TestClient(detector_app_module.app) as client:
        response = client.post("/analyze", json={"user_query": "Who?", "llm_response": "Draft."})
    assert response.status_code == 200
    assert len(calls) == 1
    assert response.json()["detection"]["status"] == "failed"
    assert "dummy-secret" not in response.text


def test_malformed_detector_output_still_invokes_verifier(monkeypatch):
    class Malformed:
        def detect(self, **kwargs):
            return {"risk_level": "LOW", "next_action": "Accept",
                    "hallucination_probability": float("nan")}
    detector_app_module.agent = Malformed()
    calls = []
    async def fake_verify(**kwargs):
        calls.append(kwargs)
        return {"claim_evidence": []}
    monkeypatch.setattr(detector_app_module.verifier_client, "verify", fake_verify)
    with TestClient(detector_app_module.app) as client:
        response = client.post("/analyze", json={"user_query": "Who?", "llm_response": "Draft."})
    assert response.status_code == 200
    assert len(calls) == 1
    assert response.json()["detection"]["probability_available"] is False


def test_verifier_failure_cannot_return_accepted_or_verified(monkeypatch):
    detector_app_module.agent = FakeDetector(_result(NextAction.ACCEPT, RiskLevel.LOW))
    async def fake_verify(**kwargs):
        raise VerifierUnavailableError("dummy-secret")
    monkeypatch.setattr(detector_app_module.verifier_client, "verify", fake_verify)
    with TestClient(detector_app_module.app) as client:
        response = client.post("/analyze", json={"user_query": "Who?", "llm_response": "Draft."})
    assert response.status_code == 503
    assert response.json()["detail"]["error_type"] == "VerifierUnavailableError"
    assert "dummy-secret" not in response.text


def test_original_detect_endpoint_remains_detector_only(monkeypatch):
    detector_app_module.agent = FakeDetector(_result(NextAction.VERIFY, RiskLevel.HIGH))

    async def fail_if_called(**kwargs):
        raise AssertionError("/detect must not invoke Verifier")

    monkeypatch.setattr(detector_app_module.verifier_client, "verify", fail_if_called)

    with TestClient(detector_app_module.app) as client:
        response = client.post(
            "/detect",
            json={
                "user_query": "What is the capital of France?",
                "llm_response": "The capital of France is Tokyo.",
            },
        )

    assert response.status_code == 200
    assert response.json()["next_action"] == "Verify"
