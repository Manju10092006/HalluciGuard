"""Tests for the single HalluciGuard detector integration seam."""
from __future__ import annotations

from orchestration import detector_bridge as db


class _StubAgent:
    def __init__(self, payload):
        self._payload = payload

    def detect(self, user_query, llm_response):
        return self._payload


class _GroundedStubAgent(_StubAgent):
    def __init__(self, payload):
        super().__init__(payload)
        self.evidence = None

    def detect(self, user_query, llm_response, evidence=None):
        self.evidence = evidence
        return self._payload


def _grounded(**overrides):
    base = {
        "hallucination_probability": 0.80,
        "confidence_score": 0.90,
        "risk_level": "HIGH",
        "next_action": "Verify",
        "model_source": "halluciguard_detector_ragtruth_deberta",
        "status": "completed",
        "calibration_applied": True,
        "inference_executed": True,
        "model_loaded": True,
        "grounded": True,
        "claims": [
            {"claim_id": 1, "text": "Claim one.", "span": [0, 10], "claim_risk": 0.72, "label": "CONTRADICTED", "risk_level": "HIGH", "requires_verification": True},
            {"claim_id": 2, "text": "Claim two.", "span": [11, 21], "claim_risk": 0.10, "label": "SUPPORTED", "risk_level": "LOW", "requires_verification": False},
        ],
        "diagnostics": {"degraded_reason": None},
    }
    base.update(overrides)
    return base


def test_grounded_result_maps_claims(monkeypatch):
    monkeypatch.setattr(db, "_get_agent", lambda: _StubAgent(_grounded()))
    out = db.run_detection("q", "r")
    assert out["hallucination_probability"] == 0.80
    assert out["probability_available"] is True
    assert out["grounded"] is True
    assert out["calibrated"] is True
    assert out["per_claim_results"][0]["claim_id"] == "c1"
    assert out["per_claim_results"][0]["label"] == "CONTRADICTED"
    assert out["per_claim_results"][1]["requires_verification"] is False


def test_grounded_production_entrypoint_passes_verifier_evidence(monkeypatch):
    agent = _GroundedStubAgent(_grounded())
    monkeypatch.setattr(db, "_get_agent", lambda: agent)
    evidence = [{"snippet": "Claim one is false."}]

    out = db.run_grounded_detection("q", "Claim one.", evidence)

    assert agent.evidence == evidence
    assert out["model_loaded"] is True
    assert out["inference_executed"] is True
    assert out["calibration_applied"] is True
    assert out["probability_available"] is True


def test_grounded_production_entrypoint_fails_closed(monkeypatch):
    class _BrokenAgent:
        def detect(self, user_query, llm_response, evidence=None):
            raise RuntimeError("checkpoint corrupt")

    monkeypatch.setattr(db, "_get_agent", lambda: _BrokenAgent())
    out = db.run_grounded_detection("q", "r", ["evidence"])

    assert out["risk_level"] == "HIGH"
    assert out["next_action"] == "Verify"
    assert out["detector_degraded"] is True
    assert "grounded_detector_failed" in out["degraded_reason"]


def test_pre_verification_triage_does_not_fabricate_probability(monkeypatch):
    payload = _grounded(
        hallucination_probability=None,
        confidence_score=None,
        model_source="halluciguard_detector_pre_verification_triage",
        calibration_applied=False,
        inference_executed=False,
        model_loaded=False,
        grounded=False,
        probability_semantics="not_computed_without_evidence",
        claims=[{"claim_id": 1, "text": "A factual claim.", "span": [0, 16], "claim_risk": None, "label": "UNVERIFIED", "risk_level": "HIGH", "requires_verification": True}],
    )
    monkeypatch.setattr(db, "_get_agent", lambda: _StubAgent(payload))
    out = db.run_detection("q", "A factual claim.")
    assert out["next_action"] == "Verify"
    assert out["hallucination_probability"] == 0.0
    assert out["probability_available"] is False
    assert out["per_claim_results"][0]["probability_available"] is False
    assert out["detector_degraded"] is False


def test_degraded_result_fails_closed(monkeypatch):
    payload = _grounded(status="degraded", next_action="Accept", risk_level="LOW")
    monkeypatch.setattr(db, "_get_agent", lambda: _StubAgent(payload))
    out = db.run_detection("q", "r")
    assert out["next_action"] == "Verify"
    assert out["risk_level"] == "HIGH"
    assert out["detector_degraded"] is True


def test_runtime_error_fails_closed(monkeypatch):
    def _boom():
        raise RuntimeError("model load failed")

    monkeypatch.setattr(db, "_get_agent", _boom)
    out = db.run_detection("q", "r")
    assert out["next_action"] == "Verify"
    assert out["risk_level"] == "HIGH"
    assert out["detector_degraded"] is True
    assert out["model_source"] == "halluciguard_detector_unavailable"
    assert "detector_failed" in out["degraded_reason"]


def test_empty_claim_text_is_skipped(monkeypatch):
    payload = _grounded(claims=[
        {"claim_id": 1, "text": "  ", "claim_risk": 0.9},
        {"claim_id": 2, "text": "Real claim.", "claim_risk": 0.9},
    ])
    monkeypatch.setattr(db, "_get_agent", lambda: _StubAgent(payload))
    out = db.run_detection("q", "r")
    assert [item["claim_id"] for item in out["per_claim_results"]] == ["c2"]
