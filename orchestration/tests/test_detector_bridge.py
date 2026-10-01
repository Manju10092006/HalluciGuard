"""Tests for the single HalluciGuard detector integration seam."""
from __future__ import annotations

import pytest

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


def test_bridge_forwards_refutation_signal_separately(monkeypatch):
    """contradiction_mass must survive the seam so Verifier/Judge can see it."""
    payload = _grounded(
        hallucination_probability=0.95,
        verification_risk=0.95,
        contradiction_mass=0.90,
        unknown_count=1,
    )
    monkeypatch.setattr(db, "_get_agent", lambda: _StubAgent(payload))
    out = db.run_detection("q", "r")
    # The triage score and the refutation signal are both present and distinct
    # in meaning: a high verification_risk with a low contradiction_mass means
    # "unverified", NOT "false".
    assert out["verification_risk"] == 0.95
    assert out["contradiction_mass"] == 0.90
    assert out["hallucination_probability"] == 0.95  # legacy alias preserved


def test_bridge_derives_contradiction_mass_from_claims_when_absent(monkeypatch):
    """An older payload without the answer-level field is still refutation-honest."""
    payload = _grounded()
    payload["claims"] = [
        {
            "claim_id": 1, "text": "Claim one.", "span": [0, 10], "claim_risk": 0.72,
            "label": "CONTRADICTED", "risk_level": "HIGH", "requires_verification": True,
            "contradicted_probability": 0.61,
        },
        {
            "claim_id": 2, "text": "Claim two.", "span": [11, 21], "claim_risk": 0.10,
            "label": "SUPPORTED", "risk_level": "LOW", "requires_verification": False,
            "contradicted_probability": 0.02,
        },
    ]
    monkeypatch.setattr(db, "_get_agent", lambda: _StubAgent(payload))
    out = db.run_detection("q", "r")
    # Max P(CONTRADICTED) across assessed claims, exactly as the agent computes.
    assert out["contradiction_mass"] == pytest.approx(0.61)


def test_bridge_excludes_non_factual_from_derived_contradiction_mass(monkeypatch):
    """An opinion claim must not contribute to the refutation signal."""
    payload = _grounded()
    payload["claims"] = [
        {
            "claim_id": 1, "text": "Best language ever.", "claim_risk": 0.5,
            "label": "NOT_ENOUGH_INFO", "contradicted_probability": 0.9,
            "non_factual": True, "requires_verification": False,
        },
    ]
    monkeypatch.setattr(db, "_get_agent", lambda: _StubAgent(payload))
    out = db.run_detection("q", "r")
    assert out["contradiction_mass"] == pytest.approx(0.0)


def test_bridge_defaults_contradiction_mass_when_absent(monkeypatch):
    """A grounded run that reports no refutation means exactly zero mass."""
    monkeypatch.setattr(db, "_get_agent", lambda: _StubAgent(_grounded()))
    out = db.run_detection("q", "r")
    assert out["contradiction_mass"] == 0.0


def test_bridge_never_fabricates_refutation_evidence_on_pre_verification(monkeypatch):
    """No detector probability => no refutation claim (None, not a fake 0.0)."""
    payload = _grounded()
    payload["hallucination_probability"] = None
    payload["verification_risk"] = None
    payload["contradiction_mass"] = None
    monkeypatch.setattr(db, "_get_agent", lambda: _StubAgent(payload))
    out = db.run_detection("q", "r")
    assert out["probability_available"] is False
    assert out["contradiction_mass"] is None
    # Still fails closed to verification.
    assert out["next_action"] == "Verify"


def test_bridge_keeps_unknown_out_of_contradiction(monkeypatch):
    """An unverified-only answer must not present a refutation signal."""
    payload = _grounded(
        hallucination_probability=0.90,
        verification_risk=0.90,
        contradiction_mass=0.0,
        contradicted_count=0,
        unknown_count=2,
    )
    monkeypatch.setattr(db, "_get_agent", lambda: _StubAgent(payload))
    out = db.run_detection("q", "r")
    assert out["contradiction_mass"] == 0.0
    assert out["unknown_count"] == 2
    # Still routed to verification, because unverified is not accepted.
    assert out["next_action"] == "Verify"


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
    assert out["hallucination_probability"] is None
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


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, "0.02"])
def test_invalid_probability_cannot_authorize_fast_path(monkeypatch, bad):
    monkeypatch.setattr(db, "_get_agent", lambda: _StubAgent(_grounded(
        hallucination_probability=bad, risk_level="LOW", next_action="Accept")))
    out = db.run_detection("q", "r")
    assert out["next_action"] == "Verify"
    assert out["detector_degraded"] is True
    assert out["probability_available"] is False
