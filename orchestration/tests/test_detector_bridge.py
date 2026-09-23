"""Unit tests for the detector integration seam (orchestration/detector_bridge.py).

These stub the DetV2 agent so no model loads — they validate the contract mapping
(DetV2 ``claims`` -> ``per_claim_results``) and the fail-closed behavior (any DetV2
error routes to Verify and never fabricates an Accept), all deterministically.
DetV2 is the sole detector; there is no V1 fallback or implementation selection.
"""
from __future__ import annotations

import pytest

from orchestration import detector_bridge as db


class _StubV2Agent:
    def __init__(self, payload):
        self._payload = payload

    def detect(self, user_query, llm_response):
        return self._payload


def _ext(**overrides):
    base = {
        "hallucination_probability": 0.80,
        "confidence_score": 0.64,
        "risk_level": "HIGH",
        "next_action": "Verify",
        "model_source": "detector_v2_stage7_calibrated",
        "status": "completed",
        "calibrated": True,
        "claims": [
            {"claim_id": 0, "text": "Claim one.", "span": [0, 10], "claim_risk": 0.72, "signals": []},
            {"claim_id": 1, "text": "Claim two.", "span": [11, 21], "claim_risk": 0.10, "signals": []},
        ],
        "diagnostics": {"degraded_reason": None},
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Isolate every test from any ambient legacy selection flags (now ignored).
    monkeypatch.delenv("HG_DETECTOR_IMPL", raising=False)
    monkeypatch.delenv("HG_DETECTOR_FALLBACK_V1", raising=False)


def test_tier_boundaries():
    assert db._tier(0.50) == "HIGH"
    assert db._tier(0.49) == "MEDIUM"
    assert db._tier(0.30) == "MEDIUM"
    assert db._tier(0.2999) == "LOW"


def test_v2_maps_claims_to_per_claim_results(monkeypatch):
    monkeypatch.setattr(db, "_get_v2_agent", lambda: _StubV2Agent(_ext()))

    out = db.run_detection("q", "r")

    # Canonical six preserved.
    for key in ("hallucination_probability", "confidence_score", "risk_level",
                "next_action", "model_source", "status"):
        assert key in out
    assert out["hallucination_probability"] == 0.80
    assert out["confidence_score"] == 0.64
    assert out["risk_level"] == "HIGH"
    assert out["next_action"] == "Verify"
    assert out["calibrated"] is True
    assert out["detector_degraded"] is False

    # claims -> per_claim_results shape.
    pcr = out["per_claim_results"]
    assert len(pcr) == 2
    assert pcr[0]["claim_id"] == "c0"
    assert pcr[0]["risk_level"] == "HIGH"        # 0.72 -> HIGH
    assert pcr[1]["claim_id"] == "c1"
    assert pcr[1]["risk_level"] == "LOW"         # 0.10 -> LOW
    # Overall routes to Verify -> every claim requires verification (fail toward it).
    assert all(c["requires_verification"] for c in pcr)


def test_v2_low_risk_accept_leaves_low_claims_unverified(monkeypatch):
    ext = _ext(
        hallucination_probability=0.05, risk_level="LOW", next_action="Accept",
        claims=[{"claim_id": 0, "text": "Safe.", "span": [0, 5], "claim_risk": 0.05, "signals": []}],
    )
    monkeypatch.setattr(db, "_get_v2_agent", lambda: _StubV2Agent(ext))

    out = db.run_detection("q", "r")
    assert out["next_action"] == "Accept"
    assert out["per_claim_results"][0]["requires_verification"] is False


def test_v2_degraded_fails_closed(monkeypatch):
    # Adversarial payload: a DEGRADED run trying to Accept with None probs.
    ext = _ext(status="degraded", next_action="Accept", risk_level="LOW",
               hallucination_probability=None, confidence_score=None)
    monkeypatch.setattr(db, "_get_v2_agent", lambda: _StubV2Agent(ext))

    out = db.run_detection("q", "r")
    assert out["next_action"] == "Verify"        # forced
    assert out["risk_level"] == "HIGH"           # forced
    assert out["status"] == "degraded"
    assert out["detector_degraded"] is True
    assert out["hallucination_probability"] == 0.0   # None coerced (graph does float())
    assert out["confidence_score"] == 0.0


def test_v2_runtime_error_fails_closed(monkeypatch):
    # DetV2 is the only detector: any runtime failure fails closed to Verify,
    # never fabricates an Accept, and never substitutes another model.
    def _boom():
        raise RuntimeError("encoder load failed")
    monkeypatch.setattr(db, "_get_v2_agent", _boom)

    out = db.run_detection("q", "r")
    assert out["next_action"] == "Verify"
    assert out["risk_level"] == "HIGH"
    assert out["detector_degraded"] is True
    assert out["model_source"] == "detector_v2_unavailable"
    assert out["per_claim_results"] == []
    assert "v2_failed" in out["degraded_reason"]


def test_v2_skips_empty_claim_text(monkeypatch):
    ext = _ext(claims=[
        {"claim_id": 0, "text": "  ", "span": [0, 2], "claim_risk": 0.9, "signals": []},
        {"claim_id": 1, "text": "Real claim.", "span": [3, 14], "claim_risk": 0.9, "signals": []},
    ])
    monkeypatch.setattr(db, "_get_v2_agent", lambda: _StubV2Agent(ext))

    pcr = db.run_detection("q", "r")["per_claim_results"]
    assert len(pcr) == 1
    assert pcr[0]["claim_id"] == "c1"
