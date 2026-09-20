"""Detector defect regressions — no model weights required.

Pins the fail-closed + honest-status fixes:
  1. _default_result (empty input / inference error) fails CLOSED: HIGH/VERIFY,
     status='degraded' — never silently ACCEPT an unverified answer.
  2. DetectionResult now carries a `status` field so orchestration's
     `detector.get("status")` degraded gate is no longer dead.
  3. A degraded detector run (model not loaded) emits an HONEST 0.5 posture that
     maps to HIGH/VERIFY with status='degraded', not a fabricated confident-low
     0.08/0.92, and marks every claim requires_verification=True.
"""
from __future__ import annotations

import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.detector_agent.detector import DetectorAgent
from agents.detector_agent.models import DetectionResult, NextAction, RiskLevel


class _FakeInference:
    """Stand-in HaluEval inference that reports NOT loaded (degraded path)."""

    _loaded = False
    model_path = "fake"

    def load(self):  # pragma: no cover - never called once _SHARED_MODEL_LOADED
        raise RuntimeError("weights absent")

    def is_loaded(self):
        return False


def _degraded_agent() -> DetectorAgent:
    agent = DetectorAgent()
    # Force the degraded (model-not-loaded) branch without touching disk/HF.
    agent._inference = _FakeInference()
    DetectorAgent._SHARED_MODEL_LOADED = True
    return agent


def test_default_result_fails_closed():
    agent = _degraded_agent()
    res = agent._default_result("Empty query")
    assert res.risk_level == RiskLevel.HIGH
    assert res.next_action == NextAction.VERIFY
    assert res.status == "degraded"
    assert res.detector_degraded is True


def test_empty_query_routes_to_verify():
    agent = _degraded_agent()
    res = agent.detect(user_query="", llm_response="The capital of France is Paris.")
    assert res.next_action == NextAction.VERIFY
    assert res.risk_level == RiskLevel.HIGH


def test_detection_result_has_status_field():
    res = DetectionResult(
        confidence_score=0.9,
        hallucination_probability=0.1,
        risk_level=RiskLevel.LOW,
        next_action=NextAction.ACCEPT,
    )
    # default present + serialized so orchestration can read detector["status"]
    assert res.status == "completed"
    assert res.model_dump()["status"] == "completed"


def test_degraded_run_is_honest_and_verifies():
    agent = _degraded_agent()
    res = agent.detect(
        user_query="Who wrote Hamlet?",
        llm_response="Hamlet was written by William Shakespeare.",
    )
    # Honest unknown posture -> HIGH/VERIFY, not fabricated confident-low.
    assert res.status == "degraded"
    assert res.detector_degraded is True
    assert res.detector_inference_executed is False
    assert res.next_action == NextAction.VERIFY
    assert res.hallucination_probability == 0.5
    assert all(c.requires_verification for c in res.per_claim_results)
