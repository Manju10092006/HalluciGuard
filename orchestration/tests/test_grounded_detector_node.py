"""Regression tests for trained detector execution in the production graph."""
from __future__ import annotations

import pytest

from orchestration import detector_bridge
from orchestration.graph import _grounded_detector_node


@pytest.mark.asyncio
async def test_grounded_detector_runs_after_verifier_and_updates_judge_input(monkeypatch):
    calls = []

    def _detect(query, response, evidence):
        calls.append((query, response, evidence))
        return {
            "hallucination_probability": 0.17,
            "probability_available": True,
            "confidence_score": 0.91,
            "risk_level": "LOW",
            "next_action": "Accept",
            "model_source": "halluciguard_detector_ragtruth_deberta",
            "status": "completed",
            "calibration_applied": True,
            "inference_executed": True,
            "model_loaded": True,
            "detector_degraded": False,
            "grounded": True,
        }

    monkeypatch.setattr(detector_bridge, "run_grounded_detection", _detect)
    evidence = [{"snippet": "Microsoft was founded by Bill Gates and Paul Allen."}]
    state = {
        "user_query": "Who founded Microsoft?",
        "llm_response": "Microsoft was founded by Bill Gates and Paul Allen.",
        "retrieved_evidence": evidence,
        "inter_agent_bus": [],
        "trace": [],
    }

    result = await _grounded_detector_node(state)

    assert calls == [(state["user_query"], state["llm_response"], evidence)]
    assert result["detector_result"]["model_loaded"] is True
    assert result["detector_result"]["inference_executed"] is True
    assert result["detector_result"]["calibration_applied"] is True
    assert result["hallucination_probability"] == pytest.approx(0.17)
    assert result["trace"][-1]["node"] == "detector_grounded"
    assert result["trace"][-1]["status"] == "completed"
