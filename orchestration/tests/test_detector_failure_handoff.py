"""Offline fault injection proves real graph handoff, not only a route label."""
import json

import pytest

from orchestration import graph, detector_bridge


def _state():
    return {"execution_id": "failure-test", "request_id": "failure-test",
            "user_query": "Who founded Microsoft?", "llm_response": "Microsoft was founded by Bill Gates and Paul Allen.",
            "domain": "general", "trace": [], "errors": []}


@pytest.mark.asyncio
async def test_raised_detector_with_a_valid_answer_reaches_verifier(monkeypatch):
    secret = "dummy-secret-must-not-be-exposed"
    calls = []

    def broken(*args, **kwargs):
        raise RuntimeError(secret)

    async def analyzer(state):
        return {"route": "verify"}

    async def verifier(state):
        calls.append(state["llm_response"])
        return {"route": "human_escalation", "verification_status": "agent_failed"}

    monkeypatch.setattr(detector_bridge, "run_detection", broken)
    compiled = graph.build_verification_graph(node_overrides={
        "claim_analyzer": analyzer, "verifier": verifier})
    result = await compiled.ainvoke(_state())
    assert calls == [_state()["llm_response"]]
    assert result["detector_result"]["status"] == "degraded"
    assert result["detector_result"]["probability_available"] is False
    assert secret not in json.dumps(result, default=str)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [None, {}, {"hallucination_probability": None},
    {"hallucination_probability": float("nan")}, {"confidence_score": float("inf")},
    {"hallucination_probability": "invalid"}, {"per_claim_results": [None]}])
async def test_malformed_or_incomplete_detector_output_never_accepts(monkeypatch, payload):
    monkeypatch.setattr(detector_bridge, "run_detection", lambda *args, **kwargs: payload)
    monkeypatch.setenv("ALWAYS_VERIFY", "false")
    monkeypatch.setenv("ALLOW_DETECTOR_FAST_PATH", "true")
    out = await graph._detector_node(_state())
    assert graph._detector_route(out) == "verifier"
    assert out["route"] == "verify"


@pytest.mark.parametrize("changes", [{"status": "unavailable"}, {"mode": "shadow"},
    {"calibration_id": None}, {"head_id": ""}, {"calibrated_risks": []},
    {"response_risk": None}, {"response_risk": float("nan")},
    {"calibrated_risks": [float("inf")]}, {"response_risk": 0.2},
    {"bypass_eligible": "true"}])
def test_invalid_phase1_records_are_not_fast_path_authority(changes):
    record = {"status": "scored", "mode": "fast_path", "bypass_eligible": True,
              "head_id": "fixture", "calibration_id": "fixture",
              "calibrated_risks": [0.1], "response_risk": 0.1}
    record.update(changes)
    assert graph._valid_phase1_bypass(record, 1) is False


def test_missing_route_is_not_implicit_acceptance():
    assert graph._detector_route(_state()) == "verifier"
    assert graph._detector_route({}) == "human_escalation"


@pytest.mark.parametrize("risk", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_bridge_rejects_invalid_risk_and_preserves_unavailability(risk, monkeypatch):
    class Agent:
        def detect(self, *args, **kwargs):
            return {"hallucination_probability": risk, "status": "completed"}
    monkeypatch.setattr(detector_bridge, "_get_agent", Agent)
    out = detector_bridge.run_detection("Query", "Generated response")
    assert out["next_action"] == "Verify"
    assert out["probability_available"] is False
    assert out["detector_degraded"] is True


def test_bridge_public_failure_reason_is_sanitized(monkeypatch):
    def broken():
        raise ValueError("dummy-key-123456 and private evidence")
    monkeypatch.setattr(detector_bridge, "_get_agent", broken)
    for out in (detector_bridge.run_detection("Q", "A"),
                detector_bridge.run_grounded_detection("Q", "A", ["Evidence"])):
        assert "ValueError" in out["degraded_reason"]
        assert "dummy-key" not in json.dumps(out)
        assert "private evidence" not in json.dumps(out)
