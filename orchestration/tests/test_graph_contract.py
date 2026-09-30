from __future__ import annotations

from orchestration.graph import (
    _detector_node,
    _grounded_detector_node,
    _verifier_node,
    _corrector_route,
    _detector_route,
    _generate_route,
    _reverifier_route,
    _verifier_route,
    build_verification_graph,
)
from orchestration.state import add_trace


def test_graph_compiles():
    assert build_verification_graph() is not None


def test_state_trace_has_observability_fields():
    trace = add_trace(
        {"execution_id": "ex-1", "retry_count": 1},
        "detector",
        "completed",
        latency_ms=7,
    )
    assert trace[0]["execution_id"] == "ex-1"
    assert trace[0]["node"] == "detector"
    assert trace[0]["latency_ms"] == 7
    assert trace[0]["retry_count"] == 1


def test_generate_route_success():
    assert _generate_route({"llm_response": "Paris is the capital."}) == "detector"


def test_generate_route_failure():
    assert _generate_route({"route": "error", "llm_response": ""}) == "human_escalation"


def test_detector_explicit_fast_path_route():
    assert _detector_route({"route": "accept"}) == "accept"


def test_detector_high_routes_to_verifier():
    assert _detector_route({"route": "verify"}) == "verifier"


def test_detector_failure_routes_to_human_escalation():
    assert _detector_route({"route": "error"}) == "human_escalation"


async def _broken_detector_node(monkeypatch):
    from orchestration import detector_bridge

    monkeypatch.setattr(detector_bridge, "run_detection", lambda *args: (_ for _ in ()).throw(RuntimeError("secret")))
    return await _detector_node({"user_query": "Question?", "llm_response": "A factual draft."})


def test_detector_node_error_with_draft_routes_to_verifier(monkeypatch):
    import asyncio

    update = asyncio.run(_broken_detector_node(monkeypatch))
    assert update["route"] == "verify"
    assert update["detector_result"]["detector_degraded"] is True
    assert update["detected_claims"][0]["text"] == "A factual draft."
    assert "secret" not in str(update)


def test_grounded_detector_trace_preserves_upstream_retrieval_degradation(monkeypatch):
    import asyncio
    from orchestration import detector_bridge

    monkeypatch.setattr(detector_bridge, "run_grounded_detection", lambda *args: {
        "inference_executed": False, "model_loaded": False,
        "hallucination_probability": 0.0, "confidence_score": 0.0,
        "risk_level": "HIGH", "detector_degraded": True,
    })
    update = asyncio.run(_grounded_detector_node({
        "user_query": "Question?", "llm_response": "A factual draft.",
        "retrieved_evidence": [{"snippet": "Evidence."}],
        "verifier": {"claim_evidence": [{"retrieval_trace": {
            "hybrid_execution": {"route": "bm25_only"}, "evidence_degraded": True,
        }}]},
    }))
    diag = update["detector_result"]["diagnostics"]
    assert diag["upstream_retrieval_degraded"] is True
    assert diag["upstream_retrieval_routes"] == ["bm25_only"]


def test_verifier_boundary_does_not_return_sensitive_exception_message(monkeypatch):
    import asyncio
    from orchestration import graph

    class Input:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Pipeline:
        async def verify(self, payload):
            raise RuntimeError("dummy-api-key-secret")

    monkeypatch.setattr(graph, "_get_verifier_imports", lambda: (Pipeline, Input, Input))
    update = asyncio.run(_verifier_node({
        "user_query": "Question?", "llm_response": "A factual draft.",
        "detected_claims": [{"claim_id": "c1", "text": "A factual draft."}],
    }))
    assert update["route"] == "error"
    assert update["verifier_result"]["status"] == "failed"
    assert "dummy-api-key-secret" not in str(update)


def test_verifier_route_success():
    assert _verifier_route({"verification_status": "verified"}) == "judge"


def test_verifier_route_error():
    assert _verifier_route({"route": "error"}) == "human_escalation"


def test_verifier_route_agent_failed():
    assert _verifier_route({"verification_status": "agent_failed"}) == "human_escalation"


def test_judge_route_accept():
    from orchestration.graph import _judge_route
    assert _judge_route({"judge_decision": "ACCEPT"}) == "memory"


def test_judge_route_correct():
    from orchestration.graph import _judge_route
    assert _judge_route({"judge_decision": "CORRECT", "active_agents": ["corrector"]}) == "corrector"


def test_judge_route_correction_unavailable_fails_closed():
    from orchestration.graph import _judge_route
    assert _judge_route({"judge_decision": "CORRECT", "active_agents": []}) == "human_escalation"


def test_corrector_and_reverifier_fail_closed_routes():
    assert _corrector_route({"route": "error"}) == "human_escalation"
    assert _corrector_route({"correction_result": {"status": "completed"}}) == "reverifier"
    assert _reverifier_route({"route": "error"}) == "human_escalation"
    assert _reverifier_route({"reverification_result": {"status": "failed"}}) == "human_escalation"
    assert _reverifier_route({"reverification_result": {"status": "completed"}}) == "judge"


def test_judge_route_verify_again():
    from orchestration.graph import _judge_route
    assert _judge_route({"judge_decision": "VERIFY_AGAIN"}) == "verifier"


def test_judge_route_reject():
    from orchestration.graph import _judge_route
    assert _judge_route({"judge_decision": "REJECT"}) == "reject"
