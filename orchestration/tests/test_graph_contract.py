from __future__ import annotations

from orchestration.graph import _detector_route, _judge_route, build_verification_graph
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


def test_detector_low_medium_bypass_verifier():
    assert _detector_route({"route": "accept"}) == "accept"


def test_detector_high_routes_to_verifier():
    assert _detector_route({"route": "verify"}) == "verify"


def test_detector_failure_routes_to_human_escalation():
    assert _detector_route({"route": "error"}) == "error"


def test_judge_accept_route():
    assert _judge_route({"judge": {"decision": "ACCEPT"}, "retry_count": 0}) == "accept"


def test_judge_correction_routes_to_corrector():
    assert (
        _judge_route({"judge": {"decision": "CORRECT"}, "retry_count": 0}) == "correct"
    )


def test_judge_retry_routes_back_to_verifier():
    assert (
        _judge_route(
            {"judge": {"decision": "VERIFY_AGAIN"}, "retry_count": 0, "max_retries": 2}
        )
        == "verify_again"
    )


def test_judge_retry_is_bounded():
    assert (
        _judge_route(
            {"judge": {"decision": "VERIFY_AGAIN"}, "retry_count": 2, "max_retries": 2}
        )
        == "retry_exhausted"
    )


def test_judge_reject_route():
    assert _judge_route({"judge": {"decision": "REJECT"}, "retry_count": 0}) == "reject"


def test_judge_human_escalation_route():
    assert (
        _judge_route({"judge": {"decision": "ESCALATE_HUMAN"}, "retry_count": 0})
        == "human_escalate"
    )


def test_judge_abstain_routes_to_human_escalation():
    assert (
        _judge_route({"judge": {"decision": "ABSTAIN"}, "retry_count": 0})
        == "human_escalate"
    )
