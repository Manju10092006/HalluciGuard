from __future__ import annotations

from orchestration.graph import _detector_route, _judge_route, build_verification_graph


def test_graph_compiles():
    graph = build_verification_graph()
    assert graph is not None


def test_detector_low_medium_bypass_verifier():
    assert _detector_route({"route": "accept"}) == "accept"


def test_detector_high_routes_to_verifier():
    assert _detector_route({"route": "verify"}) == "verify"


def test_judge_correction_routes_to_corrector():
    assert _judge_route({"judge": {"decision": "CORRECT"}, "retry_count": 0}) == "correct"


def test_judge_retry_routes_back_to_verifier():
    assert _judge_route({"judge": {"decision": "VERIFY_AGAIN"}, "retry_count": 0}) == "verify_again"


def test_judge_retry_is_bounded():
    assert _judge_route({"judge": {"decision": "VERIFY_AGAIN"}, "retry_count": 2}) == "finish"


def test_judge_accept_finishes():
    assert _judge_route({"judge": {"decision": "ACCEPT"}, "retry_count": 0}) == "finish"
