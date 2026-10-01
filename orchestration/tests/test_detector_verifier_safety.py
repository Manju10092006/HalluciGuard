"""Production graph failure routing, independent of model weights or network."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from orchestration import graph as graph_module
from orchestration.state import add_trace


def state():
    return {"execution_id": "test", "request_id": "test", "user_query": "Who?",
            "llm_response": "A factual draft.", "draft_response": "A factual draft.",
            "domain": "general", "retry_count": 0, "trace": [], "errors": [],
            "inter_agent_bus": []}


@pytest.mark.asyncio
async def test_detector_exception_still_calls_verifier(monkeypatch):
    import agents.detector_agent.detector as detector_module
    class FailingDetector:
        def detect(self, *args):
            raise RuntimeError("dummy-secret")
    monkeypatch.setattr(detector_module, "DetectorAgent", FailingDetector)
    calls = []
    async def verifier(s):
        calls.append(s["llm_response"])
        return {"verifier": {"claim_evidence": []},
                "verification_status": "unverified_insufficient_evidence",
                "trace": add_trace(s, "verifier", "completed")}
    async def memory(s):
        return {"memory": {}, "trace": add_trace(s, "memory", "completed")}
    graph = graph_module.build_verification_graph(
        node_overrides={"verifier": verifier, "memory": memory})
    result = await graph.ainvoke(state())
    assert calls == ["A factual draft."]
    assert "dummy-secret" not in str(result.get("errors"))


@pytest.mark.asyncio
async def test_verifier_exception_is_not_a_verified_result(monkeypatch):
    calls = []
    class Pipeline:
        async def verify(self, payload):
            calls.append(payload)
            raise TimeoutError("dummy-secret")
    class Payload:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    monkeypatch.setattr(graph_module, "_verifier_imports",
                        lambda: (Pipeline, Payload, Payload))
    result = await graph_module._verifier_node(state())
    assert len(calls) == 1
    assert result["route"] == "error"
    assert result["verification_status"] == "agent_failed"
    assert "verified" not in str(result.get("verifier", ""))
    assert "dummy-secret" not in str(result)


def test_malformed_or_low_risk_detector_cannot_select_fast_path():
    for detector_result in ({"route": "accept"}, {"route": "error"},
                            {"route": "verify"}):
        assert graph_module._detector_route({**state(), **detector_result}) == "verifier"
