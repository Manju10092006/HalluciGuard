from __future__ import annotations

import pytest

from orchestration.graph import build_verification_graph
from orchestration.state import add_trace


def base_state():
    return {
        "execution_id": "ex",
        "request_id": "req",
        "user_query": "q",
        "llm_response": "draft",
        "draft_response": "draft",
        "domain": "general",
        "trace": [],
        "errors": [],
        "inter_agent_bus": [],
    }


@pytest.mark.asyncio
async def test_safe_response_reaches_accept_directly_without_verifier_memory_or_judge():
    graph = build_verification_graph(
        node_overrides={
            "detector": lambda s: {
                "route": "accept",
                "detector": {"next_action": "ACCEPT"},
                "trace": add_trace(s, "detector", "completed"),
            },
        }
    )
    result = await graph.ainvoke(base_state())
    nodes = [e["node"] for e in result["trace"]]
    assert "detector" in nodes
    assert "accept" in nodes
    assert "verifier" not in nodes
    assert "memory" not in nodes
    assert "judge" not in nodes
    assert "corrector" not in nodes


@pytest.mark.asyncio
async def test_high_risk_routes_verifier_then_memory_without_judge_or_corrector():
    async def verifier(s):
        return {
            "verifier": {"claim_evidence": []},
            "judge_pairs": [],
            "trace": add_trace(s, "verifier", "completed"),
        }

    async def memory(s):
        return {"memory": {"count": 0}, "trace": add_trace(s, "memory", "completed")}

    graph = build_verification_graph(
        node_overrides={
            "detector": lambda s: {
                "route": "verify",
                "trace": add_trace(s, "detector", "completed"),
            },
            "verifier": verifier,
            "memory": memory,
        }
    )
    result = await graph.ainvoke(base_state())
    nodes = [e["node"] for e in result["trace"]]
    assert "detector" in nodes
    assert "verifier" in nodes
    assert "memory" in nodes
    assert "judge" not in nodes
    assert "corrector" not in nodes


@pytest.mark.asyncio
async def test_generation_failure_goes_to_human_escalation_directly():
    async def generate(s):
        return {
            "verification_status": "generation_failed",
            "terminal_status": "failed",
            "route": "error",
            "llm_response": "",
            "trace": add_trace(s, "base_llm", "failed"),
        }

    def detector(_):
        raise AssertionError("detector must not run after generation failure")

    graph = build_verification_graph(
        node_overrides={"generate": generate, "detector": detector}
    )
    result = await graph.ainvoke(
        {**base_state(), "llm_response": "", "draft_response": ""}
    )
    nodes = [e["node"] for e in result["trace"]]
    assert "detector" not in nodes
    assert "human_escalation" in nodes
    assert "memory" not in nodes


@pytest.mark.asyncio
async def test_verifier_failure_routes_to_human_escalation_directly():
    async def verifier(s):
        return {
            "route": "error",
            "verification_status": "agent_failed",
            "errors": [{"node": "verifier", "message": "down"}],
            "trace": add_trace(s, "verifier", "failed"),
        }

    graph = build_verification_graph(
        node_overrides={
            "detector": lambda s: {
                "route": "verify",
                "trace": add_trace(s, "detector", "completed"),
            },
            "verifier": verifier,
        }
    )
    result = await graph.ainvoke(base_state())
    nodes = [e["node"] for e in result["trace"]]
    assert "verifier" in nodes
    assert "human_escalation" in nodes
    assert nodes[-1] == "human_escalation"
    assert "memory" not in nodes
    assert "judge" not in nodes
    assert "corrector" not in nodes
