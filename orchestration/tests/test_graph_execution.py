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
        "domain": "general",
        "retry_count": 0,
        "max_retries": 1,
        "trace": [],
        "errors": [],
    }


@pytest.mark.asyncio
async def test_safe_response_reaches_memory_without_verifier():
    async def memory(state):
        return {
            "memory": {"count": 0},
            "trace": add_trace(state, "memory", "completed"),
        }

    graph = build_verification_graph(
        node_overrides={
            "detector": lambda s: {
                "route": "accept",
                "detector": {"next_action": "ACCEPT"},
                "trace": add_trace(s, "detector", "completed"),
            },
            "memory": memory,
        }
    )
    result = await graph.ainvoke(base_state())
    assert [e["node"] for e in result["trace"]] == ["detector", "accept", "memory"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision,terminal_node",
    [
        ("ACCEPT", "accept"),
        ("CORRECT", "corrector"),
        ("REJECT", "reject"),
        ("ESCALATE_HUMAN", "human_escalation"),
    ],
)
async def test_judge_terminal_routes(decision, terminal_node):
    async def verifier(s):
        return {
            "verifier": {},
            "judge_pairs": [],
            "trace": add_trace(s, "verifier", "completed"),
        }

    def judge(s):
        return {
            "judge": {"decision": decision},
            "judge_decision": decision,
            "trace": add_trace(s, "judge", "completed"),
        }

    async def corrector(s):
        return {
            "corrector": {},
            "final_response": "corrected",
            "trace": add_trace(s, "corrector", "completed"),
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
            "judge": judge,
            "corrector": corrector,
            "memory": memory,
        }
    )
    result = await graph.ainvoke(base_state())
    assert terminal_node in [e["node"] for e in result["trace"]]
    assert result["trace"][-1]["node"] == "memory"


@pytest.mark.asyncio
async def test_verify_again_loop_is_bounded_and_escalates():
    async def verifier(s):
        return {
            "verifier": {},
            "judge_pairs": [],
            "trace": add_trace(s, "verifier", "completed"),
        }

    def judge(s):
        return {
            "judge": {"decision": "VERIFY_AGAIN"},
            "judge_decision": "VERIFY_AGAIN",
            "trace": add_trace(s, "judge", "completed"),
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
            "judge": judge,
            "memory": memory,
        }
    )
    result = await graph.ainvoke(base_state())
    nodes = [e["node"] for e in result["trace"]]
    assert nodes.count("verifier") == 2
    assert "retry_exhausted" in nodes
    assert "human_escalation" in nodes
    assert result["retry_count"] == 1


@pytest.mark.asyncio
async def test_agent_failure_is_graceful():
    def detector(_):
        return {
            "route": "error",
            "errors": [{"node": "detector", "message": "boom"}],
            "trace": [{"node": "detector", "status": "failed"}],
        }

    async def memory(s):
        return {"memory": {"count": 0}, "trace": add_trace(s, "memory", "completed")}

    graph = build_verification_graph(
        node_overrides={"detector": detector, "memory": memory}
    )
    result = await graph.ainvoke(base_state())
    assert [e["node"] for e in result["trace"]][-2:] == ["human_escalation", "memory"]


@pytest.mark.asyncio
async def test_verifier_failure_routes_to_human_escalation_without_judge():
    async def verifier(s):
        return {
            "route": "error",
            "verification_status": "agent_failed",
            "errors": [{"node": "verifier", "message": "down"}],
            "trace": add_trace(s, "verifier", "failed"),
        }

    def judge(s):
        raise AssertionError("judge must not run after verifier infrastructure failure")

    async def memory(s):
        return {"memory": {"count": 0}, "trace": add_trace(s, "memory", "completed")}

    graph = build_verification_graph(
        node_overrides={
            "detector": lambda s: {
                "route": "verify",
                "trace": add_trace(s, "detector", "completed"),
            },
            "verifier": verifier,
            "judge": judge,
            "memory": memory,
        }
    )
    result = await graph.ainvoke(base_state())
    nodes = [e["node"] for e in result["trace"]]
    assert "judge" not in nodes
    assert nodes[-2:] == ["human_escalation", "memory"]


@pytest.mark.asyncio
async def test_corrector_failure_routes_to_human_escalation_without_fake_correction():
    async def verifier(s):
        return {
            "verifier": {},
            "judge_pairs": [],
            "trace": add_trace(s, "verifier", "completed"),
        }

    def judge(s):
        return {
            "judge": {"decision": "CORRECT"},
            "judge_decision": "CORRECT",
            "trace": add_trace(s, "judge", "completed"),
        }

    async def corrector(s):
        return {
            "route": "error",
            "verification_status": "agent_failed",
            "errors": [{"node": "corrector", "message": "model unavailable"}],
            "trace": add_trace(s, "corrector", "failed"),
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
            "judge": judge,
            "corrector": corrector,
            "memory": memory,
        }
    )
    result = await graph.ainvoke(base_state())
    nodes = [e["node"] for e in result["trace"]]
    assert nodes[-2:] == ["human_escalation", "memory"]
    assert (
        result["final_response"]
        == "This response requires human review before it can be delivered."
    )
