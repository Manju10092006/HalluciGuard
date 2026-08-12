from __future__ import annotations

import pytest

from orchestration.graph import build_verification_graph
from orchestration.interbus import emit_message


@pytest.mark.asyncio
async def test_active_graph_excludes_judge_and_corrector() -> None:
    async def detector(state):
        return {
            "route": "verify",
            "verification_status": "verification_required",
            "detector": {
                "hallucination_probability": 0.91,
                "confidence_score": 0.91,
                "risk_level": "HIGH",
                "next_action": "VERIFY",
            },
            "supervisor_phase": "after_detector",
            "updated_at": state["updated_at"],
        }

    async def verifier(state):
        return {
            "verification_status": "verified",
            "verifier": {
                "claim_evidence": [
                    {
                        "claim_id": "c1",
                        "claim_text": "The Earth orbits the Sun.",
                        "verdict": "verified",
                        "evidence": [
                            {
                                "source": "test",
                                "snippet": "The Earth orbits the Sun.",
                                "entailment_label": "entailment",
                                "entailment_score": 0.99,
                            }
                        ],
                    }
                ]
            },
            "supervisor_phase": "after_verifier",
            "updated_at": state["updated_at"],
        }

    async def memory(state):
        return {
            "memory": {"status": "completed", "count": 1},
            "terminal_status": "success",
            "final_response": state["llm_response"],
            "updated_at": state["updated_at"],
        }

    async def supervisor(state):
        return {
            "supervisor_route": "detector",
            "supervisor_phase": state.get("supervisor_phase", "start"),
            "updated_at": state["updated_at"],
        }

    graph = build_verification_graph(
        {
            "supervisor_start": supervisor,
            "detector": detector,
            "supervisor_after_detector": lambda state: {"supervisor_route": "verifier"},
            "verifier": verifier,
            "supervisor_after_verifier": lambda state: {"supervisor_route": "memory"},
            "memory": memory,
        }
    )

    state = await graph.ainvoke(
        {
            "execution_id": "test-execution",
            "request_id": "test-request",
            "user_query": "test",
            "llm_response": "test response",
            "domain": "general",
            "retry_count": 0,
            "max_retries": 1,
            "created_at": "now",
            "updated_at": "now",
            "trace": [],
            "errors": [],
            "inter_agent_bus": [],
        }
    )

    assert state["terminal_status"] == "success"
    assert "judge" not in [event["node"] for event in state.get("trace", [])]
    assert "corrector" not in [event["node"] for event in state.get("trace", [])]


def test_inter_agent_bus_message_is_structured() -> None:
    state = {"execution_id": "exec-1", "inter_agent_bus": []}
    bus = emit_message(
        state,
        source_agent="detector",
        target_agent="verifier",
        message_type="SUSPICIOUS_CLAIMS",
        payload={"risk_level": "HIGH"},
    )
    message = bus[0]
    assert message["execution_id"] == "exec-1"
    assert message["source_agent"] == "detector"
    assert message["target_agent"] == "verifier"
    assert message["message_type"] == "SUSPICIOUS_CLAIMS"
    assert message["payload"]["risk_level"] == "HIGH"
