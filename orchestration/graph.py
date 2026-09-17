from __future__ import annotations

import asyncio
import os
import sys
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from .claim_extraction import extract_claims
from .state import (
    HalluciGuardState,
    add_bus_message,
    add_error,
    add_trace,
    elapsed_ms,
    start_timer,
    utc_now,
)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(item) for item in value]
    return value


def _failure_update(
    state: HalluciGuardState,
    node: str,
    exc: BaseException,
    *,
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "errors": add_error(state, node, exc, retryable=retryable),
        "error": f"{node} failed: {type(exc).__name__}: {exc}",
        "route": "error",
        "terminal_status": "human_review" if retryable else "fallback",
        "verification_status": "agent_failed",
        "inter_agent_bus": add_bus_message(
            state,
            node,
            "supervisor",
            "ERROR_EVENT",
            {"error_type": type(exc).__name__, "message": str(exc)},
            status="failed",
        ),
        "updated_at": utc_now(),
        "trace": add_trace(state, node, "failed", error_type=type(exc).__name__),
    }


def _generate_route(state: HalluciGuardState) -> str:
    return "human_escalation" if state.get("route") == "error" or not state.get("llm_response") else "detector"


def _detector_route(state: HalluciGuardState) -> str:
    if state.get("route") == "error":
        return "human_escalation"
    return "verifier" if state.get("route") == "verify" else "accept"


def _verifier_route(state: HalluciGuardState) -> str:
    return "human_escalation" if state.get("route") == "error" or state.get("verification_status") == "agent_failed" else "judge"


def _corrector_route(state: HalluciGuardState) -> str:
    return "human_escalation" if state.get("route") == "error" or not state.get("correction_result") else "reverifier"


def _reverifier_route(state: HalluciGuardState) -> str:
    result = state.get("reverification_result") or {}
    if state.get("route") == "error" or str(result.get("status", "")).lower() not in {"completed", "success"}:
        return "human_escalation"
    return "judge"


def _judge_route(state: HalluciGuardState) -> str:
    if state.get("route") == "error":
        return "human_escalation"
    decision = str(state.get("judge_decision", "ACCEPT")).upper()
    if decision == "ACCEPT":
        return "memory"
    if decision == "REJECT":
        return "reject"
    if decision == "ABSTAIN":
        return "human_escalation"
    if decision == "VERIFY_AGAIN":
        return "human_escalation" if int(state.get("retry_count", 0)) >= int(state.get("max_retries", 2)) else "verifier"
    if decision == "CORRECT":
        request = state.get("correction_request")
        active = state.get("active_agents")
        if not request or (active is not None and "corrector" not in active):
            return "human_escalation"
        return "reject" if int(state.get("correction_attempt_count", 0)) >= int(state.get("max_retries", 2)) else "corrector"
    return "human_escalation"


def _verifier_imports():
    verifier_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents", "verifier_agent"))
    if verifier_dir not in sys.path:
        sys.path.insert(0, verifier_dir)
    from api.pipeline import get_pipeline
    from schemas.models import SuspiciousClaim, VerifierInputV2
    return get_pipeline, SuspiciousClaim, VerifierInputV2


async def _generate_node(state: HalluciGuardState) -> dict[str, Any]:
    response = state.get("llm_response", "")
    return {"draft_response": response, "final_response": response} if response else _failure_update(state, "base_llm", ValueError("No draft response available"))


async def _detector_node(state: HalluciGuardState) -> dict[str, Any]:
    return {"route": "verify", "verification_status": "verification_required"}


async def _verifier_node(state: HalluciGuardState) -> dict[str, Any]:
    get_pipeline, SuspiciousClaim, VerifierInputV2 = _verifier_imports()
    started = start_timer()
    try:
        draft = state.get("llm_response") or state.get("draft_response") or ""
        if not draft.strip():
            raise ValueError("No draft response available for verification.")
        draft_claims = extract_claims(draft)
        if not draft_claims:
            raise ValueError("No factual claims could be extracted from the draft response.")
        payload = VerifierInputV2(
            query_id=state.get("request_id") or state.get("execution_id") or str(uuid.uuid4()),
            domain=state.get("domain", "general"),
            suspicious_claims=[SuspiciousClaim(claim_id=f"dc-{i + 1}", text=claim) for i, claim in enumerate(draft_claims)],
        )
        timeout = float(os.environ.get("VERIFIER_TIMEOUT_SECONDS", "120.0"))
        result = _dump(await asyncio.wait_for(get_pipeline().verify(payload), timeout=timeout))
        reports = result.get("claim_evidence") or result.get("claim_reports", [])
        has_contradiction = any("contradict" in str(item.get("verdict", "")).lower() for item in reports)
        has_verified = any("verif" in str(item.get("verdict", "")).lower() and "unverif" not in str(item.get("verdict", "")).lower() for item in reports)
        status = "contradicted" if has_contradiction else "verified" if has_verified else "unverified"
        return {
            "verifier": result,
            "verifier_result": result,
            "draft_claims": draft_claims,
            "claims": reports,
            "verification_status": status,
            "verification_summary": {"claims_extracted": len(draft_claims), "claims_verified": int(has_verified), "claims_contradicted": int(has_contradiction), "overall_status": status},
            "trace": add_trace(state, "verifier", "completed", latency_ms=elapsed_ms(started), claim_count=len(reports)),
        }
    except Exception as exc:
        return _failure_update(state, "verifier", exc, retryable=True)


async def _judge_node(state: HalluciGuardState) -> dict[str, Any]:
    return {"judge_decision": "ACCEPT", "route": "memory"}


async def _corrector_node(state: HalluciGuardState) -> dict[str, Any]:
    return _failure_update(state, "corrector", RuntimeError("Correction implementation unavailable"))


async def _reverifier_node(state: HalluciGuardState) -> dict[str, Any]:
    return _failure_update(state, "reverifier", RuntimeError("Reverification implementation unavailable"))


def _accept_node(state: HalluciGuardState) -> dict[str, Any]:
    return {"final_response": state.get("llm_response", ""), "terminal_status": "accepted", "verification_status": "accepted"}


def _reject_node(state: HalluciGuardState) -> dict[str, Any]:
    return {"final_response": "The draft response could not be safely verified and has been rejected.", "terminal_status": "rejected", "verification_status": "rejected"}


def _human_escalation_node(state: HalluciGuardState) -> dict[str, Any]:
    return {"final_response": "This response requires human review before it can be delivered.", "terminal_status": "human_review", "verification_status": "human_review_required"}


def _memory_node(state: HalluciGuardState) -> dict[str, Any]:
    return {"memory": {"stored": [], "count": 0}, "terminal_status": state.get("terminal_status", "accepted")}


def build_verification_graph(node_overrides: dict[str, Callable[..., Any]] | None = None):
    nodes = {"generate": _generate_node, "detector": _detector_node, "verifier": _verifier_node, "judge": _judge_node, "corrector": _corrector_node, "reverifier": _reverifier_node, "accept": _accept_node, "reject": _reject_node, "human_escalation": _human_escalation_node, "memory": _memory_node}
    nodes.update(node_overrides or {})
    graph = StateGraph(HalluciGuardState)
    for name, function in nodes.items():
        graph.add_node(name, function)
    graph.add_edge(START, "generate")
    graph.add_conditional_edges("generate", _generate_route, {"detector": "detector", "human_escalation": "human_escalation"})
    graph.add_conditional_edges("detector", _detector_route, {"verifier": "verifier", "accept": "accept", "human_escalation": "human_escalation"})
    graph.add_conditional_edges("verifier", _verifier_route, {"judge": "judge", "human_escalation": "human_escalation"})
    graph.add_conditional_edges("judge", _judge_route, {"memory": "memory", "corrector": "corrector", "verifier": "verifier", "reject": "reject", "human_escalation": "human_escalation"})
    graph.add_conditional_edges("corrector", _corrector_route, {"reverifier": "reverifier", "human_escalation": "human_escalation"})
    graph.add_conditional_edges("reverifier", _reverifier_route, {"judge": "judge", "human_escalation": "human_escalation"})
    for terminal in ("accept", "reject", "human_escalation"):
        graph.add_edge(terminal, "memory")
    graph.add_edge("memory", END)
    return graph.compile()


_GRAPH = None


def get_verification_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_verification_graph()
    return _GRAPH
