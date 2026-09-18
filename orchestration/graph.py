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
    if state.get("route") == "error" or str(result.get("status", "")).lower() not in {"completed", "success", "skipped"}:
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


def _default_draft_from_query(state: HalluciGuardState) -> str:
    user_query = (state.get("user_query") or "").strip()
    if user_query:
        return user_query
    return "No draft response was generated."


async def _generate_node(state: HalluciGuardState) -> dict[str, Any]:
    response = state.get("llm_response", "")
    draft = response.strip() if isinstance(response, str) else ""
    if not draft:
        draft = _default_draft_from_query(state)
        return {
            "draft_response": draft,
            "llm_response": draft,
            "final_response": draft,
            "route": "detector",
            "verification_status": "verification_required",
            "trace": add_trace(state, "base_llm", "completed", latency_ms=0, fallback_used=True),
        }
    return {"draft_response": response, "final_response": response, "route": "detector"}


async def _detector_node(state: HalluciGuardState) -> dict[str, Any]:
    draft = state.get("llm_response") or state.get("draft_response") or ""
    risk = {
        "risk_level": "MEDIUM",
        "hallucination_probability": 0.5,
        "confidence": 0.5,
        "next_action": "verify",
    }
    return {
        "detector": risk,
        "detector_result": risk,
        "route": "verify",
        "verification_status": "verification_required",
        "trace": add_trace(state, "detector", "completed", latency_ms=0, risk_level=risk["risk_level"], draft_chars=len(draft)),
    }


async def _verifier_node(state: HalluciGuardState) -> dict[str, Any]:
    get_pipeline, SuspiciousClaim, VerifierInputV2 = _verifier_imports()
    started = start_timer()
    try:
        draft = state.get("llm_response") or state.get("draft_response") or ""
        if not draft.strip():
            raise ValueError("No draft response available for verification.")
        draft_claims = extract_claims(draft)
        if not draft_claims:
            draft_claims = [draft]
        payload = VerifierInputV2(
            query_id=state.get("request_id") or state.get("execution_id") or str(uuid.uuid4()),
            domain=state.get("domain", "general"),
            suspicious_claims=[SuspiciousClaim(claim_id=f"dc-{i + 1}", text=claim) for i, claim in enumerate(draft_claims)],
        )
        timeout = float(os.environ.get("VERIFIER_TIMEOUT_SECONDS", "120.0"))
        result = _dump(await asyncio.wait_for(get_pipeline().verify(payload), timeout=timeout))
        reports = result.get("claim_evidence") or result.get("claim_reports", [])
        has_contradiction = any("contradict" in str(item.get("verdict", "")).lower() for item in reports)
        has_verified = any(
            "verif" in str(item.get("verdict", "")).lower() and "unverif" not in str(item.get("verdict", "")).lower()
            for item in reports
        )
        status = "contradicted" if has_contradiction else "verified" if has_verified else "unverified"
        return {
            "verifier": result,
            "verifier_result": result,
            "draft_claims": draft_claims,
            "claims": reports,
            "verification_status": status,
            "verification_summary": {
                "claims_extracted": len(draft_claims),
                "claims_verified": int(has_verified),
                "claims_contradicted": int(has_contradiction),
                "overall_status": status,
            },
            "trace": add_trace(state, "verifier", "completed", latency_ms=elapsed_ms(started), claim_count=len(reports)),
        }
    except Exception as exc:
        return _failure_update(state, "verifier", exc, retryable=True)


def _judge_decision_from_verifier(state: HalluciGuardState) -> str:
    reports = state.get("claims") or []
    if not reports:
        return "REJECT"
    if any("contradict" in str(item.get("verdict", "")).lower() or "hallucinat" in str(item.get("verdict", "")).lower() for item in reports):
        return "REJECT"
    if any("verif" in str(item.get("verdict", "")).lower() and "unverif" not in str(item.get("verdict", "")).lower() for item in reports):
        return "ACCEPT"
    return "REJECT"


async def _judge_node(state: HalluciGuardState) -> dict[str, Any]:
    decision = _judge_decision_from_verifier(state)
    result = {
        "judge_decision": decision,
        "judge": {"decision": decision, "status": "completed"},
        "judge_result": {"decision": decision, "status": "completed"},
        "judge_summary": {"status": "pass" if decision == "ACCEPT" else "reject", "decision": decision},
        "route": "memory" if decision == "ACCEPT" else "reject",
        "trace": add_trace(state, "judge", "completed", latency_ms=0, decision=decision),
    }
    return result


async def _corrector_node(state: HalluciGuardState) -> dict[str, Any]:
    corrected = state.get("llm_response") or state.get("draft_response") or ""
    result = {
        "correction_result": {
            "status": "skipped",
            "reason": "correction workflow unavailable; continuing in fail-closed mode",
            "corrected_text": corrected,
        },
        "correction_requested": False,
        "trace": add_trace(state, "corrector", "completed", latency_ms=0, status="skipped"),
    }
    return result


async def _reverifier_node(state: HalluciGuardState) -> dict[str, Any]:
    result = {
        "reverification_result": {
            "status": "skipped",
            "reason": "reverification workflow unavailable; using fail-closed policy",
            "verifier_result": state.get("verifier_result") or {},
        },
        "trace": add_trace(state, "reverifier", "completed", latency_ms=0, status="skipped"),
    }
    return result


def _accept_node(state: HalluciGuardState) -> dict[str, Any]:
    return {
        "final_response": state.get("llm_response", "") or state.get("draft_response", ""),
        "terminal_status": "accepted",
        "verification_status": "accepted",
        "trace": add_trace(state, "accept", "completed", latency_ms=0),
    }


def _reject_node(state: HalluciGuardState) -> dict[str, Any]:
    return {
        "final_response": "The draft response could not be safely verified and has been rejected.",
        "terminal_status": "rejected",
        "verification_status": "rejected",
        "trace": add_trace(state, "reject", "completed", latency_ms=0),
    }


def _human_escalation_node(state: HalluciGuardState) -> dict[str, Any]:
    return {
        "final_response": "This response requires human review before it can be delivered.",
        "terminal_status": "human_review",
        "verification_status": "human_review_required",
        "trace": add_trace(state, "human_escalation", "completed", latency_ms=0),
    }


def _memory_node(state: HalluciGuardState) -> dict[str, Any]:
    return {
        "memory": {"stored": [], "count": 0},
        "terminal_status": state.get("terminal_status", "accepted"),
        "trace": add_trace(state, "memory", "completed", latency_ms=0, stored_count=0),
    }


def build_verification_graph(node_overrides: dict[str, Callable[..., Any]] | None = None):
    nodes = {
        "generate": _generate_node,
        "detector": _detector_node,
        "verifier": _verifier_node,
        "judge": _judge_node,
        "corrector": _corrector_node,
        "reverifier": _reverifier_node,
        "accept": _accept_node,
        "reject": _reject_node,
        "human_escalation": _human_escalation_node,
        "memory": _memory_node,
    }
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


async def run_verification(
    user_query: str,
    llm_response: str | None = None,
    domain: str = "general",
    request_id: str | None = None,
    generation_mode: str = "normal",
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Execute the verification graph with a fail-closed default pipeline."""
    execution_id = str(uuid.uuid4())
    initial_state: HalluciGuardState = {
        "execution_id": execution_id,
        "request_id": request_id or execution_id,
        "user_query": user_query or "",
        "llm_response": llm_response or "",
        "draft_response": llm_response or "",
        "generation_mode": generation_mode,
        "conversation_history": conversation_history or [],
        "domain": domain,
        "max_retries": 2,
        "retry_count": 0,
        "correction_attempt_count": 0,
        "reverification_attempt_count": 0,
        "active_agents": ["base_llm", "detector", "verifier", "judge", "corrector", "reverifier", "memory"],
        "disabled_agents": [],
        "trace": [],
        "errors": [],
        "inter_agent_bus": [],
    }

    graph = get_verification_graph()
    result = await graph.ainvoke(initial_state)

    if not result.get("final_response"):
        result["final_response"] = result.get("llm_response") or result.get("draft_response") or user_query or "No response generated."

    if result.get("terminal_status") is None:
        verification_status = str(result.get("verification_status") or "").lower()
        if verification_status in {"accepted", "verified"}:
            result["terminal_status"] = "accepted"
        elif verification_status in {"rejected", "contradicted"}:
            result["terminal_status"] = "rejected"
        elif verification_status in {"human_review_required", "human_review"}:
            result["terminal_status"] = "human_review"
        else:
            result["terminal_status"] = "accepted" if result.get("judge_decision") == "ACCEPT" else "rejected"

    return result


__all__ = [
    "build_verification_graph",
    "get_verification_graph",
    "run_verification",
]

