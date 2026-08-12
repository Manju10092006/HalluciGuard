from __future__ import annotations

import os
import sys
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from .interbus import emit_message
from .state import HalluciGuardState, add_error, add_trace, elapsed_ms, start_timer, utc_now
from .supervisor import supervisor_node, supervisor_route

MAX_VERIFICATION_RETRIES = int(os.getenv("HALLUCIGUARD_MAX_VERIFICATION_RETRIES", "2"))


# Judge and Corrector deliberately remain in agents/ but are not active graph nodes.
ACTIVE_AGENTS = ("detector", "verifier", "memory")
DISABLED_AGENTS = ("judge", "corrector")


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(v) for v in value]
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
        "verification_status": "agent_failed",
        "terminal_status": "failed" if not retryable else state.get("terminal_status"),
        "supervisor_phase": "failure",
        "supervisor_route": "failure",
        "updated_at": utc_now(),
        "trace": add_trace(
            state,
            node,
            "failed",
            error_type=type(exc).__name__,
            retryable=retryable,
        ),
    }


def _supervisor_start_node(state: HalluciGuardState) -> dict[str, Any]:
    now = utc_now()
    update = supervisor_node({**state, "supervisor_phase": "start", "updated_at": now})
    return {**update, "current_node": "supervisor", "supervisor_phase": "start", "updated_at": now}


def _detector_node(state: HalluciGuardState) -> dict[str, Any]:
    from agents.detector_agent.detector import DetectorAgent

    node_start = start_timer()
    try:
        detector = _dump(DetectorAgent().detect(state["user_query"], state["llm_response"]))
        next_action = str(detector.get("next_action", ""))
        risk_level = str(detector.get("risk_level", ""))
        route = "verify" if next_action.lower().endswith("verify") else "accept"
        if not next_action:
            raise RuntimeError("Detector returned no next_action; refusing to invent routing")

        bus = emit_message(
            state,
            source_agent="detector",
            target_agent="verifier" if route == "verify" else "memory",
            message_type="SUSPICIOUS_CLAIMS" if route == "verify" else "DETECTOR_ACCEPT",
            payload={
                "hallucination_probability": detector.get("hallucination_probability"),
                "confidence_score": detector.get("confidence_score"),
                "risk_level": risk_level,
                "next_action": next_action,
            },
        )
        return {
            "detector": detector,
            "route": route,
            "hallucination_probability": float(detector.get("hallucination_probability", 0.0)),
            "confidence": float(detector.get("confidence_score", 0.0)),
            "verification_status": "verification_required" if route == "verify" else "detector_accept",
            "inter_agent_bus": bus,
            "current_node": "detector",
            "supervisor_phase": "after_detector",
            "updated_at": utc_now(),
            "trace": add_trace(
                state,
                "detector",
                "completed",
                latency_ms=elapsed_ms(node_start),
                route=route,
                risk_level=risk_level,
            ),
        }
    except Exception as exc:
        return _failure_update(state, "detector", exc)


def _supervisor_after_detector_node(state: HalluciGuardState) -> dict[str, Any]:
    update = supervisor_node({**state, "supervisor_phase": "after_detector"})
    return {**update, "current_node": "supervisor", "supervisor_phase": "after_detector", "updated_at": utc_now()}


def _verifier_imports():
    verifier_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents", "verifier_agent"))
    if verifier_dir not in sys.path:
        sys.path.insert(0, verifier_dir)
    from api.pipeline import VerificationPipeline
    from schemas.models import SuspiciousClaim, VerifierInputV2

    return VerificationPipeline, SuspiciousClaim, VerifierInputV2


async def _verifier_node(state: HalluciGuardState) -> dict[str, Any]:
    VerificationPipeline, SuspiciousClaim, VerifierInputV2 = _verifier_imports()
    node_start = start_timer()
    try:
        payload = VerifierInputV2(
            query_id=state.get("request_id") or state.get("execution_id") or str(uuid.uuid4()),
            domain=state.get("domain", "general"),
            suspicious_claims=[SuspiciousClaim(claim_id="c1", text=state["llm_response"])],
        )
        verifier = _dump(await VerificationPipeline().verify(payload))
        claims: list[dict[str, Any]] = []
        evidence_all: list[dict[str, Any]] = []
        nli_results: list[dict[str, Any]] = []

        for report in verifier.get("claim_evidence", []):
            claims.append(
                {
                    "claim_id": report.get("claim_id"),
                    "text": report.get("claim_text"),
                    "verdict": report.get("verdict"),
                }
            )
            for evidence in report.get("evidence", []):
                evidence_all.append(evidence)
                nli_results.append(
                    {
                        "claim": report.get("claim_text", ""),
                        "label": evidence.get("entailment_label"),
                        "score": evidence.get("entailment_score"),
                        "degraded": bool(evidence.get("degraded", False)),
                    }
                )

        if not verifier.get("claim_evidence"):
            raise RuntimeError("Verifier returned no claim_evidence; refusing to claim verification")

        bus = emit_message(
            state,
            source_agent="verifier",
            target_agent="memory",
            message_type="VERIFICATION_RESULT",
            payload={
                "claim_count": len(claims),
                "evidence_count": len(evidence_all),
                "verdicts": [c.get("verdict") for c in claims],
                "nli_results": nli_results,
            },
        )
        return {
            "verifier": verifier,
            "claims": claims,
            "evidence": evidence_all,
            "retrieved_evidence": evidence_all,
            "ranked_evidence": evidence_all,
            "nli_results": nli_results,
            "verification_status": "verified",
            "inter_agent_bus": bus,
            "current_node": "verifier",
            "supervisor_phase": "after_verifier",
            "updated_at": utc_now(),
            "trace": add_trace(
                state,
                "verifier",
                "completed",
                latency_ms=elapsed_ms(node_start),
                claim_count=len(claims),
                evidence_count=len(evidence_all),
            ),
        }
    except Exception as exc:
        return _failure_update(state, "verifier", exc, retryable=True)


def _supervisor_after_verifier_node(state: HalluciGuardState) -> dict[str, Any]:
    update = supervisor_node({**state, "supervisor_phase": "after_verifier"})
    return {**update, "current_node": "supervisor", "supervisor_phase": "after_verifier", "updated_at": utc_now()}


def _retry_verifier_node(state: HalluciGuardState) -> dict[str, Any]:
    retry_count = int(state.get("retry_count", 0)) + 1
    return {
        "retry_count": retry_count,
        "verification_status": "retry_scheduled",
        "supervisor_phase": "after_detector",
        "route": "verify",
        "updated_at": utc_now(),
        "trace": add_trace(state, "verifier_retry", "scheduled", retry_count=retry_count),
    }


def _memory_node(state: HalluciGuardState) -> dict[str, Any]:
    from agents.memory_agent.memory.memory_agent import MemoryAgent
    from agents.memory_agent.schemas.models import StoreFactRequest

    node_start = start_timer()
    claim_evidence = state.get("verifier", {}).get("claim_evidence", [])
    if not claim_evidence:
        memory = {
            "status": "skipped",
            "stored": [],
            "count": 0,
            "reason": "no_verifier_claims_to_persist",
        }
        return {
            "memory": memory,
            "final_response": state.get("llm_response", ""),
            "terminal_status": "partial_success",
            "current_node": "memory",
            "updated_at": utc_now(),
            "trace": add_trace(state, "memory", "skipped", latency_ms=elapsed_ms(node_start), reason=memory["reason"]),
        }

    try:
        memory_agent = MemoryAgent()
        await memory_agent.initialize()
        stored: list[Any] = []
        try:
            for report in claim_evidence:
                evidence = report.get("evidence", [])
                request = StoreFactRequest(
                    claim_text=str(report.get("claim_text", "")),
                    domain=state.get("domain", "general"),
                    verdict=str(report.get("verdict", "insufficient_evidence")),
                    evidence=[
                        {
                            "source_id": e.get("source", ""),
                            "title": e.get("title", ""),
                            "url": e.get("url"),
                            "snippet": e.get("snippet", ""),
                        }
                        for e in evidence
                    ],
                    source_ids=[str(e.get("source", "")) for e in evidence if e.get("source")],
                    confidence=float(report.get("confidence_score", report.get("trust_score", 0.0))),
                    metadata={
                        "execution_id": state.get("execution_id", ""),
                        "request_id": state.get("request_id", ""),
                    },
                )
                stored.append(_dump(await memory_agent.store_fact(request)))
        finally:
            await memory_agent.close()

        memory = {"status": "completed", "stored": stored, "count": len(stored)}
        bus = emit_message(
            state,
            source_agent="memory",
            target_agent="supervisor",
            message_type="MEMORY_WRITE_RESULT",
            payload={"status": "completed", "count": len(stored)},
        )
        return {
            "memory": memory,
            "inter_agent_bus": bus,
            "final_response": state.get("llm_response", ""),
            "terminal_status": "success",
            "current_node": "memory",
            "supervisor_phase": "after_memory",
            "updated_at": utc_now(),
            "trace": add_trace(state, "memory", "completed", latency_ms=elapsed_ms(node_start), stored_count=len(stored)),
        }
    except Exception as exc:
        update = _failure_update(state, "memory", exc)
        update["terminal_status"] = "partial_success"
        update["final_response"] = state.get("llm_response", "")
        return update


def _terminal_failure_node(state: HalluciGuardState) -> dict[str, Any]:
    return {
        "terminal_status": "failed",
        "final_response": state.get("llm_response", ""),
        "current_node": "terminal_failure",
        "updated_at": utc_now(),
        "trace": add_trace(state, "terminal_failure", "completed", errors=len(state.get("errors", []))),
    }


def _after_detector_route(state: HalluciGuardState) -> str:
    return supervisor_route(state)


def _after_verifier_route(state: HalluciGuardState) -> str:
    if state.get("verification_status") == "verified":
        return "memory"
    if int(state.get("retry_count", 0)) < int(state.get("max_retries", MAX_VERIFICATION_RETRIES)):
        return "retry"
    return "failure"


def build_verification_graph(node_overrides: dict[str, Callable[..., Any]] | None = None):
    """Build the active production graph: Detector -> Verifier -> Memory."""
    nodes: dict[str, Callable[..., Any]] = {
        "supervisor_start": _supervisor_start_node,
        "detector": _detector_node,
        "supervisor_after_detector": _supervisor_after_detector_node,
        "verifier": _verifier_node,
        "supervisor_after_verifier": _supervisor_after_verifier_node,
        "retry": _retry_verifier_node,
        "memory": _memory_node,
        "terminal_failure": _terminal_failure_node,
    }
    if node_overrides:
        nodes.update(node_overrides)

    graph = StateGraph(HalluciGuardState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "supervisor_start")
    graph.add_conditional_edges("supervisor_start", supervisor_route, {"detector": "detector", "failure": "terminal_failure"})
    graph.add_edge("detector", "supervisor_after_detector")
    graph.add_conditional_edges(
        "supervisor_after_detector",
        _after_detector_route,
        {"verifier": "verifier", "memory": "memory", "failure": "terminal_failure"},
    )
    graph.add_edge("verifier", "supervisor_after_verifier")
    graph.add_conditional_edges(
        "supervisor_after_verifier",
        _after_verifier_route,
        {"memory": "memory", "retry": "retry", "failure": "terminal_failure"},
    )
    graph.add_edge("retry", "verifier")
    graph.add_edge("memory", END)
    graph.add_edge("terminal_failure", END)
    return graph.compile()


_GRAPH = None


def get_verification_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_verification_graph()
    return _GRAPH


async def run_verification(
    user_query: str,
    llm_response: str,
    domain: str = "general",
    request_id: str | None = None,
) -> HalluciGuardState:
    execution_id = str(uuid.uuid4())
    now = utc_now()
    initial: HalluciGuardState = {
        "execution_id": execution_id,
        "request_id": request_id or execution_id,
        "user_query": user_query,
        "llm_response": llm_response,
        "domain": domain,
        "retry_count": 0,
        "max_retries": MAX_VERIFICATION_RETRIES,
        "created_at": now,
        "updated_at": now,
        "trace": [],
        "errors": [],
        "inter_agent_bus": [],
        "terminal_status": "failed",
        "audit": {
            "graph": "halluciguard_langgraph_active_detector_verifier_memory",
            "active_agents": list(ACTIVE_AGENTS),
            "disabled_agents": list(DISABLED_AGENTS),
            "judge_executed": False,
            "corrector_executed": False,
            "max_retries": MAX_VERIFICATION_RETRIES,
        },
    }
    return await get_verification_graph().ainvoke(initial)
