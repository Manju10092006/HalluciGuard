from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
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

logger = logging.getLogger(__name__)


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


def _default_detector_risk() -> dict[str, Any]:
    """Fail-closed fallback risk profile used when the detector cannot run."""
    return {
        "risk_level": "MEDIUM",
        "hallucination_probability": 0.5,
        "confidence_score": 0.5,
        "next_action": "verify",
        "model_source": "fallback-heuristic",
    }


async def _detector_node(state: HalluciGuardState) -> dict[str, Any]:
    query = state.get("user_query") or ""
    draft = state.get("llm_response") or state.get("draft_response") or ""
    risk = _default_detector_risk()
    try:
        from agents.detector_agent.detector import DetectorAgent

        detector = DetectorAgent()
        detect = getattr(detector, "detect", None)
        if callable(detect):
            raw = detect(query, draft)
            if isinstance(raw, dict) and raw:
                risk = raw
    except Exception as exc:
        logger.warning("Detector inference unavailable (%s); using fail-closed risk.", exc)

    always_verify = os.environ.get("ALWAYS_VERIFY", "true").strip().lower() == "true"
    fast_path = os.environ.get("ALLOW_DETECTOR_FAST_PATH", "false").strip().lower() == "true"
    risk_level = str(risk.get("risk_level", "MEDIUM")).upper()
    next_action = str(risk.get("next_action", "verify")).lower()
    low_risk = risk_level == "LOW" and "accept" in next_action

    if not always_verify and fast_path and low_risk:
        return {
            "detector": risk,
            "detector_result": risk,
            "route": "accept",
            "verification_status": "detector_safe_fast_path",
            "trace": add_trace(state, "detector", "completed", latency_ms=0, risk_level=risk_level, fast_path=True),
        }

    return {
        "detector": risk,
        "detector_result": risk,
        "route": "verify",
        "verification_status": "verification_required",
        "trace": add_trace(state, "detector", "completed", latency_ms=0, risk_level=risk_level, draft_chars=len(draft)),
    }


async def _verifier_node(state: HalluciGuardState) -> dict[str, Any]:
    get_pipeline, SuspiciousClaim, VerifierInputV2 = _verifier_imports()
    started = start_timer()
    try:
        draft = state.get("llm_response") or state.get("draft_response") or ""
        draft_claims = extract_claims(draft) if draft.strip() else []
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
    """Canonical fail-closed arbitration over the verifier's claims.

    Mirrors the documented JudgeAgent decision rules:
      - No evidence evaluated (Rule B): VERIFY_AGAIN while retries remain,
        otherwise ABSTAIN (insufficient grounding -> human review).
      - Contradicted/hallucinated claims present: REJECT (fail closed).
      - Every claim verified: ACCEPT.
      - Unverified/conflicted claims (Rule D): VERIFY_AGAIN while retries
        remain, otherwise ABSTAIN (do NOT auto-accept unverified content).
    """
    reports = state.get("claims") or []
    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", 2))

    if not reports:
        return "VERIFY_AGAIN" if retry_count < max_retries else "ABSTAIN"

    if any(
        "contradict" in str(item.get("verdict", "")).lower()
        or "hallucinat" in str(item.get("verdict", "")).lower()
        for item in reports
    ):
        return "REJECT"

    has_verified = any(
        "verif" in str(item.get("verdict", "")).lower()
        and "unverif" not in str(item.get("verdict", "")).lower()
        for item in reports
    )
    has_unverified = any("unverif" in str(item.get("verdict", "")).lower() for item in reports)
    if has_verified and not has_unverified:
        return "ACCEPT"

    return "VERIFY_AGAIN" if retry_count < max_retries else "ABSTAIN"


async def _judge_node(state: HalluciGuardState) -> dict[str, Any]:
    decision = _judge_decision_from_verifier(state)
    route = {
        "ACCEPT": "memory",
        "REJECT": "reject",
        "ABSTAIN": "human_escalation",
        "VERIFY_AGAIN": "verifier",
        "CORRECT": "corrector",
    }.get(decision, "human_escalation")
    result = {
        "judge_decision": decision,
        "judge": {"decision": decision, "status": "completed"},
        "judge_result": {"decision": decision, "status": "completed"},
        "judge_summary": {"status": "pass" if decision == "ACCEPT" else "reject", "decision": decision},
        "route": route,
        "trace": add_trace(state, "judge", "completed", latency_ms=0, decision=decision),
    }
    if decision == "VERIFY_AGAIN":
        result["retry_count"] = int(state.get("retry_count", 0)) + 1
    return result


def _corrector_skip_result(
    state: HalluciGuardState, reason: str, requested: bool = True
) -> dict[str, Any]:
    """Fail-closed Corrector outcome: original draft preserved, nothing fabricated."""
    corrected = state.get("llm_response") or state.get("draft_response") or ""
    return {
        "correction_result": {
            "status": "skipped",
            "reason": reason,
            "corrected_text": corrected,
        },
        "correction_requested": requested,
        "correction_attempt_count": int(state.get("correction_attempt_count", 0)) + 1,
        "trace": add_trace(state, "corrector", "completed", latency_ms=0, reason="skipped"),
    }


async def _corrector_node(state: HalluciGuardState) -> dict[str, Any]:
    """Invoke the CorrectorAgent through the canonical contract when available.

    The node constructs a ``CorrectionRequest`` from state, calls the real
    ``CorrectorAgent.correct()``, and returns a canonical ``CorrectionResult``.
    Any path where the corrector cannot run preserves the original draft verbatim
    (never an invented correction) and is routed back through re-verification.
    """
    request_data = state.get("correction_request")
    if request_data is None:
        return _corrector_skip_result(
            state, "correction workflow unavailable; continuing in fail-closed mode", requested=False
        )

    request = request_data
    if not hasattr(request, "user_query"):
        try:
            from agents.corrector_agent.corrector import CorrectionRequest

            request = CorrectionRequest(**request_data)
        except Exception as exc:
            logger.warning("Invalid correction_request payload (%s); skipping correction.", exc)
            return _corrector_skip_result(
                state, "invalid correction_request payload; continuing in fail-closed mode"
            )

    provider = os.environ.get("HG_CORRECTOR_PROVIDER", "local").strip().lower()
    if provider in {"disabled", "none", "off"}:
        return _corrector_skip_result(
            state, f"correction provider disabled (HG_CORRECTOR_PROVIDER={provider})"
        )

    try:
        from agents.corrector_agent.corrector import CorrectorAgent

        agent = CorrectorAgent()
        outcome = agent.correct(request)
        if asyncio.iscoroutine(outcome):
            outcome = await outcome
        result = _dump(outcome)
        if not isinstance(result, dict) or not result.get("corrected_text"):
            raise ValueError("corrector returned no corrected_text")
        result.setdefault("status", "completed")
        return {
            "correction_result": result,
            "correction_requested": True,
            "correction_attempt_count": int(state.get("correction_attempt_count", 0)) + 1,
            "route": "reverifier",
            "trace": add_trace(
                state,
                "corrector",
                "completed",
                latency_ms=0,
                corrected_chars=len(str(result.get("corrected_text", ""))),
            ),
        }
    except Exception as exc:
        logger.warning("Corrector inference unavailable (%s); continuing fail-closed.", exc)
        return _corrector_skip_result(
            state, "correction workflow unavailable; continuing in fail-closed mode"
        )


async def _reverifier_node(state: HalluciGuardState) -> dict[str, Any]:
    get_pipeline, SuspiciousClaim, VerifierInputV2 = _verifier_imports()
    started = start_timer()
    try:
        correction = state.get("correction_result") or {}
        text = correction.get("corrected_text") if isinstance(correction, dict) else ""
        if not text:
            text = state.get("llm_response") or state.get("draft_response") or ""
        reverified_claims = extract_claims(text) if text.strip() else []
        payload = VerifierInputV2(
            query_id=state.get("request_id") or state.get("execution_id") or str(uuid.uuid4()),
            domain=state.get("domain", "general"),
            suspicious_claims=[
                SuspiciousClaim(claim_id=f"rev-{i + 1}", text=claim)
                for i, claim in enumerate(reverified_claims)
            ],
        )
        timeout = float(os.environ.get("VERIFIER_TIMEOUT_SECONDS", "120.0"))
        result = _dump(await asyncio.wait_for(get_pipeline().verify(payload), timeout=timeout))
        reports = result.get("claim_evidence") or result.get("claim_reports", [])
        remaining = sum(1 for item in reports if "contradict" in str(item.get("verdict", "")).lower())
        passed = remaining == 0
        reverb = {
            "passed": bool(passed),
            "remaining_contradictions": int(remaining),
            "verifier_result": result,
            "status": "completed",
        }
        return {
            "reverification_result": reverb,
            "reverification_summary": {
                "claims_extracted": len(reverified_claims),
                "passed": bool(passed),
                "remaining_contradictions": int(remaining),
            },
            "route": "judge",
            "reverification_attempt_count": int(state.get("reverification_attempt_count", 0)) + 1,
            "trace": add_trace(state, "reverifier", "completed", latency_ms=elapsed_ms(started), claim_count=len(reports)),
        }
    except Exception as exc:
        return _failure_update(state, "reverifier", exc, retryable=True)


async def _accept_node(state: HalluciGuardState) -> dict[str, Any]:
    return {
        "final_response": state.get("llm_response", "") or state.get("draft_response", ""),
        "terminal_status": "accepted",
        "verification_status": "accepted",
        "trace": add_trace(state, "accept", "completed", latency_ms=0),
    }


async def _reject_node(state: HalluciGuardState) -> dict[str, Any]:
    return {
        "final_response": "The draft response could not be safely verified and has been rejected.",
        "terminal_status": "rejected",
        "verification_status": "rejected",
        "trace": add_trace(state, "reject", "completed", latency_ms=0),
    }


async def _human_escalation_node(state: HalluciGuardState) -> dict[str, Any]:
    return {
        "final_response": "This response requires human review before it can be delivered.",
        "terminal_status": "human_review",
        "verification_status": "human_review_required",
        "trace": add_trace(state, "human_escalation", "completed", latency_ms=0),
    }


def _skip_memory_result(reason: str) -> dict[str, Any]:
    return {
        "memory_result": {
            "status": "skipped",
            "stored_count": 0,
            "fact_ids": [],
            "reason": reason,
        }
    }


async def _memory_node(state: HalluciGuardState) -> dict[str, Any]:
    judge_decision = str(state.get("judge_decision", "")).upper()
    rev_result = state.get("reverification_result") or {}
    verifier_result = state.get("verifier_result") or state.get("verifier") or {}

    reports: list[dict[str, Any]] = []
    if isinstance(rev_result, dict) and "passed" in rev_result:
        if not rev_result.get("passed"):
            return _skip_memory_result("reverification_failed")
        inner = rev_result.get("verifier_result") or {}
        if isinstance(inner, dict):
            reports = inner.get("claim_reports") or inner.get("claim_evidence") or []
    elif isinstance(verifier_result, dict):
        reports = verifier_result.get("claim_reports") or verifier_result.get("claim_evidence") or []

    if judge_decision and judge_decision != "ACCEPT":
        return _skip_memory_result(f"judge_decision_{judge_decision.lower()}")

    verified = [
        report
        for report in reports
        if isinstance(report, dict)
        and "verif" in str(report.get("verdict", "")).lower()
        and "unverif" not in str(report.get("verdict", "")).lower()
    ]
    if not verified:
        return _skip_memory_result("no_verified_facts")

    stored: list[dict[str, Any]] = []
    try:
        from agents.memory_agent.memory.memory_agent import MemoryAgent

        agent = MemoryAgent()
        init = getattr(agent, "initialize", None)
        if callable(init):
            outcome = init()
            if asyncio.iscoroutine(outcome):
                await outcome
        for report in verified:
            store = getattr(agent, "store_fact", None)
            if not callable(store):
                break
            outcome = store(SimpleNamespace(claim_text=report.get("claim_text", "")))
            if asyncio.iscoroutine(outcome):
                outcome = await outcome
            stored.append(outcome if isinstance(outcome, dict) else {"status": "stored"})
        close = getattr(agent, "close", None)
        if callable(close):
            outcome = close()
            if asyncio.iscoroutine(outcome):
                await outcome
    except Exception as exc:
        logger.warning("Memory persistence failed (%s); continuing.", exc)

    fact_ids = [
        str(item.get("fact_id"))
        for item in stored
        if isinstance(item, dict) and item.get("fact_id")
    ]
    return {
        "memory": {"stored": stored, "count": len(stored)},
        "memory_result": {
            "status": "stored" if stored else "skipped",
            "stored_count": len(stored),
            "fact_ids": fact_ids,
        },
        "terminal_status": state.get("terminal_status", "accepted"),
        "trace": add_trace(state, "memory", "completed", latency_ms=0, stored_count=len(stored)),
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

