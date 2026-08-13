from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

AgentStatus = Literal["started", "completed", "failed", "skipped", "scheduled"]
TerminalStatus = Literal[
    "accepted", "partial_success", "failed", "human_review", "fallback"
]


class AgentError(TypedDict, total=False):
    node: str
    type: str
    message: str
    timestamp: str
    retryable: bool


class ExecutionEvent(TypedDict, total=False):
    execution_id: str
    node: str
    status: AgentStatus
    timestamp: str
    latency_ms: int
    retry_count: int
    details: dict[str, Any]


class HalluciGuardState(TypedDict, total=False):
    # Request / Base LLM draft
    execution_id: str
    request_id: str
    user_query: str
    llm_response: str
    draft_response: str
    generation_mode: str
    conversation_history: list[dict[str, str]]
    generation: dict[str, Any]
    domain: str
    created_at: str
    updated_at: str

    # Shared inter-agent contract
    detector: dict[str, Any]
    route: str
    claims: list[dict[str, Any]]
    hallucination_probability: float
    confidence: float
    verification_status: str
    verifier: dict[str, Any]
    judge_pairs: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    retrieved_evidence: list[dict[str, Any]]
    ranked_evidence: list[dict[str, Any]]
    nli_results: list[dict[str, Any]]
    judge: dict[str, Any]
    judge_decision: str
    corrector: dict[str, Any]
    memory: dict[str, Any]
    memory_context: dict[str, Any]
    knowledge_graph_context: dict[str, Any]

    # Control plane
    final_response: str
    terminal_status: TerminalStatus
    retry_count: int
    max_retries: int
    current_node: str
    errors: list[AgentError]
    error: str

    # Observability / audit
    trace: list[ExecutionEvent]
    audit: dict[str, Any]
    active_agents: list[str]
    disabled_agents: dict[str, dict[str, str | bool]]
    inter_agent_bus: list[dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_timer() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def add_trace(
    state: HalluciGuardState,
    node: str,
    status: AgentStatus,
    *,
    latency_ms: int | None = None,
    **details: Any,
) -> list[ExecutionEvent]:
    trace = list(state.get("trace", []))
    event: ExecutionEvent = {
        "execution_id": state.get("execution_id", state.get("request_id", "")),
        "node": node,
        "status": status,
        "timestamp": utc_now(),
        "retry_count": int(state.get("retry_count", 0)),
        "details": details,
    }
    if latency_ms is not None:
        event["latency_ms"] = latency_ms
    trace.append(event)
    return trace


def add_error(
    state: HalluciGuardState, node: str, exc: BaseException, *, retryable: bool = False
) -> list[AgentError]:
    errors = list(state.get("errors", []))
    errors.append(
        {
            "node": node,
            "type": type(exc).__name__,
            "message": str(exc),
            "timestamp": utc_now(),
            "retryable": retryable,
        }
    )
    return errors
