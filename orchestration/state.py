from __future__ import annotations

from typing import Any, TypedDict


class HalluciGuardState(TypedDict, total=False):
    # Immutable request/draft context
    request_id: str
    user_query: str
    llm_response: str
    domain: str

    # Detector
    detector: dict[str, Any]
    route: str

    # Verifier
    verifier: dict[str, Any]

    # Judge
    judge: dict[str, Any]

    # Corrector
    corrector: dict[str, Any]

    # Memory / knowledge graph
    memory: dict[str, Any]

    # Final output and observability
    final_response: str
    trace: list[dict[str, Any]]
    retry_count: int
    error: str


def add_trace(state: HalluciGuardState, node: str, status: str, **details: Any) -> list[dict[str, Any]]:
    trace = list(state.get("trace", []))
    trace.append({"node": node, "status": status, **details})
    return trace
