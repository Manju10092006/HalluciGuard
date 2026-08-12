from __future__ import annotations

from typing import Any, TypedDict


class HalluciGuardState(TypedDict, total=False):
    request_id: str
    user_query: str
    llm_response: str
    domain: str

    detector: dict[str, Any]
    route: str
    claims: list[dict[str, Any]]

    verifier: dict[str, Any]
    judge_pairs: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    retrieval_results: list[dict[str, Any]]
    reranking_results: list[dict[str, Any]]
    nli_results: list[dict[str, Any]]

    judge: dict[str, Any]
    corrector: dict[str, Any]
    memory: dict[str, Any]
    knowledge_graph_context: dict[str, Any]

    final_response: str
    trace: list[dict[str, Any]]
    retry_count: int
    error: str


def add_trace(state: HalluciGuardState, node: str, status: str, **details: Any) -> list[dict[str, Any]]:
    trace = list(state.get("trace", []))
    trace.append({"node": node, "status": status, **details})
    return trace
