from __future__ import annotations

import os
import sys
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from .state import HalluciGuardState, add_error, add_trace, elapsed_ms, start_timer, utc_now

MAX_VERIFICATION_RETRIES = int(os.getenv("HALLUCIGUARD_MAX_VERIFICATION_RETRIES", "2"))


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


def _failure_update(state: HalluciGuardState, node: str, exc: BaseException, *, retryable: bool = False) -> dict[str, Any]:
    return {
        "errors": add_error(state, node, exc, retryable=retryable),
        "error": f"{node} failed: {type(exc).__name__}: {exc}",
        "route": "error",
        "terminal_status": "human_review" if retryable else "fallback",
        "verification_status": "agent_failed",
        "updated_at": utc_now(),
        "trace": add_trace(state, node, "failed", error_type=type(exc).__name__, retryable=retryable),
    }


def _detector_node(state: HalluciGuardState) -> dict[str, Any]:
    from agents.detector_agent.detector import DetectorAgent

    node_start = start_timer()
    try:
        detector = _dump(DetectorAgent().detect(state["user_query"], state["llm_response"]))
        next_action = str(detector.get("next_action", ""))
        route = "verify" if next_action.lower().endswith("verify") else "accept"
        return {
            "detector": detector,
            "route": route,
            "hallucination_probability": float(detector.get("hallucination_probability", 0.0)),
            "confidence": float(detector.get("confidence_score", 0.0)),
            "verification_status": "detector_safe_fast_path" if route == "accept" else "verification_required",
            "updated_at": utc_now(),
            "trace": add_trace(state, "detector", "completed", latency_ms=elapsed_ms(node_start), route=route),
        }
    except Exception as exc:
        return _failure_update(state, "detector", exc)


def _detector_route(state: HalluciGuardState) -> str:
    return state.get("route", "accept") if state.get("route") in {"verify", "accept", "error"} else "accept"


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
        judge_pairs: list[dict[str, Any]] = []
        evidence_all: list[dict[str, Any]] = []
        nli_results: list[dict[str, Any]] = []
        claims: list[dict[str, Any]] = []
        for report in verifier.get("claim_evidence", []):
            claims.append({"claim_id": report.get("claim_id"), "text": report.get("claim_text"), "verdict": report.get("verdict")})
            evidence_items = report.get("evidence", [])
            for evidence in evidence_items:
                evidence_all.append(evidence)
                nli_results.append({"claim": report.get("claim_text", ""), "label": evidence.get("entailment_label"), "score": evidence.get("entailment_score")})
                judge_pairs.append({"claim": report.get("claim_text", ""), "evidence": evidence.get("snippet", ""), "source": evidence.get("source", ""), "url": evidence.get("url", ""), "entailment_label": evidence.get("entailment_label", "neutral"), "entailment_score": evidence.get("entailment_score", 0.0), "credibility_score": evidence.get("credibility_score", 0.0)})
            if not evidence_items:
                judge_pairs.append({"claim": report.get("claim_text", ""), "evidence": "", "source": "None (Unverified)"})
        return {"verifier": verifier, "claims": claims, "judge_pairs": judge_pairs, "evidence": evidence_all, "retrieved_evidence": evidence_all, "ranked_evidence": evidence_all, "nli_results": nli_results, "verification_status": "verified", "updated_at": utc_now(), "trace": add_trace(state, "verifier", "completed", latency_ms=elapsed_ms(node_start), claim_count=len(claims), evidence_count=len(evidence_all))}
    except Exception as exc:
        return _failure_update(state, "verifier", exc, retryable=True)


def _judge_imports():
    judge_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents", "judge_agent"))
    if judge_dir not in sys.path:
        sys.path.insert(0, judge_dir)
    from decision_intelligence import DecisionIntelligenceEngine
    return DecisionIntelligenceEngine


def _judge_node(state: HalluciGuardState) -> dict[str, Any]:
    DecisionIntelligenceEngine = _judge_imports()
    node_start = start_timer()
    try:
        verifier_for_judge = dict(state.get("verifier", {}))
        verifier_for_judge["claim_evidence_pairs"] = state.get("judge_pairs", [])
        judge = _dump(DecisionIntelligenceEngine().evaluate(state["user_query"], state["llm_response"], state.get("detector", {}), verifier_for_judge, state.get("domain", ""), state.get("memory_context"), state.get("retry_count", 0)))
        decision = str(judge.get("decision", "")).upper()
        return {"judge": judge, "judge_decision": decision, "updated_at": utc_now(), "trace": add_trace(state, "judge", "completed", latency_ms=elapsed_ms(node_start), decision=decision, action=judge.get("workflow_action", {}).get("type"))}
    except Exception as exc:
        return _failure_update(state, "judge", exc)


def _judge_route(state: HalluciGuardState) -> str:
    decision = str(state.get("judge_decision") or state.get("judge", {}).get("decision", "")).upper()
    if decision.endswith("VERIFY_AGAIN"):
        return "verify_again" if state.get("retry_count", 0) < state.get("max_retries", MAX_VERIFICATION_RETRIES) else "retry_exhausted"
    if decision.endswith("CORRECT"):
        return "correct"
    if decision.endswith("REJECT"):
        return "reject"
    if decision.endswith("ESCALATE_HUMAN") or decision.endswith("HUMAN_ESCALATE") or decision.endswith("ABSTAIN"):
        return "human_escalate"
    if decision.endswith("ACCEPT"):
        return "accept"
    return "human_escalate"


def _corrector_imports():
    corrector_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents", "corrector_agent"))
    if corrector_dir not in sys.path:
        sys.path.insert(0, corrector_dir)
    from app.models import AtomicClaim, ClaimStatus, EvidencePassage, JudgeVerificationPayload, SourceMetadata
    from app.orchestrator import CorrectorOrchestrator
    return CorrectorOrchestrator, AtomicClaim, ClaimStatus, EvidencePassage, JudgeVerificationPayload, SourceMetadata


def _build_corrector_payload(state: HalluciGuardState) -> Any:
    _, AtomicClaim, ClaimStatus, EvidencePassage, JudgeVerificationPayload, SourceMetadata = _corrector_imports()
    claims=[]; supporting=[]; contradictions=[]; source_metadata=[]
    for i, report in enumerate(state.get("verifier", {}).get("claim_evidence", [])):
        verdict = str(report.get("verdict", "insufficient_evidence")).lower()
        status = ClaimStatus.VERIFIED if verdict == "verified" else ClaimStatus.CONTRADICTED if verdict == "likely_hallucinated" else ClaimStatus.INSUFFICIENT_EVIDENCE
        cid = str(report.get("claim_id", f"c{i+1}")); eids=[]
        for j, item in enumerate(report.get("evidence", [])):
            eid=f"{cid}-e{j+1}"; eids.append(eid)
            passage=EvidencePassage(id=eid, sourceTitle=str(item.get("title", item.get("source", "Evidence"))), passageText=str(item.get("snippet", "")), url=item.get("url"), relevanceScore=float(item.get("credibility_score", item.get("entailment_score", 0.0))))
            (supporting if str(item.get("entailment_label", "neutral")).lower()=="entailment" else contradictions).append(passage)
            if item.get("source"):
                source_metadata.append(SourceMetadata(id=str(item.get("source")), title=str(item.get("source")), url=item.get("url")))
        claims.append(AtomicClaim(id=cid, text=str(report.get("claim_text", "")), status=status, confidenceScore=float(report.get("confidence_score", report.get("trust_score", 0.0))), evidenceIds=eids))
    instructions = "\n".join(state.get("judge", {}).get("reasoning_chain", [])) or "Correct only claims contradicted by verified evidence. Preserve verified claims. Do not invent facts."
    return JudgeVerificationPayload(query=state["user_query"], originalResponse=state["llm_response"], claims=claims, supportingEvidence=supporting, contradictionEvidence=contradictions, trustScore=float(state.get("verifier", {}).get("overall_evidence_confidence", 0.0)), sourceMetadata=source_metadata, correctionInstructions=instructions)


async def _corrector_node(state: HalluciGuardState) -> dict[str, Any]:
    CorrectorOrchestrator, *_ = _corrector_imports()
    node_start = start_timer()
    try:
        corrector = _dump(await CorrectorOrchestrator().executeCorrectionPipeline(_build_corrector_payload(state)))
        return {"corrector": corrector, "final_response": corrector.get("finalResponse") or state["llm_response"], "terminal_status": "corrected", "updated_at": utc_now(), "trace": add_trace(state, "corrector", "completed", latency_ms=elapsed_ms(node_start), approved=corrector.get("isFullyApproved"))}
    except Exception as exc:
        return _failure_update(state, "corrector", exc)


async def _memory_node(state: HalluciGuardState) -> dict[str, Any]:
    from agents.memory_agent.memory.memory_agent import MemoryAgent
    from agents.memory_agent.schemas.models import StoreFactRequest

    node_start = start_timer()
    if not state.get("verifier", {}).get("claim_evidence"):
        memory = {"stored": [], "count": 0, "knowledge_graph": False, "vector_memory": False, "skipped_reason": "no_verified_claims_to_persist"}
        return {"memory": memory, "final_response": state.get("final_response") or state.get("llm_response", ""), "updated_at": utc_now(), "trace": add_trace(state, "memory", "skipped", latency_ms=elapsed_ms(node_start), reason=memory["skipped_reason"])}
    try:
        memory_agent = MemoryAgent(); await memory_agent.initialize(); stored=[]
        try:
            for report in state.get("verifier", {}).get("claim_evidence", []):
                evidence = report.get("evidence", []); sources=[str(e.get("source", "")) for e in evidence if e.get("source")]
                req=StoreFactRequest(claim_text=str(report.get("claim_text", "")), domain=state.get("domain", "general"), verdict=str(report.get("verdict", "insufficient_evidence")), evidence=[{"source_id": e.get("source", ""), "title": e.get("title", ""), "url": e.get("url"), "snippet": e.get("snippet", "")} for e in evidence], source_ids=sources, confidence=float(report.get("confidence_score", report.get("trust_score", 0.0))), metadata={"execution_id": state.get("execution_id", ""), "request_id": state.get("request_id", ""), "judge": state.get("judge", {})})
                stored.append(_dump(await memory_agent.store_fact(req)))
        finally:
            await memory_agent.close()
        memory={"stored": stored, "count": len(stored), "knowledge_graph": True, "vector_memory": True}
        return {"memory": memory, "final_response": state.get("final_response") or state["llm_response"], "updated_at": utc_now(), "trace": add_trace(state, "memory", "completed", latency_ms=elapsed_ms(node_start), stored_count=len(stored))}
    except Exception as exc:
        update = _failure_update(state, "memory", exc)
        update["final_response"] = state.get("final_response") or state.get("llm_response", "")
        return update


def _accept_node(state: HalluciGuardState) -> dict[str, Any]:
    return {"final_response": state["llm_response"], "terminal_status": "accepted", "updated_at": utc_now(), "trace": add_trace(state, "accept", "completed", reason="accepted by detector/judge")}


def _reject_node(state: HalluciGuardState) -> dict[str, Any]:
    msg = "The draft response could not be safely verified and has been rejected."
    return {"final_response": msg, "terminal_status": "rejected", "verification_status": "rejected", "updated_at": utc_now(), "trace": add_trace(state, "reject", "completed", decision=state.get("judge_decision"))}


def _human_escalation_node(state: HalluciGuardState) -> dict[str, Any]:
    msg = "This response requires human review before it can be delivered."
    return {"final_response": msg, "terminal_status": "human_review", "verification_status": "human_review_required", "updated_at": utc_now(), "trace": add_trace(state, "human_escalation", "completed", decision=state.get("judge_decision"), errors=state.get("errors", []))}


def _retry_exhausted_node(state: HalluciGuardState) -> dict[str, Any]:
    return {"terminal_status": "human_review", "verification_status": "retry_limit_exhausted", "updated_at": utc_now(), "trace": add_trace(state, "retry_exhausted", "completed", retry_count=state.get("retry_count", 0))}


def _retry_node(state: HalluciGuardState) -> dict[str, Any]:
    count = state.get("retry_count", 0) + 1
    return {"retry_count": count, "updated_at": utc_now(), "trace": add_trace(state, "verify_retry", "scheduled", retry_count=count)}


def build_verification_graph(node_overrides: dict[str, Callable[..., Any]] | None = None):
    nodes = {"detector": _detector_node, "accept": _accept_node, "verifier": _verifier_node, "judge": _judge_node, "retry": _retry_node, "corrector": _corrector_node, "reject": _reject_node, "human_escalation": _human_escalation_node, "retry_exhausted": _retry_exhausted_node, "memory": _memory_node}
    if node_overrides:
        nodes.update(node_overrides)
    graph = StateGraph(HalluciGuardState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "detector")
    graph.add_conditional_edges("detector", _detector_route, {"verify": "verifier", "accept": "accept", "error": "human_escalation"})
    graph.add_edge("verifier", "judge")
    graph.add_conditional_edges("judge", _judge_route, {"verify_again": "retry", "correct": "corrector", "reject": "reject", "human_escalate": "human_escalation", "accept": "accept", "retry_exhausted": "retry_exhausted"})
    graph.add_edge("retry", "verifier")
    graph.add_edge("corrector", "memory")
    graph.add_edge("accept", "memory")
    graph.add_edge("reject", "memory")
    graph.add_edge("human_escalation", "memory")
    graph.add_edge("retry_exhausted", "human_escalation")
    graph.add_edge("memory", END)
    return graph.compile()


_GRAPH = None


def get_verification_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_verification_graph()
    return _GRAPH


async def run_verification(user_query: str, llm_response: str, domain: str = "general", request_id: str | None = None) -> HalluciGuardState:
    execution_id = str(uuid.uuid4())
    now = utc_now()
    return await get_verification_graph().ainvoke({"execution_id": execution_id, "request_id": request_id or execution_id, "user_query": user_query, "llm_response": llm_response, "domain": domain, "retry_count": 0, "max_retries": MAX_VERIFICATION_RETRIES, "created_at": now, "updated_at": now, "trace": [], "errors": [], "audit": {"graph": "halluciguard_langgraph_supervisor", "max_retries": MAX_VERIFICATION_RETRIES}})
