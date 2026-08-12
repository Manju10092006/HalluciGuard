from __future__ import annotations

import os
import sys
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from .state import HalluciGuardState, add_trace

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


def _detector_node(state: HalluciGuardState) -> dict[str, Any]:
    from agents.detector_agent.detector import DetectorAgent

    result = DetectorAgent().detect(state["user_query"], state["llm_response"])
    detector = _dump(result)
    next_action = str(detector.get("next_action", ""))
    route = "verify" if next_action.lower().endswith("verify") else "accept"
    return {
        "detector": detector,
        "route": route,
        "trace": add_trace(state, "detector", "completed", route=route, hallucination_probability=detector.get("hallucination_probability")),
    }


def _detector_route(state: HalluciGuardState) -> str:
    return state.get("route", "accept")


def _verifier_imports():
    """Load the existing verifier package using its current internal import contract."""
    verifier_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents", "verifier_agent"))
    if verifier_dir not in sys.path:
        sys.path.insert(0, verifier_dir)
    from api.pipeline import VerificationPipeline
    from schemas.models import SuspiciousClaim, VerifierInputV2
    return VerificationPipeline, SuspiciousClaim, VerifierInputV2


async def _verifier_node(state: HalluciGuardState) -> dict[str, Any]:
    VerificationPipeline, SuspiciousClaim, VerifierInputV2 = _verifier_imports()

    pipeline = VerificationPipeline()
    payload = VerifierInputV2(
        query_id=state.get("request_id") or str(uuid.uuid4()),
        domain=state.get("domain", "general"),
        suspicious_claims=[SuspiciousClaim(claim_id="c1", text=state["llm_response"])],
    )
    result = await pipeline.verify(payload)
    verifier = _dump(result)

    # The Judge consumes claim/evidence pairs. Build that contract from the
    # real verifier ClaimReport/EvidenceItem output rather than inventing data.
    judge_pairs: list[dict[str, Any]] = []
    for report in verifier.get("claim_evidence", []):
        evidence_items = report.get("evidence", [])
        for evidence in evidence_items:
            judge_pairs.append({
                "claim": report.get("claim_text", ""),
                "evidence": evidence.get("snippet", ""),
                "source": evidence.get("source", ""),
                "url": evidence.get("url", ""),
                "entailment_label": evidence.get("entailment_label", "neutral"),
                "entailment_score": evidence.get("entailment_score", 0.0),
                "credibility_score": evidence.get("credibility_score", 0.0),
            })
        if not evidence_items:
            judge_pairs.append({"claim": report.get("claim_text", ""), "evidence": "", "source": "None (Unverified)"})

    return {
        "verifier": verifier,
        "judge_pairs": judge_pairs,
        "trace": add_trace(
            state,
            "verifier",
            "completed",
            retrieved_sources=verifier.get("retrieved_sources", 0),
            verified_sources=verifier.get("verified_sources", 0),
            claim_count=len(verifier.get("claim_evidence", [])),
        ),
    }


def _judge_imports():
    judge_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents", "judge_agent"))
    if judge_dir not in sys.path:
        sys.path.insert(0, judge_dir)
    from decision_intelligence import DecisionIntelligenceEngine
    return DecisionIntelligenceEngine


def _judge_node(state: HalluciGuardState) -> dict[str, Any]:
    DecisionIntelligenceEngine = _judge_imports()
    engine = DecisionIntelligenceEngine()
    verifier_for_judge = dict(state.get("verifier", {}))
    verifier_for_judge["claim_evidence_pairs"] = state.get("judge_pairs", [])

    verdict = engine.evaluate(
        user_query=state["user_query"],
        draft_response=state["llm_response"],
        detector_output=state.get("detector", {}),
        verifier_output=verifier_for_judge,
        domain=state.get("domain", ""),
        memory_context=state.get("memory"),
        retry_count=state.get("retry_count", 0),
    )
    judge = _dump(verdict)
    action = judge.get("workflow_action", {}).get("type", "")
    return {
        "judge": judge,
        "trace": add_trace(state, "judge", "completed", decision=judge.get("decision"), action=action),
    }


def _judge_route(state: HalluciGuardState) -> str:
    decision = str(state.get("judge", {}).get("decision", "")).upper()
    if decision.endswith("VERIFY_AGAIN"):
        return "verify_again" if state.get("retry_count", 0) < MAX_VERIFICATION_RETRIES else "finish"
    if decision.endswith("CORRECT"):
        return "correct"
    return "finish"


def _corrector_imports():
    corrector_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents", "corrector_agent"))
    if corrector_dir not in sys.path:
        sys.path.insert(0, corrector_dir)
    from app.models import AtomicClaim, ClaimStatus, EvidencePassage, JudgeVerificationPayload, SourceMetadata
    from app.orchestrator import CorrectorOrchestrator
    return CorrectorOrchestrator, AtomicClaim, ClaimStatus, EvidencePassage, JudgeVerificationPayload, SourceMetadata


def _build_corrector_payload(state: HalluciGuardState) -> Any:
    (
        _CorrectorOrchestrator,
        AtomicClaim,
        ClaimStatus,
        EvidencePassage,
        JudgeVerificationPayload,
        SourceMetadata,
    ) = _corrector_imports()

    verifier = state.get("verifier", {})
    reports = verifier.get("claim_evidence", [])
    claims = []
    supporting = []
    contradictions = []
    source_metadata = []

    for report_index, report in enumerate(reports):
        verdict = str(report.get("verdict", "insufficient_evidence")).lower()
        status = (
            ClaimStatus.VERIFIED if verdict == "verified"
            else ClaimStatus.CONTRADICTED if verdict == "likely_hallucinated"
            else ClaimStatus.INSUFFICIENT_EVIDENCE
        )
        claim_id = str(report.get("claim_id", f"c{report_index + 1}"))
        evidence_ids = []
        for evidence_index, item in enumerate(report.get("evidence", [])):
            evidence_id = f"{claim_id}-e{evidence_index + 1}"
            evidence_ids.append(evidence_id)
            passage = EvidencePassage(
                id=evidence_id,
                sourceTitle=str(item.get("title", item.get("source", "Evidence"))),
                passageText=str(item.get("snippet", "")),
                url=item.get("url"),
                relevanceScore=float(item.get("credibility_score", item.get("entailment_score", 0.0))),
            )
            label = str(item.get("entailment_label", "neutral")).lower()
            if label == "entailment":
                supporting.append(passage)
            elif label == "contradiction":
                contradictions.append(passage)
            source_id = str(item.get("source", ""))
            if source_id:
                source_metadata.append(SourceMetadata(id=source_id, title=source_id, url=item.get("url")))

        claims.append(
            AtomicClaim(
                id=claim_id,
                text=str(report.get("claim_text", "")),
                status=status,
                confidenceScore=float(report.get("confidence_score", report.get("trust_score", 0.0))),
                evidenceIds=evidence_ids,
            )
        )

    reasoning = state.get("judge", {}).get("reasoning_chain", [])
    instructions = "\n".join(reasoning) if reasoning else "Correct only claims contradicted by verified evidence. Preserve verified claims. Do not invent facts."
    return JudgeVerificationPayload(
        query=state["user_query"],
        originalResponse=state["llm_response"],
        claims=claims,
        supportingEvidence=supporting,
        contradictionEvidence=contradictions,
        trustScore=float(verifier.get("overall_evidence_confidence", 0.0)),
        sourceMetadata=source_metadata,
        correctionInstructions=instructions,
    )


async def _corrector_node(state: HalluciGuardState) -> dict[str, Any]:
    CorrectorOrchestrator, *_ = _corrector_imports()
    result = await CorrectorOrchestrator().executeCorrectionPipeline(_build_corrector_payload(state))
    corrector = _dump(result)
    return {
        "corrector": corrector,
        "final_response": corrector.get("finalResponse") or state["llm_response"],
        "trace": add_trace(state, "corrector", "completed", approved=corrector.get("isFullyApproved"), attempts=corrector.get("attemptsCount")),
    }


async def _memory_node(state: HalluciGuardState) -> dict[str, Any]:
    from agents.memory_agent.memory.memory_agent import MemoryAgent
    from agents.memory_agent.schemas.models import StoreFactRequest

    memory_agent = MemoryAgent()
    await memory_agent.initialize()
    try:
        stored = []
        for report in state.get("verifier", {}).get("claim_evidence", []):
            evidence = report.get("evidence", [])
            sources = [str(e.get("source", "")) for e in evidence if e.get("source")]
            request = StoreFactRequest(
                claim_text=str(report.get("claim_text", "")),
                domain=state.get("domain", "general"),
                verdict=str(report.get("verdict", "insufficient_evidence")),
                evidence=[
                    {"source_id": e.get("source", ""), "title": e.get("title", ""), "url": e.get("url"), "snippet": e.get("snippet", "")}
                    for e in evidence
                ],
                source_ids=sources,
                confidence=float(report.get("confidence_score", report.get("trust_score", 0.0))),
                metadata={"request_id": state.get("request_id", ""), "detector": state.get("detector", {}), "judge": state.get("judge", {})},
            )
            stored.append(_dump(await memory_agent.store_fact(request)))
    finally:
        await memory_agent.close()

    memory = {"stored": stored, "count": len(stored), "knowledge_graph": True, "vector_memory": True}
    return {
        "memory": memory,
        "final_response": state.get("final_response") or state["llm_response"],
        "trace": add_trace(state, "memory", "completed", stored_count=len(stored)),
    }


def _accept_node(state: HalluciGuardState) -> dict[str, Any]:
    return {"final_response": state["llm_response"], "trace": add_trace(state, "accept", "completed", reason="Detector route accepted")}


def _finish_node(state: HalluciGuardState) -> dict[str, Any]:
    return {"final_response": state.get("final_response") or state["llm_response"], "trace": add_trace(state, "finish", "completed", decision=state.get("judge", {}).get("decision"))}


def _retry_node(state: HalluciGuardState) -> dict[str, Any]:
    count = state.get("retry_count", 0) + 1
    return {"retry_count": count, "trace": add_trace(state, "verify_retry", "scheduled", retry_count=count)}


def build_verification_graph():
    graph = StateGraph(HalluciGuardState)
    for name, fn in {
        "detector": _detector_node,
        "accept": _accept_node,
        "verifier": _verifier_node,
        "judge": _judge_node,
        "retry": _retry_node,
        "corrector": _corrector_node,
        "finish": _finish_node,
        "memory": _memory_node,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "detector")
    graph.add_conditional_edges("detector", _detector_route, {"verify": "verifier", "accept": "accept"})
    graph.add_edge("accept", "memory")
    graph.add_edge("verifier", "judge")
    graph.add_conditional_edges("judge", _judge_route, {"verify_again": "retry", "correct": "corrector", "finish": "finish"})
    graph.add_edge("retry", "verifier")
    graph.add_edge("corrector", "memory")
    graph.add_edge("finish", "memory")
    graph.add_edge("memory", END)
    return graph.compile()


_GRAPH = None


def get_verification_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_verification_graph()
    return _GRAPH


async def run_verification(user_query: str, llm_response: str, domain: str = "general", request_id: str | None = None) -> HalluciGuardState:
    return await get_verification_graph().ainvoke({
        "request_id": request_id or str(uuid.uuid4()),
        "user_query": user_query,
        "llm_response": llm_response,
        "domain": domain,
        "retry_count": 0,
        "trace": [],
    })
