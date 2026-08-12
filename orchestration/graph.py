from __future__ import annotations

import asyncio
import os
import sys
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from .state import HalluciGuardState, add_trace


MAX_VERIFICATION_RETRIES = int(os.getenv("HALLUCIGUARD_MAX_VERIFICATION_RETRIES", "2"))


def _detector_node(state: HalluciGuardState) -> dict[str, Any]:
    from agents.detector_agent.detector import DetectorAgent

    result = DetectorAgent().detect(state["user_query"], state["llm_response"])
    detector = result.model_dump() if hasattr(result, "model_dump") else result.dict()
    route = "verify" if str(detector.get("next_action", "")).lower().endswith("verify") else "accept"
    return {
        "detector": detector,
        "route": route,
        "trace": add_trace(state, "detector", "completed", result=detector),
    }


def _detector_route(state: HalluciGuardState) -> str:
    return state.get("route", "accept")


async def _verifier_node(state: HalluciGuardState) -> dict[str, Any]:
    # Import the verifier's existing production pipeline; LangGraph owns routing,
    # while the Verifier keeps its existing 9-stage retrieval/NLI implementation.
    from agents.verifier_agent.api.pipeline import VerificationPipeline
    from agents.verifier_agent.schemas.models import SuspiciousClaim, VerifierInputV2

    pipeline = VerificationPipeline()
    claim = SuspiciousClaim(claim_id="c1", text=state["llm_response"])
    payload = VerifierInputV2(
        query_id=state.get("request_id") or str(uuid.uuid4()),
        domain=state.get("domain", "general"),
        suspicious_claims=[claim],
    )
    result = await pipeline.verify(payload)
    verifier = result.model_dump() if hasattr(result, "model_dump") else result.dict()
    return {
        "verifier": verifier,
        "trace": add_trace(state, "verifier", "completed", verdicts=[c.get("verdict") for c in verifier.get("claim_evidence", [])]),
    }


def _judge_imports() -> Any:
    # The existing Judge code uses directory-local absolute imports. Add only its
    # package directory to sys.path so we reuse the production implementation
    # without rewriting the Judge internals.
    judge_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents", "judge_agent"))
    if judge_dir not in sys.path:
        sys.path.insert(0, judge_dir)
    from decision_intelligence import DecisionIntelligenceEngine
    return DecisionIntelligenceEngine


def _judge_node(state: HalluciGuardState) -> dict[str, Any]:
    DecisionIntelligenceEngine = _judge_imports()
    engine = DecisionIntelligenceEngine()
    verdict = engine.evaluate(
        user_query=state["user_query"],
        draft_response=state["llm_response"],
        detector_output=state.get("detector", {}),
        verifier_output=state.get("verifier", {}),
        domain=state.get("domain", ""),
        memory_context=state.get("memory"),
        retry_count=state.get("retry_count", 0),
    )
    judge = asdict(verdict) if is_dataclass(verdict) else verdict
    action = judge.get("workflow_action", {}).get("type") or judge.get("workflow_action", {}).get("action_type")
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


def _corrector_imports() -> Any:
    corrector_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agents", "corrector_agent"))
    if corrector_dir not in sys.path:
        sys.path.insert(0, corrector_dir)
    from app.models import (
        AtomicClaim,
        ClaimStatus,
        EvidencePassage,
        JudgeVerificationPayload,
        SourceMetadata,
    )
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
        claim_text = str(report.get("claim_text", ""))
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
                text=claim_text,
                status=status,
                confidenceScore=float(report.get("confidence_score", report.get("trust_score", 0.0))),
                evidenceIds=evidence_ids,
            )
        )

    judge = state.get("judge", {})
    reasoning = judge.get("reasoning_chain", [])
    instructions = "\n".join(reasoning) if reasoning else "Correct only claims contradicted by the verified evidence. Preserve verified claims. Do not invent facts."
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
    payload = _build_corrector_payload(state)
    result = await CorrectorOrchestrator().executeCorrectionPipeline(payload)
    corrector = result.model_dump() if hasattr(result, "model_dump") else result.dict()
    final_response = corrector.get("finalResponse") or state["llm_response"]
    return {
        "corrector": corrector,
        "final_response": final_response,
        "trace": add_trace(state, "corrector", "completed", approved=corrector.get("isFullyApproved"), attempts=corrector.get("attemptsCount")),
    }


async def _memory_node(state: HalluciGuardState) -> dict[str, Any]:
    from agents.memory_agent.memory.memory_agent import MemoryAgent
    from agents.memory_agent.schemas.models import StoreFactRequest

    memory_agent = MemoryAgent()
    await memory_agent.initialize()
    try:
        verifier = state.get("verifier", {})
        reports = verifier.get("claim_evidence", [])
        stored = []
        for report in reports:
            verdict = str(report.get("verdict", "insufficient_evidence")).lower()
            evidence = report.get("evidence", [])
            sources = [str(e.get("source", "")) for e in evidence if e.get("source")]
            request = StoreFactRequest(
                claim_text=str(report.get("claim_text", "")),
                domain=state.get("domain", "general"),
                verdict=verdict,
                evidence=[
                    {"source_id": e.get("source", ""), "title": e.get("title", ""), "url": e.get("url")}
                    for e in evidence
                ],
                source_ids=sources,
                confidence=float(report.get("confidence_score", report.get("trust_score", 0.0))),
                metadata={"request_id": state.get("request_id", ""), "detector": state.get("detector", {})},
            )
            stored.append((await memory_agent.store_fact(request)).model_dump())

        memory = {"stored": stored, "count": len(stored)}
    finally:
        await memory_agent.close()

    final_response = state.get("final_response") or state["llm_response"]
    return {
        "memory": memory,
        "final_response": final_response,
        "trace": add_trace(state, "memory", "completed", stored_count=memory["count"]),
    }


def _accept_node(state: HalluciGuardState) -> dict[str, Any]:
    return {
        "final_response": state["llm_response"],
        "trace": add_trace(state, "accept", "completed", reason="Detector risk was not HIGH"),
    }


def _finish_node(state: HalluciGuardState) -> dict[str, Any]:
    final_response = state.get("final_response") or state["llm_response"]
    return {
        "final_response": final_response,
        "trace": add_trace(state, "finish", "completed", decision=state.get("judge", {}).get("decision")),
    }


def _retry_node(state: HalluciGuardState) -> dict[str, Any]:
    count = state.get("retry_count", 0) + 1
    return {
        "retry_count": count,
        "trace": add_trace(state, "verify_retry", "scheduled", retry_count=count),
    }


def build_verification_graph():
    """Build the production HalluciGuard graph.

    Graph topology:
      START -> Detector
      Detector -- LOW/MEDIUM -> Accept -> Memory -> END
      Detector -- HIGH -> Verifier -> Judge
      Judge -- VERIFY_AGAIN -> Retry -> Verifier
      Judge -- CORRECT -> Corrector -> Memory -> END
      Judge -- ACCEPT/REJECT/ABSTAIN/HUMAN -> Finish -> Memory -> END

    The five existing agents remain independent specialists. LangGraph is the
    stateful supervisor and routing layer; it does not replace their logic.
    """
    graph = StateGraph(HalluciGuardState)
    graph.add_node("detector", _detector_node)
    graph.add_node("accept", _accept_node)
    graph.add_node("verifier", _verifier_node)
    graph.add_node("judge", _judge_node)
    graph.add_node("retry", _retry_node)
    graph.add_node("corrector", _corrector_node)
    graph.add_node("finish", _finish_node)
    graph.add_node("memory", _memory_node)

    graph.add_edge(START, "detector")
    graph.add_conditional_edges("detector", _detector_route, {"verify": "verifier", "accept": "accept"})
    graph.add_edge("accept", "memory")
    graph.add_edge("verifier", "judge")
    graph.add_conditional_edges(
        "judge",
        _judge_route,
        {"verify_again": "retry", "correct": "corrector", "finish": "finish"},
    )
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


async def run_verification(
    user_query: str,
    llm_response: str,
    domain: str = "general",
    request_id: str | None = None,
) -> HalluciGuardState:
    graph = get_verification_graph()
    initial: HalluciGuardState = {
        "request_id": request_id or str(uuid.uuid4()),
        "user_query": user_query,
        "llm_response": llm_response,
        "domain": domain,
        "retry_count": 0,
        "trace": [],
    }
    return await graph.ainvoke(initial)
