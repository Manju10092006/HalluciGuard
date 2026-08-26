from __future__ import annotations

import asyncio
import os
import sys
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from services.base_llm_service import BaseLLMService
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
        return {k: _dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(v) for v in value]
    return value


def _failure_update(
    state: HalluciGuardState, node: str, exc: BaseException, *, retryable: bool = False
) -> dict[str, Any]:
    bus = add_bus_message(
        state,
        source_agent=node,
        target_agent="supervisor",
        message_type="ERROR_EVENT",
        payload={"error_type": type(exc).__name__, "message": str(exc)},
        status="failed",
    )
    return {
        "errors": add_error(state, node, exc, retryable=retryable),
        "error": f"{node} failed: {type(exc).__name__}: {exc}",
        "route": "error",
        "terminal_status": "human_review" if retryable else "fallback",
        "verification_status": "agent_failed",
        "inter_agent_bus": bus,
        "updated_at": utc_now(),
        "trace": add_trace(
            state, node, "failed", error_type=type(exc).__name__, retryable=retryable
        ),
    }


async def _generate_node(state: HalluciGuardState) -> dict[str, Any]:
    """Base LLM node: Generate initial draft using OpenRouter if not pre-supplied."""
    node_start = start_timer()
    user_query = state.get("user_query", "")
    existing_response = state.get("llm_response", "")

    # If response was already provided by caller, use it
    if existing_response and existing_response.strip():
        bus = add_bus_message(
            state,
            source_agent="caller",
            target_agent="supervisor",
            message_type="DRAFT_RESPONSE",
            payload={"draft": existing_response, "source": "pre_supplied"},
        )
        return {
            "draft_response": existing_response,
            "llm_response": existing_response,
            "base_llm": {
                "provider": "pre_supplied",
                "model": "caller_input",
                "latency_ms": 0,
            },
            "inter_agent_bus": bus,
            "updated_at": utc_now(),
            "trace": add_trace(
                state,
                "base_llm",
                "completed",
                latency_ms=0,
                provider="pre_supplied",
            ),
        }

    try:
        service = BaseLLMService()
        result = await service.generate(
            user_query=user_query,
            conversation_history=state.get("conversation_history", []),
            generation_mode=state.get("generation_mode", "normal"),
        )
        gen_result = _dump(result)
        if gen_result.get("status") not in {"success", "completed"}:
            exc = RuntimeError(str(gen_result.get("error") or "Base LLM generation failed"))
            update = _failure_update(state, "base_llm", exc)
            update["base_llm"] = gen_result
            update["final_response"] = "Base LLM generation failed. Please try again."
            return update

        draft = gen_result.get("draft_response", "")
        latency = gen_result.get("latency_ms", elapsed_ms(node_start))

        bus = add_bus_message(
            state,
            source_agent="base_llm",
            target_agent="supervisor",
            message_type="DRAFT_RESPONSE",
            payload={
                "draft": draft,
                "model": gen_result.get("model"),
                "provider": gen_result.get("provider"),
                "temperature": gen_result.get("temperature"),
            },
        )

        return {
            "draft_response": draft,
            "llm_response": draft,
            "final_response": draft,
            "base_llm": gen_result,
            "inter_agent_bus": bus,
            "updated_at": utc_now(),
            "trace": add_trace(
                state,
                "base_llm",
                "completed",
                latency_ms=latency,
                model=gen_result.get("model"),
                provider="openrouter",
            ),
        }
    except Exception as exc:
        update = _failure_update(state, "base_llm", exc)
        update["base_llm"] = {
            "provider": "openrouter",
            "status": "failed",
            "error": str(exc),
        }
        update["final_response"] = "Base LLM generation failed. Please try again."
        return update


def _generate_route(state: HalluciGuardState) -> str:
    if state.get("route") == "error" or not state.get("llm_response"):
        return "human_escalation"
    return "detector"


async def _detector_node(state: HalluciGuardState) -> dict[str, Any]:
    from agents.detector_agent.detector import DetectorAgent

    node_start = start_timer()
    try:
        llm_resp = state.get("llm_response", "")
        if not llm_resp:
            raise ValueError("No LLM response available for detection.")

        def _run_detect():
            return DetectorAgent().detect(state["user_query"], llm_resp)

        detector = _dump(await asyncio.to_thread(_run_detect))
        next_action = str(detector.get("next_action", ""))
        risk_level = str(detector.get("risk_level", "LOW")).upper()
        
        # Verify if detector flags risk, or if mode is stress_test, or if pre-supplied response, or if verification required
        should_verify = (
            next_action.lower().endswith("verify")
            or risk_level in {"MEDIUM", "HIGH"}
            or state.get("generation_mode") == "stress_test"
            or bool(state.get("llm_response") and state.get("llm_response") != state.get("draft_response"))
            or float(detector.get("hallucination_probability", 0.0)) >= 0.05
            or os.environ.get("ALWAYS_VERIFY", "true").lower() in ("true", "1")
        )
        route = "verify" if should_verify else "accept"

        # Inter-agent bus messaging
        if route == "accept":
            bus = add_bus_message(
                state,
                source_agent="detector",
                target_agent="supervisor",
                message_type="DETECTOR_ACCEPT",
                payload={
                    "hallucination_probability": detector.get("hallucination_probability"),
                    "risk_level": detector.get("risk_level", "LOW"),
                },
            )
        else:
            bus = add_bus_message(
                state,
                source_agent="detector",
                target_agent="supervisor",
                message_type="SUSPICIOUS_CLAIMS",
                payload={
                    "suspicious_claims": [llm_resp],
                    "hallucination_probability": detector.get("hallucination_probability"),
                    "risk_level": detector.get("risk_level", "HIGH"),
                },
            )

        return {
            "detector": detector,
            "route": route,
            "hallucination_probability": float(
                detector.get("hallucination_probability", 0.0)
            ),
            "confidence": float(detector.get("confidence_score", 0.0)),
            "verification_status": (
                "detector_safe_fast_path"
                if route == "accept"
                else "verification_required"
            ),
            "inter_agent_bus": bus,
            "updated_at": utc_now(),
            "trace": add_trace(
                state,
                "detector",
                "completed",
                latency_ms=elapsed_ms(node_start),
                route=route,
                risk_level=detector.get("risk_level", "LOW"),
            ),
        }
    except Exception as exc:
        return _failure_update(state, "detector", exc)


def _detector_route(state: HalluciGuardState) -> str:
    if state.get("route") == "error":
        return "human_escalation"
    return "verifier" if state.get("route") == "verify" else "accept"


def _verifier_imports():
    verifier_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "agents", "verifier_agent")
    )
    if verifier_dir not in sys.path:
        sys.path.insert(0, verifier_dir)
    from api.pipeline import VerificationPipeline
    from schemas.models import SuspiciousClaim, VerifierInputV2

    return VerificationPipeline, SuspiciousClaim, VerifierInputV2


async def _verifier_node(state: HalluciGuardState) -> dict[str, Any]:
    VerificationPipeline, SuspiciousClaim, VerifierInputV2 = _verifier_imports()
    node_start = start_timer()
    try:
        claim_text = state.get("user_query") or state.get("llm_response", "")
        payload = VerifierInputV2(
            query_id=state.get("request_id")
            or state.get("execution_id")
            or str(uuid.uuid4()),
            domain=state.get("domain", "general"),
            suspicious_claims=[
                SuspiciousClaim(claim_id="c1", text=claim_text)
            ],
        )
        try:
            verifier_timeout = float(os.environ.get("VERIFIER_TIMEOUT_SECONDS", "60.0"))
            verifier_res = await asyncio.wait_for(VerificationPipeline().verify(payload), timeout=verifier_timeout)
            verifier = _dump(verifier_res)
        except (asyncio.TimeoutError, Exception) as sub_err:
            raise RuntimeError(
                f"Verifier failed: {type(sub_err).__name__}: {sub_err}"
            ) from sub_err
        judge_pairs: list[dict[str, Any]] = []
        evidence_all: list[dict[str, Any]] = []
        nli_results: list[dict[str, Any]] = []
        claims: list[dict[str, Any]] = []
        
        has_contradiction = False
        has_verified = False
        has_conflicted = False

        for report in verifier.get("claim_evidence", []):
            verdict_raw = str(report.get("verdict", "")).lower()
            clean_verdict = "unverified"
            if "contradict" in verdict_raw or "hallucinat" in verdict_raw:
                has_contradiction = True
                clean_verdict = "contradicted"
            elif verdict_raw in ("verified", "supported", "verdictlabel.verified") or (verdict_raw.startswith("verif") and "unverif" not in verdict_raw):
                has_verified = True
                clean_verdict = "verified"
            elif "conflict" in verdict_raw:
                has_conflicted = True
                clean_verdict = "conflicted"

            claims.append(
                {
                    "claim_id": report.get("claim_id"),
                    "text": report.get("claim_text"),
                    "verdict": clean_verdict,
                }
            )
            evidence_items = report.get("evidence", [])
            for evidence in evidence_items:
                evidence_all.append(evidence)
                nli_results.append(
                    {
                        "claim": report.get("claim_text", ""),
                        "label": evidence.get("entailment_label"),
                        "score": evidence.get("entailment_score"),
                    }
                )
                judge_pairs.append(
                    {
                        "claim": report.get("claim_text", ""),
                        "evidence": evidence.get("snippet", ""),
                        "source": evidence.get("source", ""),
                        "url": evidence.get("url", ""),
                        "entailment_label": evidence.get("entailment_label", "neutral"),
                        "entailment_score": evidence.get("entailment_score", 0.0),
                        "credibility_score": evidence.get("credibility_score", 0.0),
                    }
                )

        bus = add_bus_message(
            state,
            source_agent="verifier",
            target_agent="supervisor",
            message_type="VERIFICATION_RESULT",
            payload={
                "claims_count": len(claims),
                "evidence_count": len(evidence_all),
                "has_contradiction": has_contradiction,
                "has_verified": has_verified,
            },
        )

        # -------------------------------------------------------------------
        # Build Canonical VerifierResult Contract (Phase 1)
        # -------------------------------------------------------------------
        from orchestration.schemas import (
            VerifierResult as CanonicalVerifierResult,
            ClaimReport as CanonicalClaimReport,
            Evidence as CanonicalEvidence,
            VerdictLabel as CanonicalVerdictLabel,
            EntailmentLabel as CanonicalEntailmentLabel,
            ExecutionStatus,
        )

        canonical_reports: list[CanonicalClaimReport] = []
        for report in verifier.get("claim_evidence", []):
            c_id = report.get("claim_id", "c1")
            c_text = report.get("claim_text", "")
            verdict_raw = str(report.get("verdict", "")).lower()

            if "contradict" in verdict_raw or "hallucinat" in verdict_raw:
                c_verdict = CanonicalVerdictLabel.CONTRADICTED
            elif "conflict" in verdict_raw:
                c_verdict = CanonicalVerdictLabel.CONFLICTED
            elif verdict_raw in ("verified", "supported", "verdictlabel.verified") or (verdict_raw.startswith("verif") and "unverif" not in verdict_raw):
                c_verdict = CanonicalVerdictLabel.VERIFIED
            else:
                c_verdict = CanonicalVerdictLabel.UNVERIFIED

            canonical_ev_list: list[CanonicalEvidence] = []
            for ev in report.get("evidence", []):
                entail_raw = str(ev.get("entailment_label", "neutral")).lower()
                if "contra" in entail_raw:
                    entail_lbl = CanonicalEntailmentLabel.CONTRADICTION
                elif "entail" in entail_raw or "support" in entail_raw:
                    entail_lbl = CanonicalEntailmentLabel.ENTAILMENT
                else:
                    entail_lbl = CanonicalEntailmentLabel.NEUTRAL

                canonical_ev_list.append(
                    CanonicalEvidence(
                        evidence_id=str(uuid.uuid4())[:8],
                        title=ev.get("title", ""),
                        source=ev.get("source", "Unknown"),
                        url=ev.get("url"),
                        snippet=ev.get("snippet", ""),
                        entailment_label=entail_lbl,
                        entailment_score=float(ev.get("entailment_score", 0.8)),
                        credibility_score=float(ev.get("credibility_score", 0.8)),
                    )
                )

            canonical_reports.append(
                CanonicalClaimReport(
                    claim_id=c_id,
                    claim_text=c_text,
                    verdict=c_verdict,
                    support_score=float(report.get("support_score", 0.9 if c_verdict == CanonicalVerdictLabel.VERIFIED else 0.1)),
                    contradiction_score=float(report.get("contradiction_score", 0.9 if c_verdict == CanonicalVerdictLabel.CONTRADICTED else 0.1)),
                    confidence_score=float(report.get("confidence_score", report.get("trust_score", 0.8))),
                    evidence=canonical_ev_list,
                )
            )

        overall_conf = float(verifier.get("overall_evidence_confidence", 0.8))
        canonical_verifier_result = CanonicalVerifierResult(
            query_id=verifier.get("query_id", payload.query_id),
            domain=verifier.get("domain", payload.domain),
            claim_reports=canonical_reports,
            evidence=[ev for r in canonical_reports for ev in r.evidence],
            overall_confidence=overall_conf,
            status=ExecutionStatus.COMPLETED,
        )

        overall_status = (
            "contradicted" if has_contradiction
            else "conflicted" if has_conflicted
            else "verified" if has_verified
            else "unverified"
        )

        return {
            "verifier": verifier,
            "verifier_result": _dump(canonical_verifier_result),
            "claims": claims,
            "judge_pairs": judge_pairs,
            "evidence": evidence_all,
            "retrieved_evidence": evidence_all,
            "ranked_evidence": evidence_all,
            "nli_results": nli_results,
            "verification_status": overall_status,
            "inter_agent_bus": bus,
            "updated_at": utc_now(),
            "trace": add_trace(
                state,
                "verifier",
                "completed",
                latency_ms=elapsed_ms(node_start),
                claim_count=len(claims),
                evidence_count=len(evidence_all),
                has_contradiction=has_contradiction,
            ),
        }
    except Exception as exc:
        from orchestration.schemas import VerifierResult as CanonicalVerifierResult, ExecutionStatus
        failed_res = CanonicalVerifierResult(
            query_id=payload.query_id if 'payload' in locals() else "q-failed",
            domain=state.get("domain", "general"),
            claim_reports=[],
            evidence=[],
            overall_confidence=0.0,
            status=ExecutionStatus.FAILED,
        )
        update = _failure_update(state, "verifier", exc, retryable=True)
        update["verifier_result"] = _dump(failed_res)
        return update


def _verifier_route(state: HalluciGuardState) -> str:
    if state.get("route") == "error" or state.get("verification_status") == "agent_failed":
        return "human_escalation"
    return "judge"


async def _judge_node(state: HalluciGuardState) -> dict[str, Any]:
    from agents.judge_agent.judge_agent import JudgeAgent

    node_start = start_timer()
    try:
        verifier_output = state.get("verifier_result") or state.get("verifier", {})
        detector_output = state.get("detector_result") or state.get("detector", {})
        user_query = state.get("user_query", "")
        draft_resp = state.get("llm_response") or state.get("draft_response", "")
        domain = state.get("domain", "general")
        retry_count = state.get("retry_count", 0)
        reverification_res = state.get("reverification_result")

        def _run_judge():
            agent = JudgeAgent()
            return agent.evaluate(
                verifier_result=verifier_output,
                detector_result=detector_output,
                user_query=user_query,
                original_response=draft_resp,
                domain=domain,
                reverification_result=reverification_res,
                retry_count=retry_count,
            )

        judge_result = await asyncio.to_thread(_run_judge)
        dumped_judge = _dump(judge_result)

        decision_val = str(dumped_judge.get("decision", "ABSTAIN")).upper()
        severity_val = str(dumped_judge.get("severity", "LOW")).upper()
        corr_req = dumped_judge.get("correction_request")

        if decision_val == "ACCEPT":
            route = "memory"
            verification_status = "verified_and_accepted"
        elif decision_val == "CORRECT":
            route = "corrector" if "corrector" in state.get("active_agents", []) else "memory"
            verification_status = "correction_requested"
        elif decision_val == "VERIFY_AGAIN":
            route = "verifier" if retry_count < state.get("max_retries", 2) else "human_escalation"
            verification_status = "reverification_requested"
        elif decision_val == "REJECT":
            route = "reject"
            verification_status = "rejected_by_judge"
        else:
            route = "human_escalation"
            verification_status = "judge_abstain"

        bus = add_bus_message(
            state,
            source_agent="judge",
            target_agent="supervisor",
            message_type="JUDGE_DECISION",
            payload={
                "decision": decision_val,
                "severity": severity_val,
                "reason": dumped_judge.get("reason"),
                "has_correction_request": corr_req is not None,
            },
        )

        return {
            "judge": dumped_judge,
            "judge_result": dumped_judge,
            "judge_decision": decision_val,
            "severity": severity_val,
            "correction_request": corr_req,
            "route": route,
            "verification_status": verification_status,
            "inter_agent_bus": bus,
            "updated_at": utc_now(),
            "trace": add_trace(
                state,
                "judge",
                "completed",
                latency_ms=elapsed_ms(node_start),
                decision=decision_val,
                severity=severity_val,
            ),
        }
    except Exception as exc:
        return _failure_update(state, "judge", exc)


def _judge_route(state: HalluciGuardState) -> str:
    if state.get("route") == "error":
        return "human_escalation"
    decision = str(state.get("judge_decision", "ACCEPT")).upper()
    if decision == "ACCEPT":
        return "memory"
    elif decision == "CORRECT":
        return "corrector" if "corrector" in state.get("active_agents", []) else "memory"
    elif decision == "VERIFY_AGAIN":
        return "verifier"
    elif decision == "REJECT":
        return "reject"
    return "human_escalation"


async def _memory_node(state: HalluciGuardState) -> dict[str, Any]:
    from agents.memory_agent.memory.memory_agent import MemoryAgent
    from agents.memory_agent.schemas.models import StoreFactRequest

    node_start = start_timer()
    claim_evidence = state.get("verifier", {}).get("claim_evidence", [])
    
    # Check if there are verified facts to store
    verified_reports = [r for r in claim_evidence if str(r.get("verdict", "")).lower() == "verified"]

    if not verified_reports:
        memory = {
            "stored": [],
            "count": 0,
            "knowledge_graph": False,
            "vector_memory": False,
            "skipped_reason": "no_verified_claims_to_persist",
        }
        bus = add_bus_message(
            state,
            source_agent="memory",
            target_agent="supervisor",
            message_type="MEMORY_WRITE_RESULT",
            payload={"stored_count": 0, "status": "skipped", "reason": "no_verified_claims"},
        )
        return {
            "memory": memory,
            "final_response": state.get("final_response") or state.get("llm_response", ""),
            "inter_agent_bus": bus,
            "updated_at": utc_now(),
            "trace": add_trace(
                state,
                "memory",
                "skipped",
                latency_ms=elapsed_ms(node_start),
                reason=memory["skipped_reason"],
            ),
        }
    try:
        memory_agent = MemoryAgent()
        await memory_agent.initialize()
        stored = []
        try:
            for report in verified_reports:
                evidence = report.get("evidence", [])
                sources = [
                    str(e.get("source", "")) for e in evidence if e.get("source")
                ]
                req = StoreFactRequest(
                    claim_text=str(report.get("claim_text", "")),
                    domain=state.get("domain", "general"),
                    verdict="verified",
                    evidence=[
                        {
                            "source_id": e.get("source", ""),
                            "title": e.get("title", ""),
                            "url": e.get("url"),
                            "snippet": e.get("snippet", ""),
                        }
                        for e in evidence
                    ],
                    source_ids=sources,
                    confidence=float(
                        report.get("confidence_score", report.get("trust_score", 0.0))
                    ),
                    metadata={
                        "execution_id": state.get("execution_id", ""),
                        "request_id": state.get("request_id", ""),
                    },
                )
                stored.append(_dump(await memory_agent.store_fact(req)))
        finally:
            await memory_agent.close()

        memory = {
            "stored": stored,
            "count": len(stored),
            "knowledge_graph": True,
            "vector_memory": True,
        }
        bus = add_bus_message(
            state,
            source_agent="memory",
            target_agent="supervisor",
            message_type="MEMORY_WRITE_RESULT",
            payload={"stored_count": len(stored), "status": "stored"},
        )
        return {
            "memory": memory,
            "final_response": state.get("final_response") or state["llm_response"],
            "inter_agent_bus": bus,
            "updated_at": utc_now(),
            "trace": add_trace(
                state,
                "memory",
                "completed",
                latency_ms=elapsed_ms(node_start),
                stored_count=len(stored),
            ),
        }
    except Exception as exc:
        update = _failure_update(state, "memory", exc)
        update["final_response"] = state.get("final_response") or state.get("llm_response", "")
        return update


def _accept_node(state: HalluciGuardState) -> dict[str, Any]:
    return {
        "final_response": state.get("llm_response", ""),
        "terminal_status": "accepted",
        "verification_status": "accepted",
        "updated_at": utc_now(),
        "trace": add_trace(
            state, "accept", "completed", reason="accepted by detector/supervisor"
        ),
    }


def _reject_node(state: HalluciGuardState) -> dict[str, Any]:
    msg = "The draft response could not be safely verified and has been rejected."
    return {
        "final_response": msg,
        "terminal_status": "rejected",
        "verification_status": "rejected",
        "updated_at": utc_now(),
        "trace": add_trace(
            state, "reject", "completed", decision="REJECT"
        ),
    }


def _human_escalation_node(state: HalluciGuardState) -> dict[str, Any]:
    msg = "This response requires human review before it can be delivered."
    return {
        "final_response": msg,
        "terminal_status": "human_review",
        "verification_status": "human_review_required",
        "updated_at": utc_now(),
        "trace": add_trace(
            state,
            "human_escalation",
            "completed",
            errors=state.get("errors", []),
        ),
    }


def build_verification_graph(
    node_overrides: dict[str, Callable[..., Any]] | None = None,
):
    """Build and compile the LangGraph verification pipeline with all nodes and edges."""
    nodes = {
        "generate": _generate_node,
        "detector": _detector_node,
        "accept": _accept_node,
        "verifier": _verifier_node,
        "judge": _judge_node,
        "reject": _reject_node,
        "human_escalation": _human_escalation_node,
        "memory": _memory_node,
    }
    if node_overrides:
        nodes.update(node_overrides)

    graph = StateGraph(HalluciGuardState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "generate")
    graph.add_conditional_edges(
        "generate",
        _generate_route,
        {"detector": "detector", "human_escalation": "human_escalation"},
    )
    graph.add_conditional_edges(
        "detector",
        _detector_route,
        {"verifier": "verifier", "accept": "accept", "human_escalation": "human_escalation"},
    )
    graph.add_conditional_edges(
        "verifier",
        _verifier_route,
        {"judge": "judge", "human_escalation": "human_escalation"},
    )
    graph.add_conditional_edges(
        "judge",
        _judge_route,
        {
            "memory": "memory",
            "verifier": "verifier",
            "reject": "reject",
            "human_escalation": "human_escalation",
        },
    )
    graph.add_edge("accept", END)
    graph.add_edge("reject", END)
    graph.add_edge("human_escalation", END)
    graph.add_edge("memory", END)

    return graph.compile()


_GRAPH = None


def get_verification_graph():
    """Get the singleton verification graph instance, building it if necessary."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_verification_graph()
    return _GRAPH


async def run_verification(
    user_query: str,
    llm_response: str = "",
    domain: str = "general",
    request_id: str | None = None,
    generation_mode: str = "normal",
    conversation_history: list[dict[str, str]] | None = None,
) -> HalluciGuardState:
    """Run the complete verification pipeline and return the final state."""
    execution_id = str(uuid.uuid4())
    now = utc_now()
    active_agents = ["base_llm", "detector", "verifier", "judge", "memory"]
    disabled_agents = ["corrector"]

    return await get_verification_graph().ainvoke(
        {
            "execution_id": execution_id,
            "request_id": request_id or execution_id,
            "user_query": user_query,
            "llm_response": llm_response,
            "draft_response": llm_response,
            "generation_mode": generation_mode,
            "conversation_history": conversation_history or [],
            "domain": domain,
            "active_agents": active_agents,
            "disabled_agents": disabled_agents,
            "created_at": now,
            "updated_at": now,
            "inter_agent_bus": [],
            "trace": [],
            "errors": [],
            "audit": {
                "graph": "halluciguard_production_supervisor",
                "base_llm": "openrouter_qwen",
                "active_agents": active_agents,
                "disabled_agents": disabled_agents,
            },
        }
    )
