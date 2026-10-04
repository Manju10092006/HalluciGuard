"""Run a real query and print a narrated walkthrough of the current graph.

Example: python scripts/run_seven_agent_walkthrough.py "Who founded Microsoft?"
Base LLM generation is separate from the seven downstream agents. Conditional
agents are never forced to run. Uses the existing validation observer without
changing production methods, verdicts, thresholds or graph edges. Memory uses
fresh per-run diagnostic stores. --from-json only replays an existing recording.
Exit 0 means execution/reporting completed, not that every fact was verified.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_seven_agent_validation as validation

BAR = "=" * 80
SUB = "-" * 80
STAGES = (
    ("DETECTOR", "DETECTOR AGENT (Hallucination Risk Triage)",
     "Estimate verification risk, not establish factual correctness.",
     "The query and generated response; evidence only for a grounded pass.",
     "Risk assessment, availability diagnostics and a suggested route.",
     "Evidence-free triage has no grounded neural probability. Low risk is not proof."),
    ("CLAIM_ANALYZER", "CLAIM ANALYZER (Factual Claim Decomposition)",
     "Identify independently checkable factual claims and prepare searches.",
     "The entire generated response and original question.",
     "Claim IDs, factual/discarded classifications and retrieval queries.",
     "It extracts claims; it never decides whether a claim is true."),
    ("VERIFIER", "VERIFIER AGENT (Evidence Retrieval + NLI)",
     "Check factual claims against retrieved evidence; the factual authority.",
     "The gated claims, original question and domain.",
     "Per-claim VERIFIED / CONTRADICTED / UNVERIFIED / CONFLICTED verdicts.",
     "n8n finds evidence; Python ranks it and evaluates support/contradiction."),
    ("JUDGE", "JUDGE AGENT (Evidence-Based Arbitration)",
     "Choose the next action from actual verification results.",
     "Verifier verdicts/evidence, Detector risk and pipeline state.",
     "ACCEPT / CORRECT / REJECT / VERIFY_AGAIN / ABSTAIN.",
     "CORRECT requests a repair; it is not acceptance of an unverified candidate."),
    ("CORRECTOR", "CORRECTOR AGENT (Grounded Fact Repair)",
     "Repair the requested claims using the evidence provided by the Judge.",
     "Original response, claims to correct/preserve and correction evidence.",
     "An untrusted corrected candidate plus validation/failure diagnostics.",
     "Conditional: only invoked when production routing requests correction."),
    ("REVERIFIER", "REVERIFIER (Post-Correction Verification)",
     "Re-extract and verify claims from the corrected candidate.",
     "Corrected text, original question, evidence and Judge state.",
     "Verification results, passed flag and remaining contradictions.",
     "Conditional: a candidate is not factual truth just because it was repaired."),
    ("MEMORY", "MEMORY AGENT (Verified-Knowledge Persistence)",
     "Persist only eligible, accepted, evidence-supported facts.",
     "Final decision, verified claims/evidence and repair history.",
     "Storage status, fact IDs, duplicate/failure counts or an explicit skip.",
     "A Memory graph boundary may execute while actual persistence is skipped."),
)


def mapping(value):
    return value if isinstance(value, dict) else {}


def sequence(value):
    return value if isinstance(value, list) else []


def first(data, *keys):
    """Preserve legitimate zero/False values rather than treating them as absent."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def probability(data, *keys):
    # An explicit null class probability is unavailable, not permission to use
    # a legacy label-confidence field as its replacement.
    for key in keys:
        if key in data:
            return data[key]
    return None


def number(value):
    if value is None or isinstance(value, bool):
        return "N/A"
    try:
        result = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(result):
        return "N/A (invalid/nonfinite)"
    # Do not turn a small but real model probability into an apparent zero.
    return f"{result:.6g}" if 0 < abs(result) < 0.0001 else f"{result:.4f}"


def text(value):
    if value is None:
        return "N/A"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return str(value)


def enum(value):
    return text(value).rsplit(".", 1)[-1].upper()


def value(lines, label, item):
    lines.append(f"  • {label:28} {text(item)}")


def heading(lines, index, title, role, inputs, outputs, note):
    lines.extend(("", SUB, f"{index}. [AGENT] {title}", SUB,
                  "  ▸ ROLE : " + role, "  ▸ IN   : " + inputs,
                  "  ▸ OUT  : " + outputs, "  ▸ NOTE : " + note, ""))


def nodes(run, names):
    return validation.node_events(run, names)


def components(run, call):
    start = call.get("sequence", 0)
    end = min((e.get("sequence", 0) for e in run["events"]
               if e.get("kind") == "node" and e.get("sequence", 0) > start), default=math.inf)
    return [e for e in run["events"] if e.get("kind") == "component"
            and start < e.get("sequence", 0) < end]


def latest_payload(call, *keys):
    out = mapping(call.get("output"))
    for key in keys:
        if isinstance(out.get(key), dict):
            return out[key]
    return {}


def stage_status(lines, run, label, names):
    calls = nodes(run, names)
    status = validation.agent_status(run, label, names)
    value(lines, "Execution status", status)
    value(lines, "Graph-boundary calls", len(calls))
    value(lines, "Observed node time (s)", number(sum(e.get("seconds", 0) for e in calls)) if calls else "N/A")
    if not calls:
        reason = ("Production did not request this conditional stage; no execution demonstrated."
                  if status == "NOT_REQUIRED" else "Not reached/executed on this route.")
        value(lines, "Reason", reason)
    return calls


def model_calls(lines, events):
    calls = [e for e in events if e.get("name") == "BaseLLMService.generate"]
    for call in calls:
        out = mapping(call.get("output"))
        value(lines, "Actual provider", first(out, "provider_used", "provider"))
        value(lines, "Actual model", out.get("model"))
        value(lines, "Provider status", out.get("status", call.get("status")))
        value(lines, "Finish reason", out.get("finish_reason"))


def evidence(lines, item, index):
    item = mapping(item)
    lines.append(f"      ({index}) [{text(item.get('source'))}] {text(item.get('title'))}")
    lines.append("          Source ID: " + text(item.get("source_id")))
    lines.append("          URL: " + text(item.get("url")))
    lines.append("          Evidence: " + text(item.get("snippet")))
    lines.append("          NLI label: " + text(first(item, "entailment_label", "label")))
    lines.append("          P(entailment)=" + number(probability(item, "nli_entailment", "entailment", "entailment_score"))
                 + " P(contradiction)=" + number(probability(item, "nli_contradiction", "contradiction"))
                 + " P(neutral)=" + number(probability(item, "nli_neutral", "neutral")))


def backend(lines, data):
    data = mapping(data)
    if not data:
        value(lines, "Backend execution", "N/A - no actual execution diagnostics")
        return
    # A configured route label does not establish any of these executions.
    for label, key in (("Actual retrieval route", "route"), ("Retrieval degraded", "degraded"),
                       ("BM25/sparse executed", "sparse_executed"), ("Dense executed", "dense_executed"),
                       ("Dense available", "dense_available"), ("Fusion executed", "fusion_executed"),
                       ("Sparse status", "sparse_status"), ("Dense status", "dense_status"),
                       ("Fallback reason", "fallback_reason"), ("Dense failure stage", "dense_failure_stage")):
        value(lines, label, data.get(key))


def execution_proof(lines, name, data):
    data = mapping(data)
    value(lines, name + " status", data.get("status"))
    value(lines, name + " model", first(data, "model_name", "model"))
    value(lines, name + " loaded", first(data, "loaded", "model_loaded"))
    value(lines, name + " inference executed", data.get("inference_executed"))
    value(lines, name + " degraded", data.get("degraded"))


def retrieval_calls(lines, events):
    n8n = [e for e in events if e.get("name") == "N8NRetrievalClient.retrieve_evidence"]
    value(lines, "Actual n8n calls", len(n8n))
    for index, call in enumerate(n8n, 1):
        out, inp = mapping(call.get("output")), mapping(call.get("input"))
        trace = mapping(out.get("trace"))
        value(lines, f"n8n call {index} claim", inp.get("claim"))
        value(lines, "Retrieval queries", inp.get("queries"))
        value(lines, "HTTP / success", f"{text(out.get('http_status'))} / {text(out.get('success'))}")
        value(lines, "Returned passages", len(sequence(out.get("passages"))) if "passages" in out else None)
        value(lines, "Upstream counts", trace.get("counts"))
        if call.get("status") == "FAILED":
            value(lines, "Call failure", call.get("error_type"))
    for call in events:
        name = call.get("name", "")
        if name == "HybridRetriever.retrieve":
            backend(lines, call.get("diagnostics"))
        elif name in {"CrossEncoderReranker.rerank", "NLIEngine.batch_classify"}:
            diag = mapping(call.get("diagnostics"))
            value(lines, name, f"boundary={call.get('status')}; inference_executed={text(diag.get('inference_executed'))}")
            value(lines, "Component model", first(diag, "model_name", "model"))
            value(lines, "Component degraded", diag.get("degraded"))
    if not any(e.get("name") == "HybridRetriever.retrieve" for e in events):
        value(lines, "Backend execution", "N/A - no component observation; see per-claim diagnostics if present")


def claim_reports(lines, reports):
    for index, report in enumerate(reports, 1):
        report = mapping(report)
        lines.append(f"\n    ── Claim #{index} [{text(report.get('claim_id'))}]: {text(first(report, 'claim_text', 'text', 'claim'))}")
        value(lines, "Verdict", enum(report.get("verdict")))
        value(lines, "Confidence", number(report.get("confidence_score")))
        value(lines, "Support score", number(report.get("support_score")))
        value(lines, "Contradiction score", number(report.get("contradiction_score")))
        value(lines, "Explanation", report.get("explanation"))
        trace = mapping(report.get("retrieval_trace"))
        value(lines, "Retrieval queries", trace.get("generated_queries"))
        flow = mapping(trace.get("evidence_flow"))
        flow_keys = ("raw_candidates", "deduplicated_candidates", "after_relevance_ranking",
                     "hybrid_selected", "reranker_selected", "nli_selected", "decision_grade_selected")
        value(lines, "Evidence flow counts", " → ".join(k + "=" + text(flow[k]) for k in flow_keys if k in flow) or None)
        if trace.get("backend_execution"):
            backend(lines, trace["backend_execution"])
        for name, key in (("Reranker", "reranker_execution"), ("NLI", "nli_execution")):
            if trace.get(key):
                execution_proof(lines, name, trace[key])
        for label, key in (("Retrieved candidates (not verdicts)", "retrieved_documents"),
                           ("Reranked candidates (not verdicts)", "reranked_documents")):
            documents = sequence(report.get(key))
            if documents:
                value(lines, label, len(documents))
                for rank, document in enumerate(documents[:3], 1):
                    lines.append(f"      #{rank} [{text(document.get('source'))}] {text(document.get('title'))}")
                    value(lines, "Candidate source ID", document.get("source_id"))
                    value(lines, "Candidate ranking score", number(first(document, "bge_score", "relevance_score")))
                    snippet = text(document.get("snippet"))
                    value(lines, "Candidate text", snippet[:300] + (" [display shortened; full text in raw JSON]" if len(snippet) > 300 else ""))
        admitted = sequence(report.get("evidence"))
        value(lines, "Decision-grade evidence", len(admitted))
        for ev_index, item in enumerate(admitted, 1):
            evidence(lines, item, ev_index)
        # Rejected neural outputs are not final factual verdicts.
        for rejected in sequence(flow.get("nli_outputs")):
            if rejected.get("selected_for_decision") is False:
                value(lines, "Rejected NLI source", rejected.get("source_id"))
                value(lines, "Rejected raw NLI label", rejected.get("label"))
                value(lines, "Rejection reason", rejected.get("rejection_reason"))
                lines.append("      Raw NLI output was NOT admitted as factual decision evidence.")


def claim_component_events(report, events):
    """Associate observations by claim ID or exact claim text, never similarity.

    Production strips a final full stop before calling retrieval/NLI. Apart
    from that formatting boundary, different claims must not share scores.
    """
    claim_id = report.get("claim_id")
    claim = first(report, "claim_text", "text", "claim")

    def key(item):
        return " ".join(str(item).split()).rstrip(". ") if item else ""

    matched = []
    for event in events:
        inp = mapping(event.get("input"))
        supplied_id = inp.get("claim_id")
        if claim_id and supplied_id:
            matches = claim_id == supplied_id
        else:
            supplied_claim = first(inp, "claim", "query")
            matches = bool(key(claim)) and key(claim) == key(supplied_claim)
        if matches:
            matched.append(event)
    return matched


def format_walkthrough(run, report_dir=None):
    """Presentation only: no model calls and no mutation of the recorded run."""
    run = dict(validation.scrub(run))
    run.setdefault("draft", None)
    run.setdefault("events", [])
    run.setdefault("result", {})
    result = mapping(run["result"])
    lines = [BAR, "  HALLUCIGUARD — 7 DOWNSTREAM AGENTS, FULL BACKEND WALKTHROUGH", BAR]
    value(lines, "Input query", run.get("query"))
    value(lines, "Domain", result.get("domain"))
    value(lines, "Mode", run.get("mode"))
    value(lines, "Code snapshot", mapping(run.get("git")).get("head"))
    lines.extend(("", "  Generator = Base LLM. The seven downstream agents are shown separately.",
                  "  FIRED means actual execution, NOT a factual PASS.",
                  "  No conditional agent is forced to run. Missing values stay N/A.",
                  "", BAR, "  PIPELINE EXECUTION — STAGE BY STAGE", BAR, "", SUB,
                  "0. [GENERATOR] BASE LLM (Response Generator)", SUB,
                  "  ▸ ROLE : generate an initial answer using the configured hosted LLM.",
                  "  ▸ IN   : your query and optional conversation history.",
                  "  ▸ OUT  : an unverified response for the downstream agents to check.",
                  "  ▸ WHY  : generation alone does not establish factual correctness.", ""))
    value(lines, "Execution status", validation.generator_status(run))
    base_calls = [e for e in run["events"] if e.get("name") == "BaseLLMService.generate" and e.get("node") == "generate"]
    model_calls(lines, base_calls)
    if not base_calls:
        value(lines, "Reason", "Supplied initial draft; Base LLM skipped." if run["draft"] is not None else "No generation call observed.")
    value(lines, "Generation time (s)", number(sum(e.get("seconds", 0) for e in base_calls)) if base_calls else "N/A")
    value(lines, "Initial response", first(result, "draft_response", "llm_response") or run.get("draft"))

    for index, (label, title, role, inputs, outputs, note) in enumerate(STAGES, 1):
        heading(lines, index, title, role, inputs, outputs, note)
        names = next(n for a, n, _ in validation.AGENTS if a == label)
        calls = stage_status(lines, run, label, names)
        for attempt, call in enumerate(calls, 1):
            out, inp = mapping(call.get("output")), mapping(mapping(call.get("input")).get("state"))
            events = components(run, call)
            value(lines, f"Attempt {attempt} node", call.get("name"))
            value(lines, "Boundary result", call.get("status"))
            if call.get("error_type"):
                value(lines, "Failure type", call["error_type"])
            if label == "DETECTOR":
                d = latest_payload(call, "detector_result", "detector")
                score = first(d, "verification_risk", "hallucination_probability")
                if d.get("probability_available") is False:
                    score = None
                for key, item in (("Risk level", enum(d.get("risk_level"))),
                                  ("Verification risk", number(score)),
                                  ("Contradiction mass", number(d.get("contradiction_mass"))),
                                  ("Confidence", number(first(d, "confidence_score", "confidence"))),
                                  ("Recommended route", enum(d.get("next_action"))),
                                  ("Probability semantics", d.get("probability_semantics")),
                                  ("Model source", d.get("model_source")),
                                  ("Model loaded", d.get("model_loaded")),
                                  ("Inference executed", d.get("inference_executed")),
                                  ("Detector degraded", first(d, "detector_degraded", "degraded")),
                                  ("Degraded reason", d.get("degraded_reason"))):
                    value(lines, key, item)
            elif label == "CLAIM_ANALYZER":
                analysis = latest_payload(call, "claim_analysis")
                for component in events:
                    if component.get("name") == "ClaimAnalyzer.analyze":
                        analysis = mapping(component.get("output"))
                model_calls(lines, events)
                candidates = sequence(analysis.get("candidates"))
                factual = [c for c in candidates if c.get("claim_type") == "FACTUAL_CLAIM" or c.get("is_factual") is True]
                value(lines, "Extraction source", analysis.get("source"))
                value(lines, "Extraction diagnostic", analysis.get("llm_error"))
                value(lines, "Factual candidates", len(factual))
                value(lines, "Discarded candidates", len(candidates) - len(factual))
                for candidate in candidates:
                    lines.append(f"    [{text(candidate.get('claim_id'))}] {text(candidate.get('claim_type'))}: {text(first(candidate, 'claim_text', 'text'))}")
                    value(lines, "Search queries", candidate.get("search_queries"))
            elif label == "VERIFIER":
                raw, canonical = latest_payload(call, "verifier"), latest_payload(call, "verifier_result")
                reports = sequence(raw.get("claim_evidence")) or sequence(canonical.get("claim_reports"))
                value(lines, "Verifier status", first(canonical, "status") or raw.get("status"))
                value(lines, "Returned claim reports", len(reports))
                value(lines, "Sources actually attempted", raw.get("sources_attempted"))
                value(lines, "Final verification status", out.get("verification_status"))
                retrieval_calls(lines, events)
                claim_reports(lines, reports)
                if not reports:
                    value(lines, "Completed claim verdicts", "N/A - none returned; partial calls are not completed verification.")
            elif label == "JUDGE":
                judge = latest_payload(call, "judge_result", "judge")
                for key, item in (("Decision", enum(first(judge, "decision") or out.get("judge_decision"))),
                                  ("Severity", enum(judge.get("severity"))),
                                  ("Confidence", number(judge.get("confidence"))),
                                  ("Reason", first(judge, "reason", "explanation")),
                                  ("Decision basis", judge.get("decision_basis")),
                                  ("Correction requested", bool(judge.get("correction_request") or out.get("correction_request")))):
                    value(lines, key, item)
            elif label == "CORRECTOR":
                request = mapping(inp.get("correction_request"))
                value(lines, "Requested incorrect claims", request.get("claims_to_correct"))
                value(lines, "Trusted correction evidence", request.get("trusted_evidence"))
                value(lines, "Contradictory evidence", request.get("contradictory_evidence"))
                corrected = latest_payload(call, "correction_result", "corrector")
                model_calls(lines, events)
                for key, item in (("Operation status", corrected.get("status")), ("Validation status", corrected.get("validation_status")),
                                  ("Original response", corrected.get("original_text")), ("Corrected candidate", corrected.get("corrected_text")),
                                  ("Changed claims", corrected.get("changed_claims")), ("Failure category", corrected.get("failure_category"))):
                    value(lines, key, item)
                lines.append("  Candidate remains untrusted until the required re-verification/decision.")
            elif label == "REVERIFIER":
                rev = latest_payload(call, "reverification_result")
                value(lines, "Candidate checked", inp.get("corrected_response"))
                value(lines, "Operation status", rev.get("status"))
                value(lines, "Passed", rev.get("passed"))
                value(lines, "Remaining contradictions", rev.get("remaining_contradictions"))
                value(lines, "Failure category", rev.get("failure_category"))
                retrieval_calls(lines, events)
                claim_reports(lines, sequence(mapping(rev.get("verifier_result")).get("claim_reports")))
            elif label == "MEMORY":
                memory, stores = latest_payload(call, "memory_result"), latest_payload(call, "memory")
                for key, item in (("Persistence status", memory.get("status")), ("New verified facts stored", first(memory, "stored_count", "count")),
                                  ("Duplicates reused", memory.get("duplicate_count")), ("Failed fact writes", memory.get("failed_count")),
                                  ("Stored/referenced fact IDs", memory.get("fact_ids")), ("Knowledge graph active", stores.get("knowledge_graph")),
                                  ("Vector memory active", stores.get("vector_memory")), ("Reason", first(memory, "reason", "skipped_reason"))):
                    value(lines, key, item)

    lines.extend(("", BAR, "  SUPERVISOR SUMMARY (actual execution, not a hypothetical route)", BAR))
    value(lines, "Route taken", " → ".join(e.get("name", "unknown") for e in run["events"] if e.get("kind") == "node") or "N/A")
    value(lines, "Terminal status", result.get("terminal_status"))
    value(lines, "Verification status", result.get("verification_status"))
    value(lines, "Total run time (s)", number(run.get("seconds")))
    value(lines, "Downstream agent counts", validation.counts(run))
    value(lines, "Checkpoint unchanged", run.get("checkpoint_unchanged"))
    value(lines, "Graph failures", result.get("errors"))
    lines.extend(("", "  • FINAL ANSWER DELIVERED TO USER:", "      " + text(result.get("final_response")),
                  "", "  EXECUTION TRACE (observed graph boundaries):"))
    for call in run["events"]:
        if call.get("kind") == "node":
            lines.append(f"    {call.get('sequence')} {call.get('name')} boundary={call.get('status')} time={number(call.get('seconds'))}s")
    lines.append("  Boundaries can return a failure/skip contract; RETURNED is not a factual PASS.")
    lines.append("  Exact handoffs, selected evidence and component calls remain in raw execution JSON.")
    lines.extend(("", "  AUTHORITATIVE GRAPH TRACE:", validation.pretty(result.get("trace", []))))
    if report_dir:
        value(lines, "Saved output directory", str(report_dir))
        value(lines, "Saved walkthrough", str(Path(report_dir) / "WALKTHROUGH.txt"))
    lines.append(BAR)
    return "\n".join(lines) + "\n"


def format_readable_walkthrough(run, report_dir=None):
    """Narrated user-facing view. Never substitutes a score or verdict.

    Raw execution and the detailed formatter retain every observation. This
    view omits irrelevant counters/nullable diagnostics, not real failures.
    """
    run = mapping(validation.scrub(run))
    run = {"events": [], "result": {}, "draft": None, **run}
    result = mapping(run["result"])
    lines = [BAR, "      HALLUCIGUARD — 7-AGENT PIPELINE, LIVE BACKEND WALKTHROUGH", BAR]

    def shown(label, item, missing=None):
        if item is None:
            if missing:
                value(lines, label, missing)
            return
        if isinstance(item, bool):
            item = "Yes" if item else "No"
        elif isinstance(item, list):
            item = ", ".join(str(v) for v in item) or "None"
        value(lines, label, item)

    def score(label, item, missing=None):
        formatted = number(item)
        shown(label, formatted if not formatted.startswith("N/A") else None, missing)

    def probability_value(item):
        if isinstance(item, bool):
            return None
        try:
            numeric = float(item)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) and 0 <= numeric <= 1 else None

    def observed_models(events):
        for event in events:
            if event.get("name") == "BaseLLMService.generate":
                out = mapping(event.get("output"))
                shown("Model provider", first(out, "provider_used", "provider"))
                shown("Model name", out.get("model"))
                shown("Provider status", out.get("status"))

    def retrieval_summary(events, reports):
        upstream = [e for e in events if e.get("name") == "N8NRetrievalClient.retrieve_evidence"]
        if upstream:
            returned = [mapping(e.get("output")) for e in upstream]
            successes = sum(o.get("success") is True for o in returned)
            passages = sum(len(sequence(o.get("passages"))) for o in returned)
            shown("n8n evidence retrieval", f"{successes} successful response(s) from {len(upstream)} call(s); {passages} candidate passage(s)")
        diagnostics = [mapping(e.get("diagnostics")) for e in events if e.get("name") == "HybridRetriever.retrieve"]
        diagnostics += [mapping(mapping(r.get("retrieval_trace")).get("backend_execution")) for r in reports]
        routes = sorted({str(d["route"]) for d in diagnostics if d.get("route")})
        if routes:
            shown("Actual retrieval route", routes)
        else:
            shown("Backend execution", "This checkout does not expose a complete backend execution trace.")
        reasons = sorted({str(d["fallback_reason"]) for d in diagnostics if d.get("fallback_reason")})
        if reasons:
            shown("Retrieval fallback", reasons)
        if any(d.get("degraded") is True for d in diagnostics):
            shown("Retrieval condition", "Degraded — inspect the recorded fallback before trusting coverage.")

    def component_proof(name, data, boundary=None):
        data = mapping(data)
        shown(name + " model", first(data, "model_name", "model"))
        if data.get("inference_executed") is True:
            state = "Inference executed"
        elif data.get("inference_executed") is False:
            state = "No inference executed"
        else:
            state = "Model inference not reported"
        condition = "; degraded fallback" if data.get("degraded") is True else ""
        shown(name + " execution", state + condition + (f"; boundary {boundary}" if boundary else ""))
        shown(name + " status", data.get("status"))

    def source_document(doc, rank, ranking=None):
        doc = mapping(doc)
        lines.append(f"      ({rank}) [{text(doc.get('source'))}] {text(doc.get('title'))}")
        shown("Source ID", doc.get("source_id") or None, "Not exposed by this record")
        shown("Source link", first(doc, "url", "source_url") or None, "Not exposed by this record")
        if ranking:
            score(ranking, first(doc, "bge_score", "relevance_score"), "Not reported")
        snippet = first(doc, "snippet", "text")
        if snippet is not None:
            snippet = text(snippet)
            shown("Evidence text", snippet[:420] + ("… [display shortened; complete text saved]" if len(snippet) > 420 else ""))

    def claim_retrieval_details(report, events):
        matched = claim_component_events(report, events)
        trace = mapping(report.get("retrieval_trace"))
        for snapshot in sequence(trace.get("subclaim_execution")):
            snapshot = mapping(snapshot)
            if snapshot.get("parent_claim_id") != report.get("claim_id"):
                continue
            flow_counts = mapping(snapshot.get("counts"))
            for title, key in (("Merged retrieval candidates", "retrieved"),
                               ("Relevance-selected candidates", "relevance_selected"),
                               ("NLI-selected passages", "nli_selected")):
                if key in flow_counts:
                    shown(title, flow_counts[key])
            for lookup in sequence(snapshot.get("supplementary_retrieval")):
                lookup = mapping(lookup)
                shown("Supplementary source lookup", lookup.get("adapter"))
                shown("Supplementary query", lookup.get("query"))
                shown("Supplementary retrieval", lookup.get("status"))
                shown("Supplementary passages", lookup.get("returned_count"))
                shown("Supplementary reason", lookup.get("reason"))
                shown("Supplementary failure", lookup.get("error_type"))
        queries = list(sequence(trace.get("generated_queries")))
        retrieval_events = [e for e in matched if e.get("name") == "N8NRetrievalClient.retrieve_evidence"]
        for event in retrieval_events:
            inp = mapping(event.get("input"))
            queries.extend(q for q in sequence(inp.get("queries")) if q not in queries)
        shown("Retrieval queries", queries or None, "Not recorded for this claim")
        for event in retrieval_events:
            passages = sequence(mapping(event.get("output")).get("passages"))
            shown("n8n candidates for this claim", f"{len(passages)} passage(s); retrieval alone is not a verdict; this is not the merged retrieval total")
            for rank, doc in enumerate(passages, 1):
                source_document(doc, rank)
        for event in matched:
            if event.get("name") == "BM25Retriever.retrieve":
                shown("BM25 retrieval", f"Boundary {event.get('status')}; {len(sequence(event.get('output')))} result(s) returned")
            elif event.get("name") == "DenseRetriever.retrieve":
                component_proof("Dense retriever", event.get("diagnostics"), event.get("status"))
            elif event.get("name") == "HybridRetriever.retrieve":
                diag = mapping(event.get("diagnostics"))
                shown("Hybrid component boundary", event.get("status"))
                shown("Observed backend route", diag.get("route"), "Not exposed; the HybridRetriever class name is not proof of dense/fusion execution")
                shown("Backend fallback reason", diag.get("fallback_reason"))
        # Some contracts embed documents, while the production ClaimReport
        # carries only counts. Do not treat a count as a list of documents.
        if not retrieval_events:
            for rank, doc in enumerate(sequence(report.get("retrieved_documents")), 1):
                source_document(doc, rank)
        reranks = [e for e in matched if e.get("name") == "CrossEncoderReranker.rerank"]
        reranked = []
        for event in reranks:
            component_proof("BGE reranker", event.get("diagnostics"), event.get("status"))
            docs = sequence(event.get("output"))
            reranked.extend(docs)
            shown("BGE ranking", f"{len(docs)} ranked passage(s); relevance is not factual support")
            is_neural = mapping(event.get("diagnostics")).get("inference_executed") is True
            for rank, doc in enumerate(docs, 1):
                source_document(doc, rank, "BGE relevance score" if is_neural else "Reranker fallback score")
        if not reranks:
            # Ranking often uses an object-anchored query rather than the NLI
            # assertion. Use the pipeline's explicit parent-ID binding, never
            # attach observations by query resemblance across different claims.
            snapshots = [mapping(s) for s in sequence(trace.get("subclaim_execution"))
                         if mapping(s).get("parent_claim_id") == report.get("claim_id")]
            for snapshot in snapshots:
                shown("Ranking query", snapshot.get("retrieval_query"))
                backend = mapping(snapshot.get("backend_execution"))
                shown("Observed backend route", backend.get("route"))
                shown("Backend fallback reason", backend.get("fallback_reason"))
                diag = mapping(snapshot.get("reranker_execution"))
                component_proof("BGE reranker", diag)
                docs = sequence(snapshot.get("reranked_passages"))
                reranked.extend(docs)
                shown("BGE ranking", f"{len(docs)} ranked passage(s); relevance is not factual support")
                for rank, doc in enumerate(docs, 1):
                    source_document(doc, rank, "BGE relevance score"
                                    if diag.get("inference_executed") is True else "Reranker fallback score")
            if not snapshots and trace.get("reranker_execution"):
                component_proof("BGE reranker", trace["reranker_execution"])
        nli_calls = [e for e in matched if e.get("name") == "NLIEngine.batch_classify"]
        for event in nli_calls:
            component_proof("NLI", event.get("diagnostics"), event.get("status"))
            snippets = sequence(mapping(event.get("input")).get("evidences"))
            predictions = sequence(event.get("output"))
            neural = mapping(event.get("diagnostics")).get("inference_executed") is True and mapping(event.get("diagnostics")).get("degraded") is not True
            for rank, pred in enumerate(predictions, 1):
                pred = mapping(pred)
                snippet = snippets[rank - 1] if rank <= len(snippets) else None
                docs = [mapping(d) for d in reranked if mapping(d).get("snippet") == snippet]
                lines.append(f"      NLI pair #{rank} — observed NLI output, not the final verdict")
                if len(docs) == 1:
                    shown("NLI evidence title", docs[0].get("title"))
                    shown("NLI evidence link", docs[0].get("url"))
                elif snippet is not None:
                    shown("NLI evidence text", snippet)
                prefix = "Neural" if neural else "Observed NLI"
                shown(prefix + " predicted label", pred.get("label"))
                score(prefix + " entailment", probability(pred, "entailment_score", "entailment"))
                score(prefix + " contradiction", probability(pred, "contradiction_score", "contradiction"))
                score(prefix + " neutral", probability(pred, "neutral_score", "neutral"))
                lines.append("      These scores describe the selected text before tokenizer truncation; token survival is not exposed here.")
        if not nli_calls and trace.get("nli_execution"):
            component_proof("NLI", trace["nli_execution"])
        # Preserve any actual rejection diagnostic, without guessing why a
        # neutral prediction or candidate was not admitted to the decision.
        for rejected in sequence(mapping(trace.get("evidence_flow")).get("nli_outputs")):
            rejected = mapping(rejected)
            if rejected.get("selected_for_decision") is False:
                shown("Rejected source ID", rejected.get("source_id"))
                shown("Decision gate reason", rejected.get("rejection_reason"), "Reason not exposed by this record")
        for item in sequence(trace.get("evidence_details")):
            item = mapping(item)
            shown("Evidence classification", item.get("evidence_class"))
            shown("Relation gate result", item.get("relation_check_result"))
        return reranked

    shown("Input query", run.get("query"))
    shown("Domain", result.get("domain"), "general")
    shown("Mode", "normal — real LLM generation" if run["draft"] is None else "supplied-answer correction diagnostic; initial LLM skipped")
    lines.extend(("", BAR, "  PIPELINE EXECUTION — STAGE BY STAGE", BAR, "", SUB,
                  "0. [GENERATOR] BASE LLM (Response Generator)", SUB,
                  "  ▸ ROLE : produce the initial answer using your configured LLM.",
                  "  ▸ IN   : your question.",
                  "  ▸ OUT  : a generated answer that the Verifier checks before release.", ""))
    shown("Execution status", validation.generator_status(run))
    generation = [e for e in run["events"] if e.get("name") == "BaseLLMService.generate" and e.get("node") == "generate"]
    observed_models(generation)
    if generation:
        score("Generation time (s)", sum(e.get("seconds", 0) for e in generation))
    elif run["draft"] is not None:
        shown("Reason", "Initial LLM intentionally skipped for the supplied-answer test.")
    initial = first(result, "draft_response", "llm_response") or run["draft"]
    shown("Generated answer" if run["draft"] is None else "Supplied test answer", initial, "Generation did not return an answer.")

    for index, (label, title, role, inputs, outputs, note) in enumerate(STAGES, 1):
        heading(lines, index, title, role, inputs, outputs, note)
        names = next(n for a, n, _ in validation.AGENTS if a == label)
        calls = nodes(run, names)
        status = validation.agent_status(run, label, names)
        shown("Execution status", status)
        if not calls:
            shown("Reason", "Judge did not request correction." if label == "CORRECTOR" and status == "NOT_REQUIRED"
                  else "No corrected answer needed re-verification." if label == "REVERIFIER" and status == "NOT_REQUIRED"
                  else "This stage was not reached on the actual route.")
            continue
        score("Stage time (s)", sum(e.get("seconds", 0) for e in calls))
        for call in calls:
            if call.get("error_type"):
                shown("Failure", call["error_type"])

        if label == "DETECTOR":
            d = latest_payload(calls[-1], "detector_result", "detector")
            shown("Risk level", enum(d.get("risk_level")))
            available = d.get("probability_available") is not False
            risk = probability_value(first(d, "verification_risk", "hallucination_probability")) if available else None
            score("Verification risk", risk, "Unavailable — no grounded prediction; verification is required.")
            if risk is not None:
                score("Contradiction mass", probability_value(d.get("contradiction_mass")))
                score("Detector-reported confidence", probability_value(first(d, "confidence_score", "confidence")))
            shown("Suggested route", enum(d.get("next_action")))
            inference = d.get("inference_executed")
            degraded = first(d, "detector_degraded", "degraded") is True
            failed_detection = degraded or str(d.get("status", "")).lower() in {"failed", "degraded", "unavailable", "error"}
            ungrounded = d.get("grounded") is False and not failed_detection
            shown("Model execution", "Grounded inference executed." if inference is True else
                  "Detector unavailable or failed; no neural inference executed. Verification remains required." if inference is False and failed_detection else
                  "Evidence-free triage only; no neural inference executed." if inference is False and ungrounded else
                  "No neural inference executed; consult the recorded Detector status." if inference is False else
                  "Inference execution not exposed by this checkout.")
            shown("Detector condition", "Degraded" if first(d, "detector_degraded", "degraded") is True else
                  "Not reported degraded" if first(d, "detector_degraded", "degraded") is False else "Availability not reported")
            for warning in sequence(d.get("warnings")):
                shown("Detector warning", warning)
            lines.append("  Verification risk can include insufficient evidence and deterministic guard adjustments; it is not a calibrated probability of falsehood.")
        elif label == "CLAIM_ANALYZER":
            for call in calls:
                events = components(run, call)
                analysis = latest_payload(call, "claim_analysis")
                for component in events:
                    if component.get("name") == "ClaimAnalyzer.analyze":
                        analysis = mapping(component.get("output"))
                observed_models(events)
                candidates = sequence(analysis.get("candidates"))
                shown("Extraction method", analysis.get("source"))
                shown("Extraction limitation", analysis.get("llm_error"))
                factual = [c for c in candidates if c.get("claim_type") == "FACTUAL_CLAIM" or c.get("is_factual") is True]
                shown("Factual claims extracted", len(factual))
                for c in factual:
                    lines.append(f"    [{text(c.get('claim_id'))}] {text(first(c, 'claim_text', 'text'))}")
        elif label == "VERIFIER":
            for attempt, call in enumerate(calls, 1):
                if len(calls) > 1:
                    shown("Verification attempt", attempt)
                raw = latest_payload(call, "verifier", "verifier_result")
                reports = sequence(first(raw, "claim_evidence", "claim_reports"))
                shown("Atomic claims evaluated", len(reports))
                shown("Verification status", mapping(call.get("output")).get("verification_status"))
                retrieval_summary(components(run, call), reports)
                for rank, report in enumerate(reports, 1):
                    report = mapping(report)
                    lines.append(f"\n    ── Claim #{rank} [{text(report.get('claim_id'))}]: {text(first(report, 'claim_text', 'text', 'claim'))}")
                    shown("Verdict", enum(report.get("verdict")))
                    shown("Explanation", report.get("explanation"))
                    claim_retrieval_details(report, components(run, call))
                    admitted = sequence(report.get("evidence"))
                    shown("Decision-grade evidence", f"{len(admitted)} admitted passage(s)" if admitted else "None — no passage passed the decision gate.")
                    for ev_rank, ev in enumerate(admitted, 1):
                        ev = mapping(ev)
                        label_value = first(ev, "entailment_label", "label")
                        class_key = {"entailment": "nli_entailment", "contradiction": "nli_contradiction", "neutral": "nli_neutral"}.get(str(label_value).lower())
                        class_score = probability_value(ev.get(class_key)) if class_key else None
                        # Older checkouts can apply deterministic relation floors
                        # to these fields. Do not call the exported score a raw
                        # neural probability (it may not even sum to one).
                        tail = f" reported {label_value} score={number(class_score)}" if class_score is not None else ""
                        snippet = text(ev.get("snippet"))
                        if len(snippet) > 420:
                            snippet = snippet[:420] + "… [display shortened; complete text saved]"
                        lines.append(f"      ({ev_rank}) [{text(ev.get('source'))}] NLI={text(label_value)}{tail}: {snippet}")
                        shown("Evidence title", ev.get("title"))
                        shown("Source", ev.get("url") or None, "Not exposed by this record")
                        score("Reported BGE score", ev.get("bge_score"))
                        shown("Decision classification", ev.get("classification"))
                        lines.append("      Exported decision scores may include relation-rule adjustments; observed neural outputs are listed separately above.")
                if not reports:
                    shown("Outcome", "No completed claim verdicts returned; this is not successful verification.")
        elif label == "JUDGE":
            for call in calls:
                judge = latest_payload(call, "judge_result", "judge")
                shown("Decision", enum(first(judge, "decision") or mapping(call.get("output")).get("judge_decision")))
                shown("Severity", enum(judge.get("severity")))
                score("Decision confidence", probability_value(judge.get("confidence")))
                shown("Reason", first(judge, "reason", "explanation"))
        elif label == "CORRECTOR":
            for call in calls:
                corrected = latest_payload(call, "correction_result", "corrector")
                request = mapping(mapping(mapping(call.get("input")).get("state")).get("correction_request"))
                for claim in sequence(request.get("claims_to_correct")):
                    claim = mapping(claim)
                    shown("Claim requiring repair", first(claim, "claim_text", "text"))
                for ev in sequence(request.get("evidence")):
                    ev = mapping(ev)
                    shown("Evidence supplied for repair", ev.get("snippet"))
                    shown("Repair evidence source", ev.get("url"))
                shown("Operation status", corrected.get("status"))
                shown("Corrected candidate", corrected.get("corrected_text"))
                shown("Validation", corrected.get("validation_status"))
                shown("Failure category", corrected.get("failure_category"))
        elif label == "REVERIFIER":
            for call in calls:
                rev = latest_payload(call, "reverification_result")
                shown("Operation status", rev.get("status"))
                shown("Re-verification result", "PASSED" if rev.get("passed") is True else "FAILED — candidate must not be released" if rev.get("passed") is False else "No completed verdict reported.")
                shown("Failure category", rev.get("failure_category"))
                coverage = mapping(rev.get("claim_coverage"))
                if coverage:
                    shown("Corrected claims checked", f"{coverage.get('reported_count')} reports for {coverage.get('requested_count')} extracted claims")
                    shown("Claim coverage", "Complete" if coverage.get("complete") is True else "Incomplete — release blocked")
                verified = mapping(rev.get("verifier_result"))
                for report in sequence(verified.get("claim_reports")):
                    report = mapping(report)
                    shown("Corrected claim", f"[{text(report.get('claim_id'))}] {text(report.get('claim_text'))}")
                    shown("Independent verdict", enum(report.get("verdict")))
                    claim_retrieval_details(report, components(run, call))
                    for ev in sequence(report.get("evidence")):
                        ev = mapping(ev)
                        shown("Re-verification evidence", ev.get("snippet"))
                        shown("Re-verification source", ev.get("url"))
        elif label == "MEMORY":
            memory, stores = latest_payload(calls[-1], "memory_result"), latest_payload(calls[-1], "memory")
            shown("Persistence status", memory.get("status"))
            count = first(memory, "stored_count", "count")
            shown("New verified facts stored", count if count != 0 else "None (no new writes)")
            duplicates = memory.get("duplicate_count")
            if duplicates:
                shown("Existing facts reused", duplicates)
            if memory.get("failed_count"):
                shown("Failed fact writes", memory["failed_count"])
            if memory.get("fact_ids"):
                shown("Stored/referenced fact IDs", memory["fact_ids"])
            shown("Knowledge graph", "Active" if stores.get("knowledge_graph") is True else "Inactive" if stores.get("knowledge_graph") is False else None)
            shown("Vector memory", "Active" if stores.get("vector_memory") is True else "Inactive" if stores.get("vector_memory") is False else None)
            shown("Reason", first(memory, "reason", "skipped_reason"))

    lines.extend(("", BAR, "  SUPERVISOR SUMMARY (actual production routing)", BAR))
    route = [e.get("name") for e in run["events"] if e.get("kind") == "node"]
    shown("Route taken", " → ".join(route) if route else "No graph stages were observed.")
    shown("Terminal status", result.get("terminal_status"))
    shown("Verification status", result.get("verification_status"))
    score("Total pipeline time (s)", run.get("seconds"))
    summary = validation.counts(run)
    shown("Agents fired", summary["FIRED"])
    shown("Agents not required", summary["NOT_REQUIRED"])
    shown("Failed agents", summary["FAILED"] if summary["FAILED"] else "None")
    if summary["NOT_FIRED"]:
        shown("Unreached agents", summary["NOT_FIRED"])
    if result.get("errors"):
        lines.append("  Pipeline errors occurred; full classified diagnostics are saved with this run.")
    if run.get("runner_status", "completed") != "completed":
        shown("Runner failure", run.get("runner_status"))
    lines.extend(("", "  • FINAL ANSWER DELIVERED TO USER:", "      " + text(result.get("final_response"))))
    if report_dir:
        shown("Saved walkthrough", str(Path(report_dir) / "WALKTHROUGH.txt"))
    lines.append(BAR)
    return "\n".join(lines) + "\n"


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_text", nargs="?", help="Your own query; real Base LLM generates its initial answer.")
    parser.add_argument("--query", dest="query_option")
    parser.add_argument("--draft", help="Explicit supplied-draft diagnostic; skips initial generation only.")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--checkpoint", type=Path, help="Existing production checkpoint; never written.")
    parser.add_argument("--output-dir", type=Path, help="Fresh directory; defaults to a timestamped Downloads folder.")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--from-json", type=Path, help="Replay a recorded execution JSON; no LLM/retrieval/model calls.")
    parser.add_argument("--details", action="store_true", help="Show full diagnostics; default is the narrated, readable walkthrough.")
    args = parser.parse_args(argv)
    if args.output_dir:
        args.output_dir = args.output_dir.resolve()
    if args.query_text and args.query_option:
        parser.error("use the positional query OR --query, not both")
    args.query = args.query_text or args.query_option
    if args.from_json and (args.query or args.draft is not None):
        parser.error("--from-json cannot be combined with a query or --draft")
    if not args.from_json and (not args.query or not args.query.strip()):
        parser.error("supply your query or --from-json")
    if args.draft is not None and not args.draft.strip():
        parser.error("--draft must not be empty")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("timeout must be positive and finite")
    if args.output_dir and args.output_dir.exists():
        parser.error("output directory already exists; use a fresh directory")
    secrets = lambda: tuple(v for k, v in os.environ.items() if len(v) >= 8 and re.search(r"KEY|TOKEN|PASSWORD|SECRET|AUTH", k, re.I))
    if args.from_json:
        try:
            record = json.loads(args.from_json.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or not isinstance(record.get("events"), list) or not isinstance(record.get("result"), dict):
                raise ValueError("not an execution record")
            if any(not isinstance(e, dict) or not all(k in e for k in ("kind", "name", "status"))
                   for e in record["events"]):
                raise ValueError("invalid observation events")
        except (OSError, ValueError) as exc:
            parser.error("cannot read execution record: " + type(exc).__name__)
        formatter = format_walkthrough if args.details else format_readable_walkthrough
        rendered = "REPLAY OF RECORDED EXECUTION — no new live calls.\n\n" + formatter(validation.scrub(record, secrets()), args.output_dir)
    else:
        if args.checkpoint and not args.checkpoint.is_dir():
            parser.error("checkpoint directory does not exist")
        from dotenv import load_dotenv
        load_dotenv(args.env_file)
        if args.checkpoint:
            os.environ["HALLUCIGUARD_DETECTOR_MODEL"] = str(args.checkpoint.resolve())
        if args.output_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            args.output_dir = Path.home() / "Downloads" / f"HalluciGuard-walkthrough-{stamp}-{uuid.uuid4().hex[:6]}"
        args.output_dir = args.output_dir.resolve()
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "raw").mkdir()
        validation.isolate_runtime(args.output_dir / "runtime")
        print(f"Running the real production graph for: {args.query}\n", flush=True)
        print("Conditional agents run only when required. Please wait for retrieval/model execution.\n", flush=True)
        record = asyncio.run(validation.execute_case(args, args.query, args.draft, "QUERY_RUN", secrets()))
        validation.export_reports(args.output_dir, [record])
        rendered = (format_walkthrough if args.details else format_readable_walkthrough)(record, args.output_dir)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "WALKTHROUGH.txt").open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return 0 if record.get("runner_status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
