"""Human-readable demonstration of the real HalluciGuard backend pipeline.

Examples:
    python demo_7_agents.py "What is the capital of India?"
    python demo_7_agents.py --interactive
    python demo_7_agents.py --demo 1
    python demo_7_agents.py --demo all
    python demo_7_agents.py "Who founded Microsoft?" --raw

Only a user query enters the production graph. This script does not accept a
draft, fabricate evidence, or reconstruct agent decisions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

# Windows PowerShell may expose a legacy cp1252 stdout even when model output
# contains Unicode punctuation. Keep the demo readable instead of crashing
# after an otherwise successful pipeline run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from orchestration.graph import run_verification


DEMO_QUERIES = {
    "1": ("What is the capital of India?", "normal", "TRUE FACT / GEOGRAPHY"),
    "2": (
        "For a hallucination-safety test, give a one-sentence answer to this premise: Snehith founded Microsoft.",
        "stress_test",
        "ADVERSARIAL FALSE PREMISE",
    ),
    "3": ("Who created the Java programming language?", "normal", "TECHNOLOGY FACT"),
}

WIDTH = 78
NOT_EXPOSED = "Not exposed by current agent contract"


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _enum(value: Any) -> str:
    text = str(value or "Not provided")
    return text.rsplit(".", 1)[-1].upper()


def _section(number: str, title: str) -> None:
    print("\n" + "=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def _sub(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def _field(label: str, value: Any) -> None:
    if value in (None, "", [], {}):
        value = "Not provided"
    print(f"{label + ':':<30} {value}")


def _quote(text: Any, indent: str = "") -> None:
    value = str(text or "Not provided")
    wrapped = textwrap.fill(value, width=WIDTH - len(indent) - 4)
    for line in wrapped.splitlines():
        print(f'{indent}"{line}"')


def _trace_statuses(result: dict[str, Any]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for event in result.get("trace", []):
        if isinstance(event, dict) and event.get("node"):
            statuses[str(event["node"])] = str(event.get("status", "reported")).upper()
    return statuses


def _outcome(result: dict[str, Any], name: str) -> dict[str, Any]:
    return _dict(_dict(result.get("agent_outcomes")).get(name))


def _display_base_llm(query: str, result: dict[str, Any]) -> None:
    base = _dict(result.get("base_llm"))
    draft = result.get("draft_response") or result.get("llm_response") or ""
    _section("1", "BASE LLM — RESPONSE GENERATION")
    _sub("INPUT")
    _field("User query", query)
    _sub("MODEL")
    _field("Provider", base.get("provider") or base.get("model_provider") or NOT_EXPOSED)
    _field("Model", base.get("model") or base.get("model_name") or NOT_EXPOSED)
    _field("Latency", f"{base.get('latency_ms')} ms" if base.get("latency_ms") is not None else NOT_EXPOSED)
    _sub("GENERATED RESPONSE")
    _quote(draft)
    _sub("WHAT THIS AGENT DID")
    print("Generated the initial candidate answer from the user query; no factual trust is assigned until verification completes.")
    _sub("OUTPUT SENT TO NEXT STAGE")
    print("The complete generated response was sent to Claim Analyzer and Detector.")


def _display_claim_analyzer(result: dict[str, Any]) -> list[dict[str, Any]]:
    analysis = _dict(result.get("claim_analysis"))
    candidates = [c for c in analysis.get("candidates", []) if isinstance(c, dict)]
    factual = [c for c in candidates if c.get("is_factual") or c.get("claim_type") == "FACTUAL_CLAIM"]
    discarded = [c for c in candidates if c not in factual]
    _section("2", "CLAIM ANALYZER — FACTUAL CLAIM EXTRACTION")
    _sub("INPUT")
    print("The complete Base LLM response.")
    _sub("WHAT THIS AGENT DID")
    print("Separated independently checkable factual claims from non-factual text. It did not decide whether any claim was true.")
    _field("Analyzer source", analysis.get("source") or NOT_EXPOSED)
    _field("Detected domain", analysis.get("domain") or NOT_EXPOSED)
    _field("Factual claims", len(factual))
    _field("Discarded spans", len(discarded))
    if analysis.get("llm_error"):
        _field("Fallback reason", analysis["llm_error"])
    for index, claim in enumerate(factual, 1):
        _sub(f"CLAIM {index}")
        _field("Claim ID", claim.get("claim_id"))
        _field("Type", claim.get("claim_type"))
        _field("Domain", claim.get("domain") or analysis.get("domain"))
        _field("Entities", claim.get("entities") or NOT_EXPOSED)
        _field("Verification required", "YES")
        print("Text:")
        _quote(claim.get("claim_text") or claim.get("text"), "  ")
        print("Search / retrieval queries:")
        queries = claim.get("search_queries") or []
        if queries:
            for query in queries:
                print(f"  - {query}")
        else:
            print(f"  - {NOT_EXPOSED}")
    _sub("CLAIMS DISCARDED AS NON-FACTUAL")
    if discarded:
        for item in discarded:
            print(f"- {item.get('claim_text') or item.get('text')} [{item.get('claim_type')}] — {item.get('reason') or 'reason not provided'}")
    else:
        print("None.")
    _sub("OUTPUT SENT FOR VERIFICATION")
    print(f"{len(factual)} factual claim(s), with preserved IDs and available retrieval queries.")
    return factual


def _display_detector(result: dict[str, Any]) -> None:
    detector = _dict(result.get("detector_result") or result.get("detector"))
    _section("3", "DETECTOR AGENT — HALLUCINATION RISK TRIAGE")
    _sub("INPUT ANALYZED")
    _quote(result.get("draft_response") or result.get("llm_response"))
    _sub("MODEL INFORMATION")
    for label, key in (
        ("Model source", "model_source"),
        ("Model version", "model_version"),
        ("Model loaded", "model_loaded"),
        ("Inference executed", "detector_inference_executed"),
        ("Calibration applied", "calibration_applied"),
        ("Calibrator version", "calibrator_version"),
    ):
        _field(label, detector.get(key) if key in detector else NOT_EXPOSED)
    _sub("DETECTOR RESULT")
    probability_available = detector.get("probability_available", detector.get("hallucination_probability") is not None)
    probability = detector.get("hallucination_probability") if probability_available else "N/A — evidence unavailable during triage"
    confidence = detector.get("confidence_score") if probability_available else "N/A — fail-closed routing"
    _field("Hallucination probability", probability)
    _field("Confidence score", confidence)
    _field("Risk level", _enum(detector.get("risk_level")))
    _field("Recommended action", _enum(detector.get("next_action")))
    _field("Degraded", detector.get("detector_degraded", False))
    _field("Sentence-level scores", detector.get("sentences") or NOT_EXPOSED)
    _field("Diagnostics", detector.get("diagnostics") or NOT_EXPOSED)
    _sub("HUMAN-READABLE INTERPRETATION")
    if probability_available:
        print(f"The Detector estimated a {float(probability) * 100:.2f}% risk and recommended {_enum(detector.get('next_action'))}. This is triage only; the Verifier remains the factual authority.")
    else:
        print("The Detector could not produce an evidence-grounded probability before retrieval, so it routed the answer to verification. It did not declare the answer true or false.")
    _sub("NEXT")
    print("The generated response and extracted factual claims continue to retrieval and Verifier.")


def _verifier_reports(result: dict[str, Any]) -> list[dict[str, Any]]:
    verifier = _dict(result.get("verifier_result"))
    legacy = _dict(result.get("verifier"))
    canonical_reports = [r for r in verifier.get("claim_reports", []) if isinstance(r, dict)]
    legacy_reports = [r for r in legacy.get("claim_evidence", []) if isinstance(r, dict)]
    if not canonical_reports:
        return legacy_reports

    # The canonical supervisor contract deliberately contains fewer diagnostic
    # fields than the verifier's native result. Merge both views by claim ID so
    # the demo can show real retrieval/NLI details without inventing them.
    legacy_by_id = {str(r.get("claim_id")): r for r in legacy_reports}
    merged_reports: list[dict[str, Any]] = []
    for canonical in canonical_reports:
        native = legacy_by_id.get(str(canonical.get("claim_id")), {})
        merged = {**native, **canonical}
        if native.get("evidence"):
            merged["evidence"] = native["evidence"]
        merged_reports.append(merged)
    return merged_reports


def _display_verifier(result: dict[str, Any], factual: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = _verifier_reports(result)
    _section("4", "VERIFIER AGENT — RETRIEVAL AND FACTUAL VERIFICATION")
    _sub("INPUT RECEIVED")
    print(f"{len(factual)} factual claim(s) from Claim Analyzer.")
    for claim in factual:
        print(f"- {claim.get('claim_id')}: {claim.get('claim_text') or claim.get('text')}")
    for index, report in enumerate(reports, 1):
        _sub(f"CLAIM {index} — {_enum(report.get('verdict'))}")
        claim_text = report.get("claim_text") or report.get("claim") or "Not provided"
        _field("Claim ID", report.get("claim_id"))
        _field("Original claim", claim_text)
        _field("Normalized claim", report.get("normalized_claim") or NOT_EXPOSED)
        _field("Entities", report.get("entities") or NOT_EXPOSED)
        _field("Domain", _dict(result.get("verifier_result")).get("domain") or result.get("domain") or NOT_EXPOSED)
        matching = next((c for c in factual if c.get("claim_id") == report.get("claim_id")), {})
        _field("Search queries", matching.get("search_queries") or NOT_EXPOSED)
        _field("Documents retrieved", report.get("retrieved_documents") if report.get("retrieved_documents") is not None else NOT_EXPOSED)
        _field("Documents reranked", report.get("reranked_documents") if report.get("reranked_documents") is not None else NOT_EXPOSED)
        evidence = [e for e in report.get("evidence", []) if isinstance(e, dict)]
        providers = sorted({str(e.get("source")) for e in evidence if e.get("source")})
        _field("Providers in evidence", providers or NOT_EXPOSED)
        _sub("EVIDENCE RETRIEVED")
        if not evidence:
            print("No decision-grade evidence was returned for this claim.")
        for ev_index, evidence_item in enumerate(evidence, 1):
            print(f"\nEvidence {ev_index}")
            _field("Source / provider", evidence_item.get("source") or NOT_EXPOSED)
            _field("Title", evidence_item.get("title") or NOT_EXPOSED)
            _field("URL", evidence_item.get("url") or NOT_EXPOSED)
            _field("Relevance score", evidence_item.get("bge_score") if evidence_item.get("bge_score") is not None else NOT_EXPOSED)
            _field("NLI classification", evidence_item.get("classification") or _enum(evidence_item.get("entailment_label")))
            _field("Entailment", evidence_item.get("nli_entailment", evidence_item.get("entailment_score", NOT_EXPOSED)))
            _field("Contradiction", evidence_item.get("nli_contradiction", NOT_EXPOSED))
            _field("Neutral / NEI", evidence_item.get("nli_neutral", NOT_EXPOSED))
            print("Evidence text:")
            _quote(evidence_item.get("snippet"), "  ")
            print("Why this evidence matters:")
            classification = str(evidence_item.get("classification") or evidence_item.get("entailment_label") or "neutral").lower()
            if "support" in classification or "entail" in classification:
                print("  The evidence supports the factual relationship asserted by the claim.")
            elif "contradict" in classification:
                print("  The evidence conflicts with the factual relationship asserted by the claim.")
            else:
                print("  The evidence is related but does not decisively prove or refute the claim.")
        _sub("VERIFICATION DECISION")
        _field("Verdict", _enum(report.get("verdict")))
        _field("Support score", report.get("support_score"))
        _field("Contradiction score", report.get("contradiction_score"))
        _field("Confidence", report.get("confidence_score"))
        _field("Reason", report.get("explanation") or NOT_EXPOSED)
    _sub("WHAT VERIFIER SENDS FORWARD")
    print("Per-claim verdicts, scores, and decision-grade evidence are sent to Judge.")
    return reports


def _display_judge(result: dict[str, Any], reports: list[dict[str, Any]]) -> None:
    judge = _dict(result.get("judge_result") or result.get("judge"))
    counts = Counter(_enum(r.get("verdict")) for r in reports)
    _section("5", "JUDGE AGENT — WORKFLOW DECISION")
    _sub("INPUTS RECEIVED")
    _field("Claims evaluated", len(reports))
    for label in ("VERIFIED", "CONTRADICTED", "UNVERIFIED", "CONFLICTED"):
        _field(label.title(), counts[label])
    _field("Detector risk", _enum(_dict(result.get("detector_result") or result.get("detector")).get("risk_level")))
    _field("Reverification state", _dict(result.get("reverification_result")) or "Not applicable yet")
    _sub("JUDGE DECISION")
    _field("Decision", _enum(judge.get("decision")))
    _field("Severity", _enum(judge.get("severity")))
    _field("Confidence", judge.get("confidence"))
    _field("Reason", judge.get("reason") or judge.get("explanation") or NOT_EXPOSED)
    _field("Decision basis", judge.get("decision_basis") or NOT_EXPOSED)
    _field("Correction requested", "YES" if judge.get("correction_request") else "NO")
    _sub("WHAT THIS MEANS")
    print(judge.get("explanation") or judge.get("reason") or "Judge produced a fail-closed workflow decision from the available evidence.")


def _display_corrector(result: dict[str, Any]) -> None:
    correction = _dict(result.get("correction_result") or result.get("corrector"))
    outcome = _outcome(result, "corrector")
    _section("6", "CORRECTOR AGENT — GROUNDED FACT REPAIR")
    if not correction:
        _field("Status", "NOT_REQUIRED")
        _field("Why", outcome.get("reason") or "Judge did not request a correction.")
        print("The Corrector did not execute and no repair is being claimed.")
        return
    _sub("INPUT")
    _field("Original response", correction.get("original_text") or result.get("draft_response"))
    _field("Correction instruction", correction.get("reasoning") or NOT_EXPOSED)
    _field("Provider", correction.get("provider_used") or NOT_EXPOSED)
    _field("Model", correction.get("model_used") or NOT_EXPOSED)
    _field("Attempts", correction.get("attempt_count"))
    _sub("CORRECTED OUTPUT")
    _quote(correction.get("corrected_text"))
    _sub("WHAT CHANGED")
    _field("Changed claims", correction.get("changed_claims") or NOT_EXPOSED)
    _field("Validation status", _enum(correction.get("validation_status")))
    _field("Status", _enum(correction.get("status")))
    if _enum(correction.get("status")) in {"FAILED", "FALLBACK"}:
        _field("Failure category", correction.get("failure_category") or NOT_EXPOSED)
        print("Fail-closed action: the unchanged or failed repair is not presented as successful.")


def _display_reverifier(result: dict[str, Any]) -> None:
    reverification = _dict(result.get("reverification_result"))
    outcome = _outcome(result, "reverifier")
    _section("7", "REVERIFIER — POST-CORRECTION VALIDATION")
    if not reverification:
        _field("Status", "NOT_REQUIRED")
        _field("Why", outcome.get("reason") or "No correction was performed.")
        print("The ReVerifier did not execute and no post-correction pass is being claimed.")
        return
    _field("Passed", "YES" if reverification.get("passed") else "NO")
    _field("Status", _enum(reverification.get("status")))
    _field("Remaining contradictions", reverification.get("remaining_contradictions"))
    _field("Failure category", reverification.get("failure_category") or "None")
    _field("Off-topic check", "Passed" if reverification.get("passed") else "See failure category")
    print("The ReVerifier independently checked the corrected response instead of trusting the Corrector.")


def _display_memory(result: dict[str, Any], reports: list[dict[str, Any]]) -> None:
    # Combine the native persistence diagnostics with the canonical status.
    # Canonical values win when both contracts expose the same field.
    memory = {**_dict(result.get("memory")), **_dict(result.get("memory_result"))}
    eligible = [r.get("claim_text") for r in reports if _enum(r.get("verdict")) == "VERIFIED"]
    _section("8", "MEMORY AGENT — KNOWLEDGE PERSISTENCE")
    _sub("INPUT FACTS")
    if eligible:
        for fact in eligible:
            print(f"- {fact}")
    else:
        print("No verified facts were eligible for persistence.")
    _sub("MEMORY ACTION")
    _field("Persistence status", _enum(memory.get("status")))
    _field("Stored", memory.get("stored_count", memory.get("count", 0)))
    _field("Duplicates reused", memory.get("duplicate_count", 0))
    _field("Failed", memory.get("failed_count", 0))
    knowledge_graph = memory.get("knowledge_graph")
    vector_memory = memory.get("vector_memory")
    _field("Knowledge graph", NOT_EXPOSED if knowledge_graph is None else ("ACTIVE" if knowledge_graph else "INACTIVE"))
    _field("Vector memory", NOT_EXPOSED if vector_memory is None else ("ACTIVE" if vector_memory else "INACTIVE"))
    _field("Stored fact IDs", memory.get("fact_ids") or NOT_EXPOSED)
    _field("Reason", memory.get("reason") or memory.get("skipped_reason") or "Persistence completed")


def _display_final(query: str, result: dict[str, Any], reports: list[dict[str, Any]]) -> None:
    counts = Counter(_enum(r.get("verdict")) for r in reports)
    detector = _dict(result.get("detector_result") or result.get("detector"))
    judge = _dict(result.get("judge_result") or result.get("judge"))
    memory = _dict(result.get("memory_result") or result.get("memory"))
    _section("FINAL", "FINAL PIPELINE RESULT")
    _field("Original user query", query)
    _field("Detector risk", _enum(detector.get("risk_level")))
    _field("Claims", len(reports))
    _field("Verified", counts["VERIFIED"])
    _field("Contradicted", counts["CONTRADICTED"])
    _field("Unverified", counts["UNVERIFIED"])
    _field("Conflicted", counts["CONFLICTED"])
    _field("Judge", _enum(judge.get("decision")))
    _field("Corrector", "EXECUTED" if result.get("correction_result") else "NOT_REQUIRED")
    if result.get("reverification_result"):
        _field("ReVerifier", "PASSED" if _dict(result["reverification_result"]).get("passed") else "FAILED")
    else:
        _field("ReVerifier", "NOT_REQUIRED")
    _field("Memory", _enum(memory.get("status")))
    _field("Terminal status", result.get("terminal_status"))
    _field("Verification status", result.get("verification_status"))
    _sub("FINAL RESPONSE TO USER")
    _quote(result.get("final_response"))
    _sub("COMPLETE PIPELINE TRACE")
    statuses = _trace_statuses(result)
    labels = [
        ("base_llm", "Base LLM"),
        ("claim_analyzer", "Claim Analyzer"),
        ("detector", "Detector"),
        ("verifier", "Retrieval + Verifier"),
        ("judge", "Judge"),
        ("corrector", "Corrector"),
        ("reverifier", "ReVerifier"),
        ("memory", "Memory"),
    ]
    for key, label in labels:
        outcome = _outcome(result, key)
        status = statuses.get(key) or str(outcome.get("status") or "NOT_REQUIRED").upper()
        reason = outcome.get("reason")
        print(f"{label:<23} {status}" + (f" — {reason}" if reason else ""))


def _display_failures(result: dict[str, Any]) -> None:
    errors = [e for e in result.get("errors", []) if isinstance(e, dict)]
    if not errors:
        return
    _section("!", "AGENT FAILURE")
    for error in errors:
        _field("Agent", error.get("node") or error.get("agent"))
        _field("Error", error.get("message") or error.get("error"))
        _field("Failure category", error.get("error_type") or error.get("type") or NOT_EXPOSED)
        _field("Fail-closed action", result.get("terminal_status") or "human review")


def display(query: str, result: dict[str, Any], *, raw: bool = False, title: str = "") -> None:
    print("\n" + "=" * WIDTH)
    print(" " * 20 + "HALLUCIGUARD 7-AGENT DEMO")
    print("=" * WIDTH)
    if title:
        _field("Scenario", title)
    _field("User query", query)
    print("Pipeline: Base LLM → Claim Analyzer → Detector → Retrieval/Verifier →")
    print("          Judge → Corrector → ReVerifier → Memory → Final Answer")
    _display_base_llm(query, result)
    factual = _display_claim_analyzer(result)
    _display_detector(result)
    reports = _display_verifier(result, factual)
    _display_judge(result, reports)
    _display_corrector(result)
    _display_reverifier(result)
    _display_memory(result, reports)
    _display_failures(result)
    _display_final(query, result, reports)
    if raw:
        _section("RAW", "STRUCTURED PIPELINE STATE")
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))


async def run_one(query: str, mode: str, *, raw: bool, title: str = "") -> dict[str, Any]:
    try:
        result = await run_verification(
            user_query=query,
            llm_response="",
            domain="general",
            generation_mode=mode,
        )
    except Exception as exc:
        print("\nAGENT FAILURE\n-------------")
        _field("Agent", "Pipeline entry point")
        _field("Error", f"{type(exc).__name__}: {exc}")
        _field("Fail-closed action", "No answer was accepted or persisted")
        return {}
    display(query, result, raw=raw, title=title)
    return result


async def async_main(args: argparse.Namespace) -> int:
    if args.interactive:
        query = input("Enter one user query: ").strip()
        if not query:
            print("A non-empty query is required.")
            return 2
        await run_one(query, "normal", raw=args.raw)
        return 0

    if args.demo:
        demo_ids = list(DEMO_QUERIES) if args.demo == "all" else [args.demo]
        for demo_id in demo_ids:
            query, mode, title = DEMO_QUERIES[demo_id]
            await run_one(query, mode, raw=args.raw, title=f"DEMO {demo_id}: {title}")
        return 0

    query = " ".join(args.query).strip()
    if not query:
        print('Usage: python demo_7_agents.py "Your question"')
        print("       python demo_7_agents.py --interactive")
        print("       python demo_7_agents.py --demo 1")
        return 2
    await run_one(query, "normal", raw=args.raw)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain one real HalluciGuard backend execution.")
    parser.add_argument("query", nargs="*", help="user query; no draft is accepted")
    parser.add_argument("--interactive", action="store_true", help="prompt for exactly one query")
    parser.add_argument("--demo", choices=["1", "2", "3", "all"], help="run one built-in query-only scenario")
    parser.add_argument("--raw", action="store_true", help="append the raw structured pipeline state")
    args = parser.parse_args()
    if args.interactive and (args.query or args.demo):
        parser.error("--interactive cannot be combined with a query or --demo")
    if args.demo and args.query:
        parser.error("--demo cannot be combined with a custom query")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
