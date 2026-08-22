"""
HalluciGuard CLI Tool — Verify any custom claim or question.

Usage:
  python scripts/verify_claim.py "Guido van Rossum created Python in 1991."
  python scripts/verify_claim.py --force-tavily "The Earth is flat."
  python scripts/verify_claim.py --retrieval-mode primary_only "Paris is the capital of France."
  python scripts/verify_claim.py -i  # Interactive mode
"""
from __future__ import annotations

import os
import sys
import asyncio
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VERIFIER_DIR = os.path.join(PROJECT_ROOT, "agents", "verifier_agent")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if VERIFIER_DIR not in sys.path:
    sys.path.insert(0, VERIFIER_DIR)

# Load environment variables (.env)
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _print_retrieval_trace(trace: dict) -> None:
    """Pretty-print the retrieval trace for full observability."""
    if not trace:
        print("\n  [Retrieval Trace] Not available (adapter does not support tracing)")
        return

    print("\n" + "-" * 72)
    print("  RETRIEVAL TRACE (V1.1 Quality Gate Observability)")
    print("-" * 72)

    print(f"  Domain          : {trace.get('requested_domain', '?')}")
    print(f"  Primary Adapter : {trace.get('primary_adapter', '?')}")
    print(f"  Retrieval Mode  : {trace.get('retrieval_mode', '?')}")

    primary = trace.get("primary", {})
    print(f"\n  --- PRIMARY ADAPTER ---")
    print(f"  Called           : {primary.get('called', False)}")
    if primary.get("called"):
        print(f"  Result Count     : {primary.get('result_count', 0)}")
        print(f"  Usable Count     : {primary.get('usable_count', 0)}")
        print(f"  Relevant Count   : {primary.get('relevant_count', 0)}")
        print(f"  Top Relevance    : {primary.get('top_relevance', 0.0):.4f}")
        print(f"  Source Diversity  : {primary.get('source_diversity', 0)}")
        print(f"  Sufficient       : {primary.get('sufficient', False)}")
        print(f"  Latency          : {primary.get('latency_ms', 0)} ms")
    print(f"  Reason           : {primary.get('reason', 'N/A')}")
    if primary.get("error"):
        print(f"  Error            : {primary['error']}")

    tavily = trace.get("tavily", {})
    print(f"\n  --- TAVILY FALLBACK ---")
    print(f"  Called           : {tavily.get('called', False)}")
    print(f"  Reason           : {tavily.get('reason', 'N/A')}")
    if tavily.get("called"):
        print(f"  Query            : {tavily.get('query', '')[:100]}")
        print(f"  Result Count     : {tavily.get('result_count', 0)}")
        print(f"  Usable Count     : {tavily.get('usable_count', 0)}")
        print(f"  Latency          : {tavily.get('latency_ms', 0)} ms")
    if tavily.get("error"):
        print(f"  Error            : {tavily['error']}")

    merged = trace.get("merged", {})
    print(f"\n  --- MERGE/DEDUP ---")
    print(f"  Candidates       : {merged.get('candidate_count', 0)}")
    print(f"  After Dedup      : {merged.get('deduplicated_count', 0)}")

    audit = trace.get("gate_relevance_audit")
    if audit:
        print(f"\n  --- GATE RELEVANCE AUDIT ---")
        print(f"  Source Confidence Hint     : {audit.get('source_confidence_hint', 0.0):.4f}")
        print(f"  Gate-Time Signal (Overlap) : {audit.get('gate_time_relevance_signal', 0.0):.4f}")
        print(f"  Final BGE Relevance Score  : {audit.get('final_bge_relevance_score', 0.0):.4f}")
        print(f"  Signals Agree              : {audit.get('signals_agree', False)}")

    rel = trace.get("relation_check")
    if rel:
        print(f"\n  --- RELATION VERIFICATION LAYER ---")
        print(f"  Claim Triple       : ({rel.get('claim_subject', '')}, {rel.get('claim_relation', '')}, {rel.get('claim_object', '')})")
        print(f"  Evidence Triple    : ({rel.get('evidence_subject', '')}, {rel.get('evidence_relation', '')}, {rel.get('evidence_object', '')})")
        print(f"  Relation Result    : {rel.get('check_result', 'N/A')}")
        print(f"  Combination Rule   : {rel.get('combination_rule_applied', 'N/A')}")


async def verify_custom_claim(
    claim_text: str,
    domain: str = "general",
    retrieval_mode: str = "hybrid",
    source_mode: Optional[str] = None,
    show_trace: bool = True,
) -> None:
    print("\n" + "=" * 72)
    print("  HALLUCIGUARD VERIFIER \u2014 CLAIM VERIFICATION TRACE")
    print(f"  Claim           : '{claim_text}'")
    print(f"  Domain          : {domain}")
    print(f"  Retrieval Mode  : {retrieval_mode}")
    if source_mode:
        print(f"  Source Mode     : {source_mode}")
    print("=" * 72)
    print("[1/3] Initializing pipeline & loading ML models (DeBERTa NLI + BGE Reranker)...")

    from api.pipeline import VerificationPipeline
    from schemas.models import SuspiciousClaim, VerifierInputV2

    pipeline = VerificationPipeline()
    payload = VerifierInputV2(
        query_id="custom-cli-verification",
        domain=domain,
        suspicious_claims=[SuspiciousClaim(claim_id="c1", text=claim_text)],
        retrieval_mode=retrieval_mode,
        source_mode=source_mode,
    )

    print("[2/3] Retrieving evidence (Authoritative Adapters + Tavily Web)...")
    print("[3/3] Running Cross-Encoder Reranking & DeBERTa NLI inference...")
    result = await pipeline.verify(payload)

    report = result.claim_evidence[0] if result.claim_evidence else None
    if not report:
        print("\n[ERROR] No verification report returned.")
        return

    verdict_str = report.verdict.value.upper()
    verdict_badge = {
        "VERIFIED": "[VERIFIED - SUPPORTED BY EVIDENCE]",
        "CONTRADICTED": "[CONTRADICTED - REFUTED BY EVIDENCE]",
        "UNVERIFIED": "[UNVERIFIED - INSUFFICIENT EVIDENCE]",
        "CONFLICTED": "[CONFLICTED - CONTRADICTORY SOURCES]",
    }.get(verdict_str, f"[{verdict_str}]")

    print("\n" + "=" * 72)
    print(f"  FINAL VERDICT : {verdict_badge}")
    print(f"  Trust Score   : {report.trust_score * 100:.1f}%")
    print(f"  Confidence    : {report.confidence_score * 100:.1f}%")
    print(f"  Support Score : {report.support_score * 100:.1f}%")
    print(f"  Contradict    : {report.contradiction_score * 100:.1f}%")
    print(f"  Retrieved     : {result.retrieved_sources} passages ({report.verified_evidence} verified items)")
    print("=" * 72)

    print(f"\n[Explanation]\n  {report.explanation}")

    if report.evidence:
        print("\n" + "-" * 72)
        print("  DECISION-GRADE EVIDENCE CITATIONS")
        print("-" * 72)
        for i, ev in enumerate(report.evidence, 1):
            lbl = ev.entailment_label.value.upper()
            tag = "[SUPPORTING]" if "ENTAIL" in lbl else ("[CONTRADICTING]" if "CONTRADICT" in lbl else "[NEUTRAL]")
            print(f"\n[{i}] {tag} {ev.title}")
            print(f"    Source      : {ev.source}")
            print(f"    URL         : {ev.url}")
            print(f"    NLI Match   : {ev.entailment_score * 100:.1f}% ({lbl})")
            print(f"    Credibility : {ev.credibility_score * 100:.1f}%")
            print(f'    Snippet     : "{ev.snippet[:200]}..."' if len(ev.snippet) > 200 else f'    Snippet     : "{ev.snippet}"')
    else:
        print("\n  No evidence passed the decision-grade relevance and NLI thresholds.")

    # Retrieval trace (V1.1)
    if show_trace:
        _print_retrieval_trace(report.retrieval_trace)

    print("\n" + "=" * 72 + "\n")


def interactive_mode(default_domain: str = "general", retrieval_mode: str = "hybrid") -> None:
    print("\n" + "=" * 72)
    print("  HALLUCIGUARD INTERACTIVE VERIFICATION CONSOLE")
    print("  Type any claim to verify against real web/authoritative sources.")
    print("  Type 'domain <name>' to change domain (general, healthcare, cybersecurity, ai_research).")
    print("  Type 'mode <name>' to change retrieval mode (hybrid, primary_only, tavily_only).")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 72 + "\n")

    current_domain = default_domain
    current_mode = retrieval_mode

    while True:
        try:
            claim = input(f"[{current_domain}|{current_mode}] Enter claim > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not claim:
            continue
        if claim.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if claim.lower().startswith("domain "):
            new_domain = claim.split(" ", 1)[1].strip()
            if new_domain:
                current_domain = new_domain
                print(f"Switched domain to: {current_domain}\n")
            continue
        if claim.lower().startswith("mode "):
            new_mode = claim.split(" ", 1)[1].strip()
            if new_mode in ("hybrid", "primary_only", "tavily_only"):
                current_mode = new_mode
                print(f"Switched retrieval mode to: {current_mode}\n")
            else:
                print(f"Invalid mode '{new_mode}'. Use: hybrid, primary_only, tavily_only\n")
            continue

        asyncio.run(verify_custom_claim(claim, current_domain, current_mode))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify any custom claim using HalluciGuard Verifier.")
    parser.add_argument(
        "claim",
        nargs="?",
        default=None,
        help="The claim text you want to verify. Omit to start interactive mode.",
    )
    parser.add_argument(
        "--domain",
        default="general",
        help="Domain intelligence profile (general, healthcare, legal, finance, cybersecurity, ai_research).",
    )
    parser.add_argument(
        "--retrieval-mode",
        default="hybrid",
        choices=["hybrid", "primary_only", "tavily_only"],
        help="Retrieval mode: hybrid (default), primary_only (no Tavily), tavily_only (skip primary).",
    )
    parser.add_argument(
        "--source-mode",
        default=None,
        help="Granular adapter source mode (e.g. healthcare-pubmed, healthcare-fda, healthcare-pmc, healthcare-who, healthcare-clinicaltrials).",
    )
    parser.add_argument(
        "--force-tavily",
        action="store_true",
        help="Force Tavily-only mode (equivalent to --retrieval-mode tavily_only).",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Suppress retrieval trace output.",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Launch interactive verification REPL console.",
    )
    args = parser.parse_args()

    # --force-tavily overrides --retrieval-mode
    retrieval_mode = args.retrieval_mode
    if args.force_tavily:
        retrieval_mode = "tavily_only"

    if args.interactive or not args.claim:
        interactive_mode(args.domain, retrieval_mode)
    else:
        asyncio.run(verify_custom_claim(
            args.claim,
            args.domain,
            retrieval_mode,
            source_mode=args.source_mode,
            show_trace=not args.no_trace,
        ))


if __name__ == "__main__":
    main()
