"""
HalluciGuard — Vertical Slice Integration Test.
Chain: Base LLM -> Detector -> n8n Retrieval V2 -> BGE Reranker -> DeBERTa NLI -> EvidenceScorer -> Final Verdict.

Usage:
  python scripts/test_llm_detector_verifier_slice.py
  python scripts/test_llm_detector_verifier_slice.py --query "Who is the father of Allu Arjun?"
  python scripts/test_llm_detector_verifier_slice.py --query "Allu Arjun's father is Chiranjeevi." --force-verifier
  python scripts/test_llm_detector_verifier_slice.py --all-tests
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from services.llm_detector_verifier_service import BaseLLMDetectorVerifierService


def print_slice_execution_report(result_dict: Dict[str, Any], domain: str = "general") -> None:
    """Print the required transparent diagnostic log for the full vertical slice."""
    query = result_dict.get("user_query", "")
    draft = result_dict.get("draft_response", "")
    detector = result_dict.get("detector") or {}
    verifier = result_dict.get("verifier") or {}

    print("\n" + "=" * 76)
    print("  HALLUCIGUARD — FULL VERTICAL SLICE EXECUTION TRACE")
    print("=" * 76)
    print(f"Original Question / Claim : \"{query}\"")
    print(f"Domain                    : {domain}")
    print(f"LLM Draft Response        : \"{draft.strip()}\"")
    print("-" * 76)

    # ── Detector ────────────────────────────────────────────────────────────
    print("DETECTOR:")
    print(f"  • Hallucination Prob    : {detector.get('hallucination_probability', 0.0):.4f}")
    print(f"  • Risk Tier             : {detector.get('risk_tier', 'UNKNOWN')}")
    print(f"  • Decision              : {detector.get('decision', 'UNKNOWN')}")
    print(f"  • Model Source          : {detector.get('model_source', 'HaluEval DistilBERT')}")
    # §6 execution proof — a failed detector load must never masquerade as real ML inference
    print(f"  • Model Loaded          : {detector.get('detector_model_loaded', False)}")
    print(f"  • Inference Executed    : {detector.get('detector_inference_executed', False)}")
    print(f"  • Degraded (baseline)   : {detector.get('detector_degraded', False)}")
    print(f"  • Detector Provenance   : {detector.get('detector_model_source', '') or 'n/a'}")
    print("-" * 76)

    # ── Verifier & n8n ──────────────────────────────────────────────────────
    if not verifier.get("executed", False):
        print(f"VERIFIER                  : SKIPPED ({verifier.get('reason', 'LOW Risk')})")
        print("=" * 76 + "\n")
        return

    claims = verifier.get("claim_evidence", [])
    print(f"VERIFIER (Executed: True | Total Claims Evaluated: {len(claims)}):")

    for c_idx, claim_rep in enumerate(claims, 1):
        claim_text = claim_rep.get("claim_text", query)
        trace = claim_rep.get("retrieval_trace") or {}
        n8n_trace = trace.get("n8n_trace") or {}

        print(f"\n  Claim #{c_idx}: \"{claim_text}\"")
        print("  N8N RETRIEVAL:")
        sources = n8n_trace.get("primary_sources") or trace.get("primary_sources") or ["wikipedia"]
        retrieved_count = n8n_trace.get("evidence_count", verifier.get("retrieved_sources", len(claim_rep.get("evidence", []))))
        tavily_used = n8n_trace.get("tavily_called", False)
        counts = n8n_trace.get("counts") or {}
        perf = n8n_trace.get("performance") or {}

        print(f"    • Sources Used        : {sources}")
        print(f"    • Retrieved Count     : {retrieved_count} candidate passages")
        print(f"    • Tavily Fallback     : {tavily_used}")
        if counts:
            print(f"    • Counts Breakdown    : {counts}")
        if perf:
            print(f"    • Performance Metrics : {perf}")

        # §13/§15/§26 model-execution proof — distinguishes REAL inference from
        # a fail-soft fallback. In certification mode these gate the verdict.
        rr_exec = trace.get("reranker_execution") or {}
        nli_exec = trace.get("nli_execution") or {}
        if rr_exec or nli_exec:
            print("  MODEL EXECUTION PROOF:")
            if rr_exec:
                print(
                    f"    • BGE   : status={rr_exec.get('status', 'n/a')} "
                    f"loaded={rr_exec.get('loaded', False)} "
                    f"executed={rr_exec.get('inference_executed', False)} "
                    f"degraded={rr_exec.get('degraded', False)} "
                    f"device={rr_exec.get('device', '?')} "
                    f"scored={rr_exec.get('scored_count', 0)} "
                    f"{rr_exec.get('latency_ms', 0)}ms"
                )
            if nli_exec:
                print(
                    f"    • NLI   : status={nli_exec.get('status', 'n/a')} "
                    f"loaded={nli_exec.get('loaded', False)} "
                    f"executed={nli_exec.get('inference_executed', False)} "
                    f"degraded={nli_exec.get('degraded', False)} "
                    f"device={nli_exec.get('device', '?')} "
                    f"pairs={nli_exec.get('batch_size', 0)} "
                    f"{nli_exec.get('latency_ms', 0)}ms"
                )

        evidence_items = claim_rep.get("evidence", [])
        print(f"\n  EVIDENCE & NLI EVALUATION ({len(evidence_items)} decision-grade citations):")
        if not evidence_items:
            print("    (No candidate passages met the BGE relevance and NLI decision gate)")
        for e_idx, ev in enumerate(evidence_items, 1):
            lbl = ev.get("entailment_label", "neutral").upper()
            classification = ev.get("classification", lbl).upper()
            tag = f"[{classification}]"

            print(f"\n    Evidence #{e_idx} {tag} \"{ev.get('title')}\"")
            print(f"      • URL               : {ev.get('url') or 'N/A'}")
            print(f"      • Source            : {ev.get('source')}")
            print(f"      • adapter_score     : {ev.get('adapter_score', 0.0):.4f}")
            print(f"      • bge_score         : {ev.get('bge_score', 0.0):.4f}")
            print(f"      • NLI Label         : {lbl}")
            print(f"      • nli_entailment    : {ev.get('nli_entailment', 0.0):.4f}")
            print(f"      • nli_contradiction : {ev.get('nli_contradiction', 0.0):.4f}")
            print(f"      • nli_neutral       : {ev.get('nli_neutral', 0.0):.4f}")
            print(f"      • classification    : {classification}")
            snip = ev.get("snippet", "").replace("\n", " ").strip()
            print(f"      • Snippet           : \"{snip[:180]}...\"" if len(snip) > 180 else f"      • Snippet           : \"{snip}\"")

        print("\n  FINAL VERDICT:")
        v_str = claim_rep.get("verdict", "UNVERIFIED").upper()
        badge = {
            "VERIFIED": "[VERIFIED - SUPPORTED BY EVIDENCE]",
            "CONTRADICTED": "[CONTRADICTED - REFUTED BY EVIDENCE]",
            "UNVERIFIED": "[UNVERIFIED - INSUFFICIENT EVIDENCE]",
            "CONFLICTED": "[CONFLICTED - CONFLICTING SOURCES]",
        }.get(v_str, f"[{v_str}]")
        print(f"    • Verdict             : {badge}")
        print(f"    • Confidence Score    : {claim_rep.get('confidence_score', 0.0) * 100:.1f}%")
        print(f"    • Support Score       : {claim_rep.get('support_score', 0.0) * 100:.1f}%")
        print(f"    • Contradiction Score : {claim_rep.get('contradiction_score', 0.0) * 100:.1f}%")
        print(f"    • Trust Score         : {claim_rep.get('trust_score', 0.0) * 100:.1f}%")
        print(f"    • Explanation         : {claim_rep.get('explanation', '')}")

    print("=" * 76 + "\n")


async def run_single_test(
    title: str,
    query: str,
    domain: str = "general",
    force_verifier: bool = True,
) -> Dict[str, Any]:
    print(f"\n>>> Running: {title}")
    service = BaseLLMDetectorVerifierService()
    try:
        result = await service.execute_slice(
            user_query=query,
            domain=domain,
            force_verifier=force_verifier,
        )
    except Exception as exc:  # noqa: BLE001 - surface certification failures cleanly
        if type(exc).__name__ == "CertificationError":
            print("\n" + "=" * 76)
            print("  CERTIFICATION FAILURE (controlled, fail-closed)")
            print("=" * 76)
            print(f"  {exc}")
            print("  -> A required model did not run for real (or evidence was mock/empty).")
            print("  -> This is the intended certification outcome, NOT a crash.\n")
            return {"user_query": query, "certification_failed": True, "error": str(exc)}
        raise
    dumped = result.model_dump()
    print_slice_execution_report(dumped, domain=domain)
    return dumped


async def run_all_live_tests() -> None:
    print("\n" + "=" * 76)
    print("  RUNNING 8 TARGET VERTICAL SLICE LIVE TESTS (matrix A-H)")
    print("=" * 76)

    # TEST A: True Claim Question
    await run_single_test(
        title="TEST A — True Claim Question ('Who is the father of Allu Arjun?')",
        query="Who is the father of Allu Arjun?",
        domain="general",
        force_verifier=True,
    )

    # TEST B: False Relation Claim
    await run_single_test(
        title="TEST B — False Relation ('Allu Arjun's father is Chiranjeevi.')",
        query="Allu Arjun's father is Chiranjeevi.",
        domain="general",
        force_verifier=True,
    )

    # TEST C: General True Fact
    await run_single_test(
        title="TEST C — General True Fact ('Paris is the capital of France.')",
        query="Paris is the capital of France.",
        domain="general",
        force_verifier=True,
    )

    # TEST D: General False Fact
    await run_single_test(
        title="TEST D — General False Fact ('The Eiffel Tower is located in London.')",
        query="The Eiffel Tower is located in London.",
        domain="general",
        force_verifier=True,
    )

    # TEST E: Healthcare Domain
    await run_single_test(
        title="TEST E — Healthcare Claim ('Aspirin is used to relieve mild to moderate pain.')",
        query="Aspirin is used to relieve mild to moderate pain.",
        domain="healthcare",
        force_verifier=True,
    )

    # TEST F: Nonsense / unverifiable claim (expect UNVERIFIED or CONTRADICTED — never VERIFIED)
    await run_single_test(
        title="TEST F — Nonsense Claim ('The moon is made of green cheese and orbits Jupiter.')",
        query="The moon is made of green cheese and orbits Jupiter.",
        domain="general",
        force_verifier=True,
    )

    # TEST G: Time-sensitive claim (current office-holder — tests live retrieval recency)
    await run_single_test(
        title="TEST G — Time-Sensitive Claim ('Who is the current president of the United States?')",
        query="Who is the current president of the United States?",
        domain="general",
        force_verifier=True,
    )

    # TEST H: Multi-sentence answer — proves decomposition into per-claim verdicts
    await run_single_test(
        title="TEST H — Multi-Sentence Decomposition ('Tell me three facts about the Eiffel Tower.')",
        query="Tell me three facts about the Eiffel Tower.",
        domain="general",
        force_verifier=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HalluciGuard Vertical Slice Integration Test.")
    parser.add_argument("--query", default=None, help="Query or claim to run through the slice.")
    parser.add_argument("--domain", default="general", help="Domain (general, healthcare, etc.)")
    parser.add_argument("--force-verifier", action="store_true", default=True, help="Force verifier execution.")
    parser.add_argument("--all-tests", action="store_true", help="Run all 8 target validation tests (matrix A-H).")
    parser.add_argument(
        "--certify",
        action="store_true",
        help="Run in CERTIFICATION MODE (fail-closed): sets CERTIFICATION_MODE=true and "
        "CACHE_ENABLED=false so detector/BGE/NLI fallback or mock/empty evidence raises a "
        "controlled CertificationError instead of a best-effort verdict.",
    )
    args = parser.parse_args()

    if args.certify:
        os.environ["CERTIFICATION_MODE"] = "true"
        os.environ["CACHE_ENABLED"] = "false"
        _vdir = os.path.join(PROJECT_ROOT, "agents", "verifier_agent")
        if _vdir not in sys.path:
            sys.path.insert(0, _vdir)
        try:  # defensive: ensure Settings re-reads the env we just set
            from config.settings import get_settings

            get_settings.cache_clear()
        except Exception:
            pass
        print("\n" + "*" * 76)
        print("  CERTIFICATION MODE ENABLED — fail-closed runtime proof")
        print("  A degraded detector/BGE/NLI, or mock/empty/malformed evidence, will raise")
        print("  a controlled CertificationError rather than produce a verdict.")
        print("*" * 76)

    if args.all_tests:
        asyncio.run(run_all_live_tests())
    elif args.query:
        asyncio.run(
            run_single_test(
                title=f"Custom Query: {args.query}",
                query=args.query,
                domain=args.domain,
                force_verifier=args.force_verifier,
            )
        )
    else:
        # Default: run all 5 target tests
        asyncio.run(run_all_live_tests())


if __name__ == "__main__":
    main()
