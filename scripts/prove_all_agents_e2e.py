"""Prove ALL agents in the REAL orchestration graph run end-to-end.

Runs the full production pipeline via run_verification and, walking the live
execution TRACE, prints for EACH agent that fired: what it received, what it
produced, and its status. No agent is mocked; this is the production graph.

Run:  python scripts/prove_all_agents_e2e.py [--query Q] [--draft D]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from orchestration.graph import run_verification

DEFAULT_QUERY = "Who is the founder of Microsoft?"
DEFAULT_DRAFT = "Snehith is the founder of Microsoft."


def _as_dict(obj):
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return {"_raw": obj}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return {}


def _short(v, n=280):
    s = json.dumps(v, default=str) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + " …"


def _hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument(
        "--draft",
        default="",
        help="OPTIONAL. Real product flow: leave empty and the Generator (base_llm) "
        "writes the draft from the query itself. Supply one only to inject a known "
        "hallucination so the Corrector's repair is visible on demand.",
    )
    args = ap.parse_args()

    # llm_response="" -> _generate_node calls BaseLLMService to produce the draft
    # from the query alone (exactly how a user hits the product).
    result = asyncio.run(run_verification(args.query, args.draft, domain="general"))

    trace = result.get("trace", [])
    nodes = [t.get("node") for t in trace]

    _hr("INPUT")
    print("query :", args.query)
    if args.draft:
        print("draft : (supplied by caller to inject a known hallucination)")
        print("      :", args.draft)
    else:
        print("draft : (none supplied -> Generator writes it from the query, real product flow)")

    _hr("EXECUTION TRACE (real production graph)")
    print(" -> ".join(str(n) for n in nodes))

    det = _as_dict(result.get("detector_result") or result.get("detector"))
    claims = result.get("detected_claims") or []
    analysis = _as_dict(result.get("claim_analysis"))
    ver = _as_dict(result.get("verifier_result") or result.get("verifier"))
    raw_ver = _as_dict(result.get("verifier"))
    judge = _as_dict(result.get("judge"))
    corr = _as_dict(result.get("corrector") or result.get("correction_result"))
    rev = _as_dict(result.get("reverification_result"))
    mem = _as_dict(result.get("memory_result"))

    def agent(idx, name, node_key, got, did):
        ran = node_key in nodes
        mark = "FIRED" if ran else "did NOT run"
        print(f"\n[{idx}] {name:<16} -> {mark}")
        if ran:
            print("    GOT :", got)
            print("    DID :", did)

    _hr("PER-AGENT REPORT (in execution order)")

    agent(1, "GENERATOR", "base_llm",
          f"user query: {args.query!r}",
          f"draft under test: {_short(result.get('draft_response') or args.draft)}")

    agent(2, "DETECTOR", "detector",
          f"draft response ({len(args.draft)} chars)",
          f"triage={det.get('next_action')} risk={det.get('risk_level')} "
          f"score={det.get('hallucination_probability') if det.get('probability_available') else 'N/A (pre-retrieval)'} "
          f"atomic_claims={len(claims)}")

    agent(3, "CLAIM_ANALYZER", "claim_analyzer",
          f"{len(claims)} detected claim(s)",
          f"analysis={_short(analysis) if analysis else 'passed claims to verifier'}")

    agent(4, "VERIFIER", "verifier",
          f"{len(claims)} suspicious claim(s) + live retrieval",
          f"status={ver.get('status')} overall_confidence={ver.get('overall_confidence')} "
          f"claim_reports={len(ver.get('claim_reports', []))} "
          f"verdicts={[c.get('verdict') for c in ver.get('claim_reports', [])]}\n"
          f"          retrieval={[(c.get('retrieved_documents'), c.get('reranked_documents'), c.get('verified_evidence')) for c in raw_ver.get('claim_evidence', [])]}\n"
          f"          evidence_titles={[[e.get('title') for e in c.get('evidence', [])] for c in raw_ver.get('claim_evidence', [])]}")

    agent(5, "JUDGE", "judge",
          "verifier report + (later) reverification result",
          f"decision={judge.get('decision')} basis={judge.get('decision_basis')} "
          f"metrics={_short(judge.get('decision_metrics', {}))}")

    agent(6, "CORRECTOR", "corrector",
          "judge's correction request (bad claim + evidence)",
          f"status={corr.get('status')} provider={corr.get('provider_used')}\n"
          f"          BEFORE: {corr.get('original_text')}\n"
          f"          AFTER : {corr.get('corrected_text')}")

    agent(7, "REVERIFIER", "reverifier",
          "corrector's candidate text (nothing trusted yet)",
          f"passed={rev.get('passed')} remaining_contradictions={rev.get('remaining_contradictions')} "
          f"failure_category={rev.get('failure_category')}")

    agent(8, "MEMORY", "memory",
          "accepted+verified claims only",
          f"status={mem.get('status')} stored={mem.get('stored_count', mem.get('facts_stored'))}")

    _hr("FINAL OUTCOME")
    print("final_response  :", result.get("final_response"))
    print("verification    :", result.get("verification_status"))
    print("terminal_status :", result.get("terminal_status"))
    print("active_agents   :", result.get("active_agents"))
    print("disabled_agents :", result.get("disabled_agents"))
    print("errors          :", result.get("errors"))

    _hr("SCORECARD")
    expected = ["base_llm", "detector", "claim_analyzer", "verifier",
                "judge", "corrector", "reverifier", "memory"]
    names = {"base_llm": "Generator", "detector": "Detector",
             "claim_analyzer": "ClaimAnalyzer", "verifier": "Verifier",
             "judge": "Judge", "corrector": "Corrector",
             "reverifier": "ReVerifier", "memory": "Memory"}
    for key in expected:
        print(f"  {names[key]:<14}: {'OK  (fired)' if key in nodes else 'not run'}")
    fired = sum(1 for k in expected if k in nodes)
    print(f"\n  {fired}/{len(expected)} agents fired; errors={len(result.get('errors') or [])}")


if __name__ == "__main__":
    main()
