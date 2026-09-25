"""
================================================================================
 HalluciGuard — 7-Agent Pipeline Local Tester (single script, one command)
================================================================================

WHAT THIS SCRIPT IS
    A single end-to-end test that runs your query through ALL SEVEN agents of
    the HalluciGuard pipeline (in-process, via the LangGraph in
    orchestration/graph.py) and then prints a fully-narrated breakdown so anyone
    reading the output understands exactly what happened at each stage — what
    the agent's job is, what it received, what it did, what it decided, and
    where control went next.

HOW TO RUN
    cd "C:\\Users\\S.Manjunath Reddy\\OneDrive\\Music\\Pictures\\Videos\\HalluciGuard"
    python test_direct.py "what is the capital of india"

    Options (all optional):
      --draft "..."     supply your OWN draft answer (skips the Base LLM call)
      --domain general  verification domain: general | biomedical | finance ...
      --stress          generation_mode=stress_test (nudges the LLM to slip up,
                        so you can watch Detector/Verifier/Corrector actually fire)
      --json            also dump the full raw pipeline state as JSON at the end

    Needs your .env (OPENROUTER_API_KEY for the Base LLM; TAVILY_API_KEY + the
    n8n webhook for evidence retrieval). Wikipedia is used with no key.

THE PIPELINE (what the backend actually does, in order)

    YOUR QUERY
        │
        ▼
    (1) BASE LLM ............ generates a DRAFT answer. Not trusted yet.
        │  draft_answer
        ▼
    (2) DETECTOR ............ TRIAGE only. Scores hallucination RISK of the
        │                     draft. Never decides truth. Routes to the Verifier.
        │  risk score
        ▼
    ( ) CLAIM EXTRACTION .... splits the draft into atomic, checkable claims
        │                     (a function inside the pipeline, not a separate
        │  atomic claims       agent). Claim IDs are preserved end-to-end.
        ▼
    (3) VERIFIER ............ the ARBITER. For each claim it retrieves evidence
        │                     (n8n + Tavily + Wikipedia), runs NLI/entailment,
        │  per-claim verdicts   and returns SUPPORTED / CONTRADICTED / UNKNOWN.
        ▼
    (4) JUDGE ............... reads Detector risk + Verifier verdicts and DECIDES
        │                     the action: ACCEPT / CORRECT_AND_ACCEPT / REJECT /
        │  decision            VERIFY_AGAIN. Retrieves nothing itself.
        ├──────── ACCEPT ─────────────────────────────────┐
        │                                                  │
        ▼ CORRECT_AND_ACCEPT                               │
    (5) CORRECTOR .......... repairs ONLY the contradicted claims using the      │
        │                    verified evidence; preserves the correct parts.     │
        │  corrected_answer   Runs conditionally — skipped when Judge = ACCEPT.   │
        ▼                                                                         │
    (6) REVERIFIER ......... re-extracts claims FROM THE CORRECTED answer and     │
        │                    verifies again, catching any NEW error the repair    │
        │  pass / fail        introduced. Fails back to correction if needed.     │
        └──────────────────────────────┬───────────────────────────────────────┘
                                        ▼
                                  FINAL ANSWER
                                        │
                                        ▼
    (7) MEMORY ............. persists the VERIFIED facts (+ correction history)
                             to the knowledge graph / vector store for reuse.

    Key principle: the Base LLM GENERATES, the Detector TRIAGES, the Verifier
    VERIFIES with evidence, the Judge DECIDES, the Corrector REPAIRS, the
    ReVerifier VALIDATES the repair, and Memory PRESERVES. Only the Verifier and
    Judge decide anything about truth — the Detector is a risk signal only.
================================================================================
"""

import argparse
import asyncio
import json
import sys
import time

from dotenv import load_dotenv

# Windows CMD defaults to cp1252 and chokes on the box-drawing / bullet glyphs
# below; force UTF-8 so the narrated output renders correctly.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

# The whole 7-agent pipeline is wired as one LangGraph. run_verification() feeds
# the query in at the top and returns the final state dict with every agent's
# output attached — that single call is what exercises all seven agents.
from orchestration.graph import run_verification


# ── tiny presentation helpers ────────────────────────────────────────────────
BAR = "=" * 80
SUB = "-" * 80


def header(step: str, name: str, tagline: str) -> None:
    """Print a labelled section header for one agent."""
    print("\n" + SUB)
    print(f"{step}. [AGENT] {name}")
    print(f"     {tagline}")
    print(SUB)


def explain(*lines: str) -> None:
    """Print the 'what this agent does' explanation (indented, wrapped by hand)."""
    for ln in lines:
        print(f"  ▸ {ln}")
    print()  # blank line before the runtime values


def val(label: str, value) -> None:
    print(f"  • {label:26} {value}")


def _clean_enum(x) -> str:
    """RiskLevel.HIGH -> HIGH; 'accept' -> ACCEPT."""
    return str(x).upper().split(".")[-1]


# ── the run ───────────────────────────────────────────────────────────────────
async def run(query: str, draft: str | None, domain: str, stress: bool, dump_json: bool) -> None:
    print("\n" + BAR)
    print("      HALLUCIGUARD — 7-AGENT PIPELINE, FULL BACKEND WALKTHROUGH")
    print(BAR)
    print(f'  Input query : "{query}"')
    print(f"  Domain      : {domain}")
    print(f"  Mode        : {'stress_test (provoke hallucinations)' if stress else 'normal'}")
    if draft:
        print(f'  Supplied draft (Base LLM skipped): "{draft}"')
    print("\n  Running the LangGraph multi-agent workflow end-to-end ...")
    print("  (first run downloads/loads the detector, reranker, and NLI models)")

    t0 = time.time()
    kwargs = {"user_query": query, "domain": domain}
    if draft:
        kwargs["llm_response"] = draft
    if stress:
        kwargs["generation_mode"] = "stress_test"
    result = await run_verification(**kwargs)
    elapsed = time.time() - t0

    print("\n" + BAR)
    print("  PIPELINE EXECUTION — STAGE BY STAGE")
    print(BAR)

    # ── 1. BASE LLM ──────────────────────────────────────────────────────────
    header("1", "BASE LLM  (Draft Generator)",
           "Produces the first-draft answer. This is a DRAFT, not the truth.")
    explain(
        "ROLE : answer the query from the model's own parametric knowledge.",
        "IN   : your query (+ optional conversation history).",
        "OUT  : a draft answer that the rest of the pipeline will fact-check.",
        "WHY  : the LLM may be confidently wrong — everything below exists to",
        "       catch and repair that before the answer reaches the user.",
    )
    base_llm = result.get("base_llm", {})
    draft_out = result.get("draft_response", result.get("llm_response", ""))
    val("Model provider", base_llm.get("provider", "openrouter"))
    val("Model name", base_llm.get("model", "qwen/qwen-2.5-7b-instruct"))
    val("Generation latency (ms)", base_llm.get("latency_ms", 0))
    print(f'  • Draft answer:\n      "{draft_out}"')

    # ── 2. DETECTOR ────────────────────────────────────────────────────────────
    header("2", "DETECTOR AGENT  (Hallucination Risk Triage)",
           "Scores how risky the draft looks. TRIAGE ONLY — decides nothing.")
    explain(
        "ROLE : estimate hallucination RISK of the draft (a priority signal).",
        "IN   : the query + the draft answer.",
        "OUT  : a risk level / probability and a suggested route.",
        "NOTE : this model has known train/serve skew (near-constant output), so",
        "       by design it can NEVER accept/reject on its own — it only decides",
        "       how urgently the Verifier should look. The Verifier is the arbiter.",
    )
    detector = result.get("detector_result") or result.get("detector", {})
    probability_available = bool(
        detector.get(
            "probability_available",
            detector.get("hallucination_probability") is not None,
        )
    )
    degraded = bool(detector.get("detector_degraded"))
    val("Hallucination risk level", _clean_enum(detector.get("risk_level", "UNKNOWN")))
    if probability_available:
        prob = float(detector.get("hallucination_probability", 0.0))
        conf = float(detector.get("confidence_score", detector.get("confidence", 0.0)))
        val("Hallucination probability", f"{prob:.4f}")
        val("Confidence score", f"{conf:.4f}")
    else:
        val("Hallucination probability", "N/A — grounded evidence is unavailable at triage")
        val("Confidence score", "N/A — routing fail-closed to Verifier")
    val("Recommended route", _clean_enum(detector.get("next_action", "verify")))
    val("Detector degraded?", degraded)

    # ── (claim extraction) + 3. VERIFIER ───────────────────────────────────────
    header("3", "VERIFIER AGENT  (Evidence Retrieval + NLI  — THE ARBITER)",
           "Splits the draft into atomic claims and checks each against evidence.")
    explain(
        "ROLE : the real fact-checker. First CLAIM EXTRACTION breaks the draft",
        "       into atomic, checkable statements (claim IDs preserved). Then for",
        "       each claim it retrieves evidence and runs NLI/entailment.",
        "IN   : the draft's atomic claims (+ domain to pick sources).",
        "SOURCES: n8n workflow (routes to Wikipedia/PubMed/NVD/arXiv/SEC),",
        "         Tavily web search, and Wikipedia — n8n only FINDS evidence, it",
        "         never decides truth; this Python layer does the judging.",
        "OUT  : per-claim verdict SUPPORTED / CONTRADICTED / UNKNOWN with the",
        "       grounding evidence snippets and entailment scores.",
    )
    verifier = result.get("verifier_result") or result.get("verifier", {})
    claim_evidence = verifier.get("claim_evidence") or verifier.get("claim_reports", [])
    val("Search integrations", verifier.get("sources_attempted", ["n8n", "tavily", "wikipedia"]))
    val("Atomic claims evaluated", len(claim_evidence))
    val("Verification status", result.get("verification_status", "N/A"))
    for idx, claim in enumerate(claim_evidence, 1):
        c_text = claim.get("claim_text") or claim.get("claim") or claim.get("text", "")
        print(f'\n    ── Claim #{idx}: "{c_text}"')
        print(f"       Verdict     : {_clean_enum(claim.get('verdict', 'unverified'))}")
        if claim.get("explanation") and claim.get("explanation") != "N/A":
            print(f"       Explanation : {claim.get('explanation')}")
        ev_list = claim.get("evidence", [])
        print(f"       Evidence    : {len(ev_list)} snippet(s) retrieved & entailment-scored")
        for j, ev in enumerate(ev_list[:3], 1):
            src = ev.get("source", "unknown")
            lbl = ev.get("entailment_label", "")
            score = ev.get("entailment_score", "")
            snippet = (ev.get("snippet", "") or "")[:120]
            print(f"         ({j}) [{src}] NLI={lbl} {score}: {snippet}...")

    # ── 4. JUDGE ───────────────────────────────────────────────────────────────
    header("4", "JUDGE AGENT  (Arbitration & Decision)",
           "Turns evidence verdicts into ONE action. Retrieves nothing itself.")
    explain(
        "ROLE : decide what the system does with the answer.",
        "IN   : Detector risk + the Verifier's per-claim verdicts + evidence.",
        "OUT  : one decision —",
        "         ACCEPT ............. all important claims supported -> ship as-is",
        "         CORRECT_AND_ACCEPT . a claim is contradicted -> send to Corrector",
        "         REJECT ............. answer is unsalvageable",
        "         VERIFY_AGAIN ....... evidence insufficient -> re-check",
        "RULE : evidence outranks the Detector — a 'no hallucination' hunch never",
        "       overrides a CONTRADICTED verdict, and vice-versa.",
    )
    judge = result.get("judge_result") or result.get("judge", {})
    corr_req = judge.get("correction_request") or result.get("correction_request")
    val("Decision", _clean_enum(judge.get("decision", result.get("judge_decision", "N/A"))))
    val("Severity", _clean_enum(judge.get("severity", result.get("severity", "LOW"))))
    val("Confidence", judge.get("confidence", 0.0))
    val("Reason", judge.get("reason", "N/A"))
    val("Correction requested?", bool(corr_req))

    # ── 5. CORRECTOR ───────────────────────────────────────────────────────────
    header("5", "CORRECTOR AGENT  (Grounded Fact-Repair)",
           "Repairs ONLY the contradicted claims using verified evidence.")
    explain(
        "ROLE : rewrite the wrong parts, keep the correct parts, using ONLY the",
        "       verified evidence — never invent new facts.",
        "IN   : the draft + contradicted claims + evidence + Judge decision.",
        "OUT  : a corrected answer.",
        "WHEN : conditional — runs only if the Judge requested correction.",
        "       Otherwise it reports NOT REQUIRED instead of an empty state.",
    )
    corrector = result.get("correction_result") or result.get("corrector")
    if corrector:
        print("  • Status: TRIGGERED & COMPLETED")
        val("Validation status", corrector.get("validation_status", "validated"))
        val("Attempts made", result.get("correction_attempt_count", 1))
        val("Changed claims", corrector.get("changed_claims", []))
        print(f'  • Original draft:\n      "{corrector.get("original_text", draft_out)}"')
        print(f'  • Corrected answer:\n      "{corrector.get("corrected_text", "")}"')
        if corrector.get("reasoning"):
            val("Repair reasoning", corrector.get("reasoning"))
    else:
        outcome = (result.get("agent_outcomes") or {}).get("corrector", {})
        print("  • Status: EVALUATED — NOT REQUIRED")
        val("Reason", outcome.get("reason", "Judge found no contradicted claim requiring repair"))

    # ── 6. REVERIFIER ──────────────────────────────────────────────────────────
    header("6", "REVERIFIER  (Post-Correction Re-Validation)",
           "Re-checks the CORRECTED answer so a repair can't sneak in a new error.")
    explain(
        "ROLE : independently validate the repaired answer.",
        "IN   : the corrected answer + original claims + evidence + Judge decision.",
        "DOES : RE-EXTRACTS claims from the corrected text (not just the one that",
        "       was flagged) and verifies each again — catching a fix that",
        "       corrected the creator but introduced a wrong year, etc.",
        "OUT  : VERIFIED -> ship it;  FAILED -> route back to correction.",
        "WHEN : conditional — only runs if the Corrector ran.",
    )
    reverifier = result.get("reverification_result")
    if reverifier:
        print("  • Status: TRIGGERED & EVALUATED")
        val("Passed", reverifier.get("passed"))
        val("Remaining contradictions", reverifier.get("remaining_contradictions", 0))
        val("Reverification attempts", result.get("reverification_attempt_count", 1))
    else:
        outcome = (result.get("agent_outcomes") or {}).get("reverifier", {})
        print("  • Status: EVALUATED — NOT REQUIRED")
        val("Reason", outcome.get("reason", "No corrected answer was produced"))

    # ── 7. MEMORY ──────────────────────────────────────────────────────────────
    header("7", "MEMORY AGENT  (Verified-Knowledge Persistence)",
           "Stores the VERIFIED facts (+ correction history) for future reuse.")
    explain(
        "ROLE : persist what was actually verified — not raw LLM output.",
        "IN   : the final verified claims, evidence, verdicts, correction history.",
        "OUT  : records in the knowledge graph + vector store, with fact IDs.",
        "WHY  : lets future queries reuse trusted facts as additional context",
        "       (as evidence, never as unquestionable truth).",
    )
    memory = result.get("memory_result") or result.get("memory", {})
    mem_status = memory.get("status", "stored" if memory.get("count", 0) > 0 else "skipped")
    stored_cnt = memory.get("stored_count", memory.get("count", 0))
    fact_ids = memory.get("fact_ids", [f.get("fact_id") for f in memory.get("stored", []) if isinstance(f, dict)])
    val("Persistence status", mem_status)
    val("New verified facts stored", stored_cnt)
    val("Existing duplicates reused", memory.get("duplicate_count", 0))
    if fact_ids:
        val("Stored/referenced fact IDs", fact_ids)
    val("Knowledge graph active", memory.get("knowledge_graph", True))
    val("Vector memory active", memory.get("vector_memory", True))
    if memory.get("skipped_reason") or memory.get("reason"):
        val("Note", memory.get("skipped_reason") or memory.get("reason"))

    # ── final summary ──────────────────────────────────────────────────────────
    print("\n" + BAR)
    print("  SUPERVISOR SUMMARY  (how the pipeline resolved)")
    print(BAR)
    path = _describe_path(result)
    val("Route taken", path)
    val("Terminal status", result.get("terminal_status", "completed"))
    val("Verification status", result.get("verification_status", "completed"))
    val("Total pipeline time", f"{elapsed:.2f}s")
    print(f'\n  • FINAL ANSWER DELIVERED TO USER:\n      "{result.get("final_response", draft_out)}"')
    print(BAR + "\n")

    if dump_json:
        print("\n[--json] full raw pipeline state:\n")
        print(json.dumps(result, indent=2, default=str))


def _describe_path(result: dict) -> str:
    """One-line human summary of which branch the pipeline took."""
    corrected = bool(result.get("correction_result") or result.get("corrector"))
    judge_info = result.get("judge_result") or result.get("judge") or {}
    decision = _clean_enum(judge_info.get("decision", result.get("judge_decision", "")))
    term_status = str(result.get("terminal_status", "")).lower()
    ver_status = str(result.get("verification_status", "")).lower()

    if term_status == "human_review" or ver_status == "human_review_required" or decision == "ABSTAIN":
        return f"generate → detector → verifier → judge({decision or 'ABSTAIN'}) → human_escalation → memory (human review required)"
    if corrected:
        return "generate → detector → verifier → judge(CORRECT) → corrector → reverifier → memory"
    if decision == "REJECT" or term_status == "rejected":
        return f"generate → detector → verifier → judge({decision or 'REJECT'}) → reject → memory (answer withheld)"
    if decision == "ACCEPT":
        return "generate → detector → verifier → judge(ACCEPT) → memory (no repair needed)"
    return f"generate → detector → verifier → judge({decision or 'UNKNOWN'}) → memory"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run one query through all 7 HalluciGuard agents with a narrated breakdown.")
    ap.add_argument("query", nargs="*", help="the query to verify (quote it)")
    ap.add_argument("--draft", help="supply your own draft answer (skips the Base LLM)")
    ap.add_argument("--domain", default="general", help="verification domain (general/biomedical/finance/...)")
    ap.add_argument("--stress", action="store_true", help="generation_mode=stress_test to provoke hallucinations")
    ap.add_argument("--json", action="store_true", help="also dump the full raw pipeline state as JSON")
    args = ap.parse_args()

    q = " ".join(args.query) if args.query else "What is the capital of India?"
    asyncio.run(run(q, args.draft, args.domain, args.stress, args.json))
