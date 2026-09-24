"""End-to-end demo of ONLY three agents: Judge -> Corrector -> ReVerifier.

This deliberately does NOT run the detector, the first-pass verifier, or memory.
We hand the Judge a canonical VerifierResult (as if the Verifier had already run),
then chain the REAL production components:

    1. JudgeAgent.evaluate(...)                 -> decision + correction_request
    2. CharacterRegenerator.regenerate(...)     -> live LLM fact repair
    3. VerificationPipeline.verify(...)         -> live retrieval + NLI re-check
    4. JudgeAgent.evaluate(reverification=...)   -> final ACCEPT / REJECT

Run:  python scripts/demo_three_agents_e2e.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
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

from agents.judge_agent.judge_agent import JudgeAgent
from services.character_regenerator import CharacterRegenerator
from orchestration.graph import _get_verifier_imports
from orchestration.schemas import (
    ClaimReport,
    Evidence,
    VerifierResult,
    ReverificationResult,
    VerdictLabel,
    EntailmentLabel,
    ExecutionStatus,
)


def _hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _dump(obj) -> str:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    return json.dumps(obj, indent=2, default=str)


# --- The draft under test (overridable via CLI) ----------------------------
DEFAULT_QUERY = "Who created the Java programming language and when?"
DEFAULT_DRAFT = "Java was created by Dennis Ritchie in 1972."
DEFAULT_EVIDENCE = (
    "Java was originally developed by James Gosling at Sun Microsystems and "
    "released in 1995. Dennis Ritchie created C, not Java."
)


def build_verifier_result(draft: str, evidence_snippet: str) -> VerifierResult:
    """Stand in for a completed first-pass Verifier: the core claim is CONTRADICTED
    by evidence. This is the ONLY hand-built object; everything downstream is real."""
    contradictory = Evidence(
        evidence_id="ev-1",
        title="Reference source",
        source="Encyclopaedia / official record",
        url="https://en.wikipedia.org/",
        snippet=evidence_snippet,
        entailment_label=EntailmentLabel.CONTRADICTION,
        entailment_score=0.98,
        credibility_score=0.95,
    )
    claim = ClaimReport(
        claim_id="c1",
        claim_text=draft,
        verdict=VerdictLabel.CONTRADICTED,
        support_score=0.02,
        contradiction_score=0.98,
        confidence_score=0.97,
        evidence=[contradictory],
    )
    return VerifierResult(
        query_id="demo-1",
        domain="general",
        claim_reports=[claim],
        evidence=[contradictory],
        overall_confidence=0.97,
        retrieved_sources_count=3,
        verified_sources_count=3,
        status=ExecutionStatus.COMPLETED,
    )


async def run_reverifier(corrected_text: str):
    """Faithful replica of orchestration.graph._reverifier_node's core: decompose
    the corrected text, run the REAL VerificationPipeline (live retrieval + NLI)."""
    Pipeline, Suspicious, Payload = _get_verifier_imports()

    # Claim decomposition (same fallback as the node).
    try:
        from agents.verifier_agent.claims.claim_decomposer import ClaimDecomposer

        claims = [c.text if hasattr(c, "text") else str(c)
                  for c in ClaimDecomposer().decompose(corrected_text)]
    except Exception:
        import re

        claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", corrected_text) if s.strip()]

    suspicious = [Suspicious(claim_id=f"rev-{i}", text=t) for i, t in enumerate(claims)]
    payload = Payload(
        query_id="rev-demo-1",
        domain="general",
        suspicious_claims=suspicious,
    )
    timeout = float(os.environ.get("VERIFIER_TIMEOUT_SECONDS", "120"))
    raw = await asyncio.wait_for(Pipeline().verify(payload), timeout=timeout)
    return raw


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--draft", default=DEFAULT_DRAFT)
    ap.add_argument("--evidence", default=DEFAULT_EVIDENCE,
                    help="contradictory-evidence snippet the (stand-in) Verifier found")
    ap.add_argument("--skip-reverify", action="store_true",
                    help="run ONLY Judge + Corrector (skip the slow live ReVerifier)")
    args = ap.parse_args()
    user_query, draft = args.query, args.draft

    judge = JudgeAgent()

    _hr("INPUT  (any draft response under test)")
    print(f"user_query : {user_query}")
    print(f"draft      : {draft}")
    print(f"(verifier's contradictory evidence): {args.evidence}")

    vr = build_verifier_result(draft, args.evidence)

    # ---- STAGE 1: JUDGE (first pass) ------------------------------------
    _hr("STAGE 1 - JUDGE")
    print("GOT  : the Verifier report (1 claim, verdict=CONTRADICTED + evidence).")
    print("DID  : arbitrated the verdicts against the query.")
    print("HOW  : deterministic rule tree, NO network/LLM; evidence dominates.\n")
    jr = judge.evaluate(
        verifier_result=vr,
        user_query=user_query,
        original_response=draft,
        draft_response=draft,
        domain="general",
        reverification_result=None,
        retry_count=0,
    )
    print(f"decision        : {jr.decision}")
    print(f"decision_basis  : {jr.decision_basis}")
    print(f"reason          : {jr.reason}")
    print(f"decision_metrics: {json.dumps(jr.decision_metrics, default=str)}")
    if jr.correction_request is None:
        print("\n[STOP] Judge did not request a correction; nothing to hand the Corrector.")
        return
    print(f"OUTPUT -> correction_request for: "
          f"{[c.claim_text for c in jr.correction_request.claims_to_correct]}")

    # ---- STAGE 2: CORRECTOR (live LLM) ----------------------------------
    _hr("STAGE 2 - CORRECTOR")
    print("GOT  : the correction_request (bad claim + the contradictory evidence).")
    print("DID  : located the wrong sentence and regenerated it, grounded in evidence.")
    print("HOW  : live LLM per-sentence rewrite (Groq->Gemini->OpenRouter failover).\n")
    corr = await CharacterRegenerator().regenerate(jr.correction_request)
    print(f"status           : {corr.status}")
    print(f"validation_status: {corr.validation_status}  (candidate NOT self-certified)")
    print(f"failure_category : {corr.failure_category}")
    print(f"provider_used    : {corr.provider_used}")
    print(f"BEFORE           : {corr.original_text}")
    print(f"AFTER            : {corr.corrected_text}")

    if not corr.corrected_text or str(corr.status).endswith("failed"):
        print("\n[FAIL-CLOSED] Corrector produced no usable candidate -> would escalate.")
        return

    if args.skip_reverify:
        _hr("SUMMARY (Judge + Corrector only)")
        print(f"draft     : {draft}")
        print(f"corrected : {corr.corrected_text}")
        print(f"judge     : {jr.decision} ({jr.decision_basis})")
        print(f"corrector : {corr.status} / {corr.validation_status} via {corr.provider_used}")
        print("\n(ReVerifier skipped via --skip-reverify.)")
        return

    # ---- STAGE 3: REVERIFIER (live retrieval + NLI) ---------------------
    _hr("STAGE 3 - REVERIFIER")
    print("GOT  : the corrector's candidate text (nothing is trusted yet).")
    print("DID  : independently re-extracted its claims and re-checked them.")
    print("HOW  : live web retrieval + NLI; UNKNOWN is NOT treated as CONTRADICTED.\n")
    try:
        raw = await run_reverifier(corr.corrected_text)
    except Exception as exc:  # timeout/degraded -> fail closed, never a silent pass
        print(f"[REVERIFIER FAILED] {type(exc).__name__}: {exc}")
        print("-> non-authoritative run; would route to human_escalation.")
        return

    claim_ev = getattr(raw, "claim_evidence", None) or []
    contradicted = [
        c for c in claim_ev
        if str((c.get("verdict") if isinstance(c, dict) else getattr(c, "verdict", ""))).lower()
        .endswith("contradicted")
    ]
    remaining = len(contradicted)
    print(f"re-extracted claims     : {len(claim_ev)}")
    print(f"remaining_contradictions: {remaining}")
    for c in claim_ev:
        cd = c if isinstance(c, dict) else c.__dict__
        print(f"  - [{cd.get('verdict')}] {cd.get('claim_text', cd.get('text',''))}")

    rev = ReverificationResult(
        passed=(remaining == 0),
        verifier_result=VerifierResult(query_id="rev-demo-1", domain="general",
                                       status=ExecutionStatus.COMPLETED),
        remaining_contradictions=remaining,
        status=ExecutionStatus.COMPLETED,
        failure_category=None if remaining == 0 else "REMAINING_CONTRADICTION",
    )

    # ---- STAGE 4: JUDGE (final post-reverify arbitration) ---------------
    _hr("STAGE 4 - JUDGE (final verdict on the correction)")
    print("GOT  : the reverification result.")
    print("DID  : decided whether the corrected answer may be released.")
    print("HOW  : ACCEPT only if reverify passed AND no contradictions remain.\n")
    final = judge.evaluate(
        verifier_result=vr,
        user_query=user_query,
        original_response=corr.corrected_text,
        domain="general",
        reverification_result=rev,
        retry_count=1,
    )
    print(f"final decision      : {final.decision}")
    print(f"final decision_basis: {final.decision_basis}")

    _hr("END-TO-END SUMMARY")
    print(f"draft     : {draft}")
    print(f"corrected : {corr.corrected_text}")
    print(f"judge#1   : {jr.decision} ({jr.decision_basis})")
    print(f"corrector : {corr.status} / {corr.validation_status} via {corr.provider_used}")
    print(f"reverify  : passed={rev.passed} remaining={rev.remaining_contradictions}")
    print(f"judge#2   : {final.decision} ({final.decision_basis})")


if __name__ == "__main__":
    asyncio.run(main())
