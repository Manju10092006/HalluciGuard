"""
HalluciGuard CLI Tool — Verify any custom claim or question.

Usage:
  python scripts/verify_claim.py "Guido van Rossum created Python in 1991."
  python scripts/verify_claim.py "The Earth is flat."
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


async def verify_custom_claim(claim_text: str, domain: str = "general") -> None:
    print("\n" + "=" * 72)
    print("  HALLUCIGUARD VERIFIER — CLAIM VERIFICATION TRACE")
    print(f"  Claim  : '{claim_text}'")
    print(f"  Domain : {domain}")
    print("=" * 72)
    print("[1/3] Initializing pipeline & loading ML models (DeBERTa NLI + BGE Reranker)...")

    from api.pipeline import VerificationPipeline
    from schemas.models import SuspiciousClaim, VerifierInputV2

    pipeline = VerificationPipeline()
    payload = VerifierInputV2(
        query_id="custom-cli-verification",
        domain=domain,
        suspicious_claims=[SuspiciousClaim(claim_id="c1", text=claim_text)],
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
            print(f"    Snippet     : \"{ev.snippet[:200]}...\"" if len(ev.snippet) > 200 else f"    Snippet     : \"{ev.snippet}\"")
    else:
        print("\n  No evidence passed the decision-grade relevance and NLI thresholds.")

    print("\n" + "=" * 72 + "\n")


def interactive_mode(default_domain: str = "general") -> None:
    print("\n" + "=" * 72)
    print("  HALLUCIGUARD INTERACTIVE VERIFICATION CONSOLE")
    print("  Type any claim to verify against real web/authoritative sources.")
    print("  Type 'domain <name>' to change domain (general, healthcare, cybersecurity, ai_research).")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 72 + "\n")

    current_domain = default_domain

    while True:
        try:
            claim = input(f"[{current_domain}] Enter claim > ").strip()
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

        asyncio.run(verify_custom_claim(claim, current_domain))


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
        "-i", "--interactive",
        action="store_true",
        help="Launch interactive verification REPL console.",
    )
    args = parser.parse_args()

    if args.interactive or not args.claim:
        interactive_mode(args.domain)
    else:
        asyncio.run(verify_custom_claim(args.claim, args.domain))


if __name__ == "__main__":
    main()

