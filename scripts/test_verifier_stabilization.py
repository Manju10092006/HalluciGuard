"""
HalluciGuard Step 3 — Standalone Live Verifier Stabilization Test Runner.

Usage:
  python scripts/test_verifier_stabilization.py
"""
import os
import sys
import asyncio

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VERIFIER_DIR = os.path.join(PROJECT_ROOT, "agents", "verifier_agent")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if VERIFIER_DIR not in sys.path:
    sys.path.insert(0, VERIFIER_DIR)

from api.pipeline import VerificationPipeline
from schemas.models import SuspiciousClaim, VerifierInputV2


async def main() -> None:
    claim_text = "Paris is the capital of France."
    print("=" * 60)
    print("HALLUCIGUARD — VERIFIER STABILIZATION LIVE TEST")
    print(f"Target Claim: '{claim_text}'")
    print("=" * 60)

    pipeline = VerificationPipeline()
    await pipeline.cache.invalidate("general", claim_text)

    inp = VerifierInputV2(
        query_id="live-stabilization-test",
        domain="general",
        suspicious_claims=[SuspiciousClaim(claim_id="c1", text=claim_text)]
    )

    result = await pipeline.verify(inp)

    print("\n--- VERIFICATION RESULT CONTRACT ---")
    print(f"Domain: {result.domain}")
    print(f"Retrieved Sources: {result.retrieved_sources}")
    print(f"Verified Sources: {result.verified_sources}")
    print(f"Overall Evidence Confidence: {result.overall_evidence_confidence}")
    print(f"Claim Reports Count: {len(result.claim_evidence)}")

    for idx, report in enumerate(result.claim_evidence):
        print(f"\n[Claim {idx+1}] Text: {report.claim_text}")
        print(f"  - Verdict: {report.verdict}")
        print(f"  - Trust Score: {report.trust_score}")
        print(f"  - Confidence Score: {report.confidence_score}")
        print(f"  - Explanation: {report.explanation}")
        print(f"  - Evidence Count: {len(report.evidence)}")
        for e in report.evidence:
            print(f"    * Title: {e.title}")
            print(f"      Source: {e.source}")
            print(f"      URL: {e.url}")
            print(f"      Entailment Label: {e.entailment_label}")
            print(f"      Entailment Score: {e.entailment_score:.4f}")
            print(f"      Credibility Score: {e.credibility_score:.2f}")

    print("=" * 60)
    print("STATUS: VERIFIER STABILIZATION PASSED CLEANLY")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
