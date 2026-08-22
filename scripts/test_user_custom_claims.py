"""
HalluciGuard Custom User Claims Live Verification Test Runner.
Executes the user's custom claims end-to-end across general, healthcare, cybersecurity, finance, and AI research domains.
"""
from __future__ import annotations

import os
import sys
import asyncio
from dotenv import load_dotenv

# Ensure verifier_agent is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VERIFIER_DIR = os.path.join(PROJECT_ROOT, "agents", "verifier_agent")
if VERIFIER_DIR not in sys.path:
    sys.path.insert(0, VERIFIER_DIR)

load_dotenv()
os.environ["VERIFIER_CACHE_ENABLED"] = "false"

from api.pipeline import VerificationPipeline
from schemas.models import VerifierInputV2, SuspiciousClaim


async def run_custom_claims():
    claims_to_test = [
        {
            "category": "General",
            "domain": "general",
            "claim": "Amazon company was built by Sundar Pichai",
            "notes": "Factually false: Amazon was founded by Jeff Bezos. Sundar Pichai is CEO of Alphabet/Google."
        },
        {
            "category": "Healthcare / Medicine",
            "domain": "healthcare",
            "claim": "This iron tablet cures the headache",
            "notes": "Medical claim: Iron supplements treat iron deficiency anemia, not general headache cures."
        },
        {
            "category": "Cybersecurity",
            "domain": "cybersecurity",
            "claim": "The Trojan virus is the most dangerous cyber attack",
            "notes": "Cybersecurity taxonomy / terminology evaluation."
        },
        {
            "category": "Finance",
            "domain": "finance",
            "claim": "The 1992 Indian securities scam was the biggest financial scam in Indian stock market history",
            "notes": "Financial historical claim regarding Harshad Mehta / 1992 securities scam."
        },
        {
            "category": "AI Research (Fresh / Unused)",
            "domain": "ai_research",
            "claim": "AlphaFold was developed by DeepMind to predict 3D protein structures from amino acid sequences",
            "notes": "True AI research breakthrough fact."
        },
        {
            "category": "Computer Science / Quantum (Fresh / Unused)",
            "domain": "general",
            "claim": "Shor's algorithm is a quantum algorithm for integer factorization in polynomial time",
            "notes": "True foundational quantum computing theorem by Peter Shor (1994)."
        }
    ]

    pipeline = VerificationPipeline()

    print("=" * 80)
    print("  HALLUCIGUARD LIVE VERIFICATION — USER CUSTOM CLAIMS TEST SUITE")
    print("=" * 80)

    for idx, item in enumerate(claims_to_test, 1):
        print(f"\n[{idx}/{len(claims_to_test)}] TEST CATEGORY: {item['category']}")
        print(f"     CLAIM: \"{item['claim']}\"")
        print(f"     DOMAIN: {item['domain']}")
        print(f"     NOTES: {item['notes']}")
        print("-" * 80)

        payload = VerifierInputV2(
            query_id=f"user_test_{idx}",
            domain=item["domain"],
            suspicious_claims=[
                SuspiciousClaim(claim_id=f"c_{idx}", text=item["claim"])
            ]
        )

        result = await pipeline.verify(payload)
        report = result.claim_evidence[0]

        verdict_str = report.verdict.value.upper()
        print(f"  --> FINAL VERDICT       : [{verdict_str}]")
        print(f"  --> CONFIDENCE SCORE    : {report.confidence_score * 100:.2f}%")
        print(f"  --> TRUST SCORE         : {report.trust_score * 100:.2f}%")
        print(f"  --> SUPPORT SCORE       : {report.support_score:.4f}")
        print(f"  --> CONTRADICT SCORE    : {report.contradiction_score:.4f}")
        print(f"  --> TOTAL EVIDENCE ITEMS: {len(report.evidence)}")
        print(f"  --> EXPLANATION         : {report.explanation[:200]}...")

        if report.evidence:
            print("\n  DECISION-GRADE EVIDENCE CITATIONS:")
            for e_idx, e in enumerate(report.evidence, 1):
                label_val = e.entailment_label.value.upper()
                print(f"    [{e_idx}] [{label_val}] {e.title}")
                print(f"        Source     : {e.source}")
                print(f"        URL        : {e.url}")
                print(f"        NLI Match  : {e.entailment_score * 100:.1f}%")
                print(f"        Credibility: {e.credibility_score * 100:.1f}%")
                print(f"        Snippet    : \"{e.snippet[:150]}...\"")
        else:
            print("  (No decision-grade evidence met the threshold for citation)")

        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_custom_claims())
