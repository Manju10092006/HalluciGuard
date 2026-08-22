"""
HalluciGuard Verifier Diagnostic CLI Tool (`scripts/diagnose_verifier.py`).

Usage:
  python scripts/diagnose_verifier.py --claim "Paris is the capital of France." --domain general --cache false
  python scripts/diagnose_verifier.py --claim "Aspirin is used to treat mild pain." --domain healthcare --cache false
"""
from __future__ import annotations

import os
import sys
import asyncio
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VERIFIER_DIR = os.path.join(PROJECT_ROOT, "agents", "verifier_agent")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if VERIFIER_DIR not in sys.path:
    sys.path.insert(0, VERIFIER_DIR)

from api.pipeline import VerificationPipeline
from schemas.models import SuspiciousClaim, VerifierInputV2


async def diagnose(claim_text: str, domain: str, cache_enabled: bool) -> None:
    os.environ["VERIFIER_CACHE_ENABLED"] = "true" if cache_enabled else "false"

    pipeline = VerificationPipeline()
    if not cache_enabled:
        await pipeline.cache.invalidate(domain, claim_text)

    inp = VerifierInputV2(
        query_id="diagnostic-run",
        domain=domain,
        suspicious_claims=[SuspiciousClaim(claim_id="c1", text=claim_text)],
    )

    out = await pipeline.verify(inp)
    report = out.claim_evidence[0] if out.claim_evidence else None

    print("=" * 60)
    print("HALLUCIGUARD VERIFIER DIAGNOSTIC")
    print("=" * 60)

    print(f"\nClaim: {claim_text}")
    print(f"Domain: {domain}")

    print(f"\nCACHE:\n  enabled: {cache_enabled}\n  hit: {out.cache_hit}")

    print(f"\nDOMAIN ROUTING:")
    print(f"  requested_domain: {domain}")
    print(f"  resolved_domain: {out.domain}")
    print(f"  adapter: {out.adapter}")

    print(f"\nSOURCE EXECUTION:")
    print(f"  attempted: {out.sources_attempted}")
    print(f"  succeeded: {out.sources_succeeded}")
    print(f"  failed: {out.sources_failed}")

    print(f"\nRETRIEVAL:")
    print(f"  raw_passages: {out.retrieved_sources}")
    print(f"  verified_passages: {out.verified_sources}")

    print(f"\nNLI:")
    print(f"  model: {out.runtime_models.nli_model if out.runtime_models else 'cross-encoder/nli-deberta-v3-base'}")
    print(f"  executed: {bool(report and report.evidence)}")

    if report:
        print(f"\nEVIDENCE:")
        print(f"  supporting_sources: {report.supporting_sources}")
        print(f"  contradicting_sources: {report.contradicting_sources}")
        print(f"  verified_evidence_items: {len(report.evidence)}")
        for idx, ev in enumerate(report.evidence, 1):
            print(f"    #{idx} [{ev.entailment_label.value.upper()}] {ev.title} (Entailment Score: {ev.entailment_score:.4f}, Credibility: {ev.credibility_score:.2f})")
            print(f"       URL: {ev.url}")
            print(f"       Snippet: {ev.snippet[:120]}...")

        print(f"\nFINAL:")
        print(f"  verdict: {report.verdict.value.upper()}")
        print(f"  confidence: {report.confidence_score:.4f}")
        print(f"  support_score: {report.support_score:.4f}")
        print(f"  contradiction_score: {report.contradiction_score:.4f}")
        print(f"  trust_score: {report.trust_score:.4f}")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="HalluciGuard Verifier Diagnostic CLI.")
    parser.add_argument("--claim", required=True, help="Claim text to verify.")
    parser.add_argument("--domain", default="general", help="Domain profile.")
    parser.add_argument("--cache", default="false", choices=["true", "false"], help="Enable/disable cache.")
    args = parser.parse_args()

    cache_enabled = args.cache.lower() == "true"
    asyncio.run(diagnose(args.claim, args.domain, cache_enabled))


if __name__ == "__main__":
    main()
