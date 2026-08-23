"""
HalluciGuard — n8n Retrieval Service V2 Boundary Diagnostics.

Validates the n8n service boundary only without invoking downstream BGE/DeBERTa models.
This is a POST-only retrieval boundary test — it does NOT call any health endpoint:
  1. Calls the n8n verification webhook (POST) with auth headers
  2. Validates HTTP response status and latency
  3. Validates JSON payload and schema
  4. Prints evidence count and normalized provenance records

Usage:
  python scripts/test_n8n_retrieval.py
  python scripts/test_n8n_retrieval.py --claim "Paris is the capital of France."
  python scripts/test_n8n_retrieval.py --header-name X-API-Key --secret my-secret
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from services.n8n_retrieval_client import N8NRetrievalClient


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


async def run_diagnostics(
    claim: str = "Paris is the capital of France.",
    domain: str = "general",
    retrieval_mode: str = "hybrid",
    webhook_url: str | None = None,
    auth_mode: str | None = None,
    header_name: str | None = None,
    secret: str | None = None,
) -> int:
    client = N8NRetrievalClient(
        webhook_url=webhook_url,
        auth_mode=auth_mode or os.environ.get("N8N_AUTH_MODE", "header"),
        header_name=header_name or os.environ.get("N8N_HEADER_NAME", "X-API-Key"),
        webhook_secret=secret if secret is not None else os.environ.get("N8N_WEBHOOK_SECRET", ""),
    )

    print("=" * 72)
    print("  HALLUCIGUARD — N8N RETRIEVAL BOUNDARY TEST")
    print("=" * 72)
    print(f"  Claim         : {claim}")
    print(f"  Domain        : {domain}")
    print(f"  Webhook URL   : {client.webhook_url}")
    print(f"  Auth Mode     : {client.auth_mode}")
    print(f"  Header Name   : {client.header_name}")
    print(f"  Secret Set    : {'YES (masked)' if bool(client.webhook_secret) else 'NO (empty)'}")
    print(f"  Timeout       : {client.timeout_seconds}s")
    print("-" * 72)

    # ── Retrieval Webhook Check (POST) ──────────────────────────────────────
    print(f"\n[Retrieval Webhook] Executing POST request for claim: '{claim}'...")
    retrieval_res = await client.retrieve_evidence(
        claim=claim,
        domain=domain,
        retrieval_mode=retrieval_mode,
        request_id="diag-boundary-test",
    )

    print(f"  • Success          : {retrieval_res.success}")
    print(f"  • HTTP Status      : {retrieval_res.http_status}")
    if retrieval_res.trace:
        t = retrieval_res.trace
        print(f"  • Latency          : {t.latency_ms} ms")
        print(f"  • Workflow Version : {t.workflow_version}")
        print(f"  • Mode Used        : {t.retrieval_mode}")
        print(f"  • Primary Sources  : {t.primary_sources}")
        print(f"  • Tavily Used      : {t.tavily_called}")
        if t.counts:
            print(f"  • Counts           : {t.counts}")
        if t.performance:
            print(f"  • Performance      : {t.performance}")
        if t.primary_trace:
            print(f"  • Primary Trace    : {t.primary_trace}")
        if t.tavily_trace:
            print(f"  • Tavily Trace     : {t.tavily_trace}")
        if t.stage_traces:
            print(f"  • Stages Run       : {[s.get('stage') for s in t.stage_traces if isinstance(s, dict)]}")
        if t.legacy_n8n_output:
            print(f"  • Diagnostic Output: {t.legacy_n8n_output}")

    if retrieval_res.error:
        print(f"  • Error Details    : {retrieval_res.error}")

    passages = retrieval_res.passages
    print(f"\n  • Normalized Evidence Passages Count: {len(passages)}")

    if passages:
        print("\n" + "-" * 72)
        print("  PROVENANCE & PASSAGE RECORDS:")
        print("-" * 72)
        for idx, p in enumerate(passages, 1):
            print(f"\n  [{idx}] {p.title}")
            print(f"      Source          : {p.source} (ID: {p.source_id})")
            print(f"      URL             : {p.url or 'N/A'}")
            print(f"      Publication Date: {p.publication_date}")
            print(f"      adapter_score   : {p.source_confidence_hint:.4f}")
            print(f"      bge_score       : {p.relevance_score:.4f} (Reserved for Python BGE inference)")
            snip = p.snippet.replace("\n", " ").strip()
            print(f"      Snippet Preview : \"{snip[:160]}...\"" if len(snip) > 160 else f"      Snippet Preview : \"{snip}\"")
    else:
        print("  (No normalized passages returned)")

    print("\n" + "=" * 72)
    if retrieval_res.success and passages:
        print("  RESULT: SUCCESS — n8n retrieval boundary verified with valid evidence payload!")
        print("=" * 72 + "\n")
        return 0
    elif retrieval_res.success and not passages:
        print("  RESULT: SUCCESS (EMPTY EVIDENCE) — Controlled low/no-evidence response")
        print("=" * 72 + "\n")
        return 0
    else:
        print("  RESULT: FAILURE — n8n boundary retrieval failed (check errors above)")
        print("=" * 72 + "\n")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Test HalluciGuard n8n Retrieval Boundary.")
    parser.add_argument("--claim", default="Paris is the capital of France.", help="Claim text to retrieve evidence for.")
    parser.add_argument("--domain", default="general", help="Domain context (general, healthcare, etc.)")
    parser.add_argument("--retrieval-mode", default="hybrid", choices=["hybrid", "primary_only", "tavily_only"])
    parser.add_argument("--webhook-url", default=None, help="Override webhook URL")
    parser.add_argument("--auth-mode", default=None, choices=["none", "header"])
    parser.add_argument("--header-name", default=None, help="Custom header name for auth (default: X-API-Key)")
    parser.add_argument("--secret", default=None, help="Secret for header auth mode")
    args = parser.parse_args()

    sys.exit(
        asyncio.run(
            run_diagnostics(
                claim=args.claim,
                domain=args.domain,
                retrieval_mode=args.retrieval_mode,
                webhook_url=args.webhook_url,
                auth_mode=args.auth_mode,
                header_name=args.header_name,
                secret=args.secret,
            )
        )
    )


if __name__ == "__main__":
    main()
