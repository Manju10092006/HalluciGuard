"""
HalluciGuard Verifier End-to-End Diagnostic Tool (`scripts/diagnose_verifier_e2e.py`).

Provides complete visibility into all 8 pipeline stages:
1. QUERY (Original & Expanded)
2. RAW RETRIEVAL (Source, Title, URL, Snippet)
3. RELEVANCE / BGE (Reranker Scores)
4. RELEVANCE GATE (Accepted / Rejected Passages)
5. NLI (DeBERTa Softmax Probabilities: Entailment, Contradiction, Neutral)
6. EVIDENCE CLASSIFICATION (Supporting, Contradicting, Neutral, Irrelevant)
7. FINAL VERDICT (Verdict Label, Confidence, Support/Contradiction/Trust Scores)
8. PROVENANCE (Actual External Sources, Cache Status, Models Used)

Usage:
  python scripts/diagnose_verifier_e2e.py --claim "Paris is the capital of France." --domain general
  python scripts/diagnose_verifier_e2e.py --claim "Aspirin is used to treat mild pain." --domain healthcare
  python scripts/diagnose_verifier_e2e.py --claim "The Pacific Ocean is the largest ocean on Earth." --domain general
"""
from __future__ import annotations

import os
import sys
import asyncio
import argparse
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Setup sys.path before imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VERIFIER_DIR = os.path.join(PROJECT_ROOT, "agents", "verifier_agent")

if VERIFIER_DIR not in sys.path:
    sys.path.insert(0, VERIFIER_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import Verifier pipeline sub-modules via top-level packages
from claims import ClaimNormalizer
from routers import DomainValidator, QueryExpander
from adapters.registry import get_registry
from rerankers import CrossEncoderReranker
from nli import NLIEngine
from scorers import EvidenceScorer
from schemas.models import Passage, EvidenceItem, ClaimReport, VerdictLabel
from cache import SqliteCache


async def run_e2e_diagnostic(claim_text: str, domain: str = "general") -> None:
    # Force cache OFF for forensic verification
    os.environ["VERIFIER_CACHE_ENABLED"] = "false"
    cache = SqliteCache()
    await cache.invalidate(domain, claim_text)

    print("=" * 70)
    print("HALLUCIGUARD VERIFIER FORENSIC E2E DIAGNOSTIC")
    print("=" * 70)

    # STAGE 1: QUERY & EXPANSION
    print("\n" + "-" * 70)
    print("1. QUERY & EXPANSION")
    print("-" * 70)
    normalizer = ClaimNormalizer()
    normalized_claim = normalizer.normalize(claim_text)
    expander = QueryExpander()
    expanded_query_str = expander.expand(normalized_claim, domain)
    
    search_queries = [normalized_claim]
    if expanded_query_str and expanded_query_str != normalized_claim:
        search_queries.append(expanded_query_str)

    print(f"Original Claim  : {claim_text}")
    print(f"Normalized Claim: {normalized_claim}")
    print("Search Queries:")
    for idx, q in enumerate(search_queries, 1):
        print(f"  [{idx}] {q}")

    # STAGE 2: ROUTING & RETRIEVAL
    print("\n" + "-" * 70)
    print("2. DOMAIN ROUTING & ADAPTER RETRIEVAL")
    print("-" * 70)
    validator = DomainValidator()
    resolved_domain, _ = validator.validate(normalized_claim, domain)
    registry = get_registry()
    adapter = registry.get_adapter(resolved_domain)
    adapter_name = getattr(adapter, "name", adapter.__class__.__name__)
    
    print(f"Requested Domain: {domain}")
    print(f"Resolved Domain : {resolved_domain}")
    print(f"Selected Adapter: {adapter_name} ({adapter.__class__.__name__})")

    # Execute search queries asynchronously
    raw_passages: List[Passage] = []
    for q in search_queries:
        results = await adapter.search(q)
        raw_passages.extend(results)

    # Deduplicate passages by text/url
    unique_passages: List[Passage] = []
    seen = set()
    for p in raw_passages:
        key = p.url or p.snippet[:50]
        if key not in seen:
            seen.add(key)
            unique_passages.append(p)

    print(f"\nRaw Passages Retrieved: {len(unique_passages)}")
    for idx, p in enumerate(unique_passages, 1):
        print(f"\n  [{idx}] Title: {p.title}")
        print(f"      URL  : {p.url}")
        print(f"      Source: {p.source} (ID: {p.source_id})")
        print(f"      Snippet: {p.snippet[:140]}...")

    if not unique_passages:
        print("\n🚨 RETRIEVAL FAILED: No passages returned from external adapter!")
        print("Verdict: UNVERIFIED (Insufficient Evidence)")
        print("=" * 70)
        return

    # STAGE 3: BGE RERANKING & RELEVANCE SCORING
    print("\n" + "-" * 70)
    print("3. BGE RERANKING & RELEVANCE SCORING")
    print("-" * 70)
    reranker = CrossEncoderReranker()
    reranked_passages = reranker.rerank(normalized_claim, unique_passages, 5)

    print(f"{'Source / Title':<45} | {'BGE Relevance Score':<20}")
    print("-" * 70)
    for p in reranked_passages:
        title_str = (p.title[:42] + "...") if len(p.title) > 45 else p.title
        print(f"{title_str:<45} | {p.relevance_score:.4f}")

    # STAGE 4: RELEVANCE GATE
    print("\n" + "-" * 70)
    print("4. RELEVANCE GATE (Threshold >= 0.40)")
    print("-" * 70)
    accepted_passages = [
        p for p in reranked_passages
        if EvidenceScorer.relevance_weight(p.relevance_score) >= 0.40
    ]
    rejected_passages = [
        p for p in reranked_passages
        if p not in accepted_passages
    ]

    print(f"Accepted Passages: {len(accepted_passages)}")
    for p in accepted_passages:
        print(f"  ✅ [ACCEPT] {p.title} (Relevance: {p.relevance_score:.4f})")
    print(f"Rejected Passages: {len(rejected_passages)}")
    for p in rejected_passages:
        print(f"  ❌ [REJECT] {p.title} (Relevance: {p.relevance_score:.4f})")

    if not accepted_passages:
        print("\n🚨 RELEVANCE GATE FAILED: All passages rejected as off-topic!")
        print("Verdict: UNVERIFIED (No Relevant Evidence)")
        print("=" * 70)
        return

    # STAGE 5: DEBERTA NLI INFERENCE
    print("\n" + "-" * 70)
    print("5. DEBERTA NLI SOFTMAX PROBABILITIES")
    print("-" * 70)
    nli = NLIEngine()
    
    # Run batch classification
    evidences_list = [str(p.snippet) for p in accepted_passages]
    nli_results = nli.batch_classify(str(normalized_claim), evidences_list)

    nli_evidence_items: List[EvidenceItem] = []
    for idx, (p, r) in enumerate(zip(accepted_passages, nli_results), 1):
        ent = float(r.get("entailment_score", 0.0))
        con = float(r.get("contradiction_score", 0.0))
        neu = float(r.get("neutral_score", 0.0))
        label = str(r.get("label", "neutral")).upper()

        print(f"\nEvidence #{idx}: {p.title}")
        print(f"  URL          : {p.url}")
        print(f"  NLI Label    : {label}")
        print(f"  Entailment   : {ent:.4f}")
        print(f"  Contradiction: {con:.4f}")
        print(f"  Neutral      : {neu:.4f}")

        ev = EvidenceItem(
            source=p.source or "unknown",
            title=p.title or "Untitled",
            url=p.url or "",
            snippet=p.snippet[:300],
            publication_date=getattr(p, "publication_date", "2026-01-01") or "2026-01-01",
            entailment_score=ent,
            contradiction_score=con,
            neutral_score=neu,
            credibility_score=float(p.source_id and adapter.credibility_of(p.source_id) or 0.80),
            relevance_score=float(p.relevance_score or 0.50),
            entailment_label=r.get("label", "neutral"),
        )
        nli_evidence_items.append(ev)

    # STAGE 6: EVIDENCE CLASSIFICATION & DECISION GRADE SELECTION
    print("\n" + "-" * 70)
    print("6. EVIDENCE CLASSIFICATION")
    print("-" * 70)
    
    supporting_count = sum(1 for r in nli_results if "entailment" in str(r.get("label", "")).lower())
    contradicting_count = sum(1 for r in nli_results if "contradiction" in str(r.get("label", "")).lower())
    neutral_count = sum(1 for r in nli_results if "neutral" in str(r.get("label", "")).lower())
    irrelevant_count = len(rejected_passages)

    print(f"Supporting     : {supporting_count}")
    print(f"Contradicting  : {contradicting_count}")
    print(f"Neutral        : {neutral_count}")
    print(f"Irrelevant     : {irrelevant_count}")

    # STAGE 7: FINAL SCORING & VERDICT
    print("\n" + "-" * 70)
    print("7. FINAL VERDICT & CONFIDENCE SCORING")
    print("-" * 70)
    scorer = EvidenceScorer()
    scores_dict = scorer.score_evidence(normalized_claim, accepted_passages, nli_results, domain)

    verdict_val = scores_dict.get("verdict", VerdictLabel.UNVERIFIED)
    if hasattr(verdict_val, "value"):
        verdict_str = verdict_val.value.upper()
    else:
        verdict_str = str(verdict_val).upper()

    print(f"FINAL VERDICT       : {verdict_str}")
    print(f"Confidence Score    : {scores_dict.get('confidence_score', 0.0):.4f}")
    print(f"Support Score       : {scores_dict.get('support_score', 0.0):.4f}")
    print(f"Contradiction Score : {scores_dict.get('contradiction_score', 0.0):.4f}")
    print(f"Trust Score         : {scores_dict.get('trust_score', 0.0):.4f}")

    # STAGE 8: PROVENANCE
    print("\n" + "-" * 70)
    print("8. SYSTEM PROVENANCE")
    print("-" * 70)
    print(f"Actual External Sources : YES ({adapter_name})")
    print(f"Cache Status            : DISABLED (Bypassed)")
    print(f"NLI Model               : cross-encoder/nli-deberta-v3-base")
    print(f"Reranker Model          : BAAI/bge-reranker-large")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="HalluciGuard Verifier End-to-End Diagnostic Tool.")
    parser.add_argument("--claim", required=True, help="Claim text to verify.")
    parser.add_argument("--domain", default="general", help="Domain profile (general, healthcare, ai_research).")
    args = parser.parse_args()

    asyncio.run(run_e2e_diagnostic(args.claim, args.domain))


if __name__ == "__main__":
    main()
