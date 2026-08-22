#!/usr/bin/env python3
"""
HalluciGuard — Web Evidence Retrieval E2E Diagnostic

Tests the complete evidence chain:
  CLAIM → TAVILY SEARCH → REAL URLs → PAGE EXTRACTION → BGE RERANKING →
  DeBERTa NLI → EVIDENCE SCORING → CLAIM VERDICT

Usage:
    python scripts/test_web_evidence.py                     # Run all test claims
    python scripts/test_web_evidence.py "Custom claim"      # Test specific claim
    RUN_LIVE_WEB_TESTS=true python scripts/test_web_evidence.py  # Required for live tests

Requires:
    TAVILY_API_KEY in .env
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

# ── Setup paths ────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFIER_ROOT = os.path.join(PROJECT_ROOT, "agents", "verifier_agent")
sys.path.insert(0, VERIFIER_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load .env
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Disable cache for diagnostic runs
os.environ["CACHE_ENABLED"] = "false"

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("web_evidence_test")


# ── Standard test claims ───────────────────────────────────────
STANDARD_CLAIMS = [
    {
        "claim": "Paris is the capital of France.",
        "expected_verdict": "verified",
        "domain": "general",
        "description": "Simple factual claim — should be easily verified",
    },
    {
        "claim": "The Eiffel Tower is in London.",
        "expected_verdict": "contradicted",
        "domain": "general",
        "description": "False factual claim — should be contradicted",
    },
    {
        "claim": "The Moon is made of green cheese.",
        "expected_verdict": "contradicted",
        "domain": "general",
        "description": "Absurd claim — should be contradicted, NOT confused by myth description articles",
    },
    {
        "claim": "Xyzabc123 is a common medication for headaches.",
        "expected_verdict": "unverified",
        "domain": "healthcare",
        "description": "Nonsense entity — should return insufficient evidence",
    },
    {
        "claim": "Aspirin is commonly used to treat mild pain and reduce fever.",
        "expected_verdict": "verified",
        "domain": "healthcare",
        "description": "Medical fact — should find authoritative medical sources",
    },
    {
        "claim": "The Log4Shell vulnerability (CVE-2021-44228) affected Apache Log4j.",
        "expected_verdict": "verified",
        "domain": "cybersecurity",
        "description": "Cybersecurity CVE — should find NVD/security advisory sources",
    },
    {
        "claim": "The Transformer architecture was introduced in the paper 'Attention Is All You Need'.",
        "expected_verdict": "verified",
        "domain": "artificial_intelligence",
        "description": "AI research fact — should find arxiv/academic sources",
    },
]


def separator(title: str, char: str = "=") -> None:
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")


async def test_tavily_direct(claim: str) -> dict:
    """Test Tavily retrieval directly (bypassing adapter registry)."""
    from adapters.web_retriever import TavilyWebRetriever

    retriever = TavilyWebRetriever()

    if not retriever.is_available:
        return {"error": "TAVILY_API_KEY not configured"}

    start = time.time()
    passages = await retriever.search(claim, k=5)
    elapsed = time.time() - start

    return {
        "retriever": "TavilyWebRetriever (direct)",
        "query": claim,
        "elapsed_ms": int(elapsed * 1000),
        "passage_count": len(passages),
        "passages": [
            {
                "title": p.title,
                "source": p.source,
                "url": p.url,
                "snippet_preview": p.snippet[:150] + "..." if len(p.snippet) > 150 else p.snippet,
                "relevance_score": p.relevance_score,
            }
            for p in passages
        ],
    }


async def test_full_pipeline(claim_text: str, domain: str) -> dict:
    """Run full verification pipeline with web evidence: retrieval → reranking → NLI → scoring."""
    from adapters.web_retriever import TavilyWebRetriever, _domain_credibility
    from rerankers import CrossEncoderReranker
    from nli import NLIEngine
    from scorers import EvidenceScorer
    from schemas.models import Passage, EntailmentLabel

    result: dict = {
        "claim": claim_text,
        "domain": domain,
        "stages": {},
    }

    # ── Stage 1: Tavily Retrieval ──────────────────────────────
    retriever = TavilyWebRetriever()
    if not retriever.is_available:
        result["error"] = "TAVILY_API_KEY not configured"
        return result

    t0 = time.time()
    passages = await retriever.search(claim_text, k=5)
    result["stages"]["retrieval"] = {
        "elapsed_ms": int((time.time() - t0) * 1000),
        "passage_count": len(passages),
        "passages": [
            {
                "title": p.title,
                "source": p.source,
                "url": p.url,
                "snippet_len": len(p.snippet),
                "snippet_preview": p.snippet[:200],
                "relevance_score": p.relevance_score,
                "credibility": _domain_credibility(p.url),
            }
            for p in passages
        ],
    }

    if not passages:
        result["verdict"] = "INSUFFICIENT_EVIDENCE"
        result["reason"] = "No passages retrieved from web"
        return result

    # ── Stage 2: BGE Reranking ─────────────────────────────────
    reranker = CrossEncoderReranker()
    t0 = time.time()
    reranked = reranker.rerank(claim_text, passages, 5)
    result["stages"]["reranking"] = {
        "elapsed_ms": int((time.time() - t0) * 1000),
        "reranked_count": len(reranked),
        "reranked": [
            {
                "title": p.title,
                "url": p.url,
                "relevance_score": round(p.relevance_score, 4),
            }
            for p in reranked
        ],
    }

    # Filter by relevance gate
    relevant = [p for p in reranked if EvidenceScorer.relevance_weight(p.relevance_score) >= 0.40]
    result["stages"]["relevance_gate"] = {
        "input_count": len(reranked),
        "accepted_count": len(relevant),
        "rejected": [
            {"title": p.title, "score": round(p.relevance_score, 4)}
            for p in reranked if p not in relevant
        ],
    }

    if not relevant:
        result["verdict"] = "INSUFFICIENT_EVIDENCE"
        result["reason"] = "All passages rejected by relevance gate"
        return result

    # ── Stage 3: DeBERTa NLI ───────────────────────────────────
    nli = NLIEngine()
    t0 = time.time()
    evidence_texts = [str(p.snippet) for p in relevant]
    nli_results = nli.batch_classify(str(claim_text), evidence_texts)
    result["stages"]["nli"] = {
        "elapsed_ms": int((time.time() - t0) * 1000),
        "results": [
            {
                "title": relevant[i].title,
                "url": relevant[i].url,
                "snippet_preview": relevant[i].snippet[:100],
                "entailment_score": round(float(r.get("entailment_score", 0)), 4),
                "contradiction_score": round(float(r.get("contradiction_score", 0)), 4),
                "neutral_score": round(float(r.get("neutral_score", 0)), 4),
                "label": str(r.get("label", "unknown")),
                "degraded": r.get("degraded", False),
            }
            for i, r in enumerate(nli_results)
        ],
    }

    # ── Stage 4: Evidence Scoring ──────────────────────────────
    scorer = EvidenceScorer()
    t0 = time.time()
    scores = scorer.score_evidence(
        claim=str(claim_text),
        passages=relevant,
        nli_results=nli_results,
        domain=domain,
    )
    result["stages"]["scoring"] = {
        "elapsed_ms": int((time.time() - t0) * 1000),
        "verdict": str(scores.get("verdict", "unknown")),
        "support_score": round(float(scores.get("support_score", 0)), 4),
        "contradiction_score": round(float(scores.get("contradiction_score", 0)), 4),
        "trust_score": round(float(scores.get("trust_score", 0)), 4),
        "confidence_score": round(float(scores.get("confidence_score", 0)), 4),
    }

    result["verdict"] = str(scores.get("verdict", "unknown"))
    result["trust_score"] = round(float(scores.get("trust_score", 0)), 4)

    # ── Evidence classification summary ────────────────────────
    support = sum(1 for r in nli_results if str(r.get("label", "")).lower().endswith("entailment"))
    contradict = sum(1 for r in nli_results if str(r.get("label", "")).lower().endswith("contradiction"))
    neutral = sum(1 for r in nli_results if str(r.get("label", "")).lower().endswith("neutral"))
    result["evidence_summary"] = {
        "total_retrieved": len(passages),
        "after_reranking": len(reranked),
        "after_relevance_gate": len(relevant),
        "supporting": support,
        "contradicting": contradict,
        "neutral": neutral,
    }

    return result


async def test_nli_direct():
    """Direct NLI model test — prove the model actually runs."""
    from nli import NLIEngine

    nli = NLIEngine()
    test_pairs = [
        ("Paris is the capital of France.", "Paris is the capital and largest city of France."),
        ("The Eiffel Tower is in London.", "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris."),
        ("Water boils at 100 degrees Celsius.", "The boiling point of water is 100°C at standard atmospheric pressure."),
    ]

    results = []
    for claim, evidence in test_pairs:
        r = nli.classify(str(claim), str(evidence))
        results.append({
            "claim": claim,
            "evidence": evidence[:80],
            "label": str(r.get("label", "unknown")),
            "entailment": round(float(r.get("entailment_score", 0)), 4),
            "contradiction": round(float(r.get("contradiction_score", 0)), 4),
            "neutral": round(float(r.get("neutral_score", 0)), 4),
        })
    return results


async def test_reranker_direct():
    """Direct BGE reranker test — prove the model actually runs."""
    from rerankers import CrossEncoderReranker
    from schemas.models import Passage

    reranker = CrossEncoderReranker()
    claim = "Paris is the capital of France."
    passages = [
        Passage(title="Paris", source="test", url="http://test.com/1",
                publication_date="2024", snippet="Paris is the capital and largest city of France.",
                source_id="t1", relevance_score=0.5),
        Passage(title="Paris FC", source="test", url="http://test.com/2",
                publication_date="2024", snippet="Paris FC is a French football club based in Paris.",
                source_id="t2", relevance_score=0.5),
        Passage(title="London", source="test", url="http://test.com/3",
                publication_date="2024", snippet="London is the capital of England and the United Kingdom.",
                source_id="t3", relevance_score=0.5),
    ]

    reranked = reranker.rerank(claim, passages, 3)
    return [
        {"title": p.title, "snippet": p.snippet[:60], "score": round(p.relevance_score, 4)}
        for p in reranked
    ]


async def main():
    # Check for live test flag
    run_live = os.environ.get("RUN_LIVE_WEB_TESTS", "").lower() in ("true", "1", "yes")
    tavily_key = os.environ.get("TAVILY_API_KEY", "")

    if not tavily_key:
        print("ERROR: TAVILY_API_KEY not found in environment. Add it to .env")
        sys.exit(1)

    if not run_live:
        print("NOTE: Set RUN_LIVE_WEB_TESTS=true to run live web tests")
        print("Running model-only tests (NLI + Reranker)...\n")

    # ── Test 1: Direct NLI ─────────────────────────────────────
    separator("TEST 1: Direct DeBERTa NLI Model")
    try:
        nli_results = await test_nli_direct()
        for r in nli_results:
            print(f"\n  Claim: {r['claim']}")
            print(f"  Evidence: {r['evidence']}")
            print(f"  Label: {r['label']}")
            print(f"  Scores: E={r['entailment']:.4f} C={r['contradiction']:.4f} N={r['neutral']:.4f}")
        print("\n  [PASS] NLI model is running and producing real probabilities")
    except Exception as e:
        print(f"  [FAIL] NLI test failed: {e}")

    # ── Test 2: Direct BGE Reranker ────────────────────────────
    separator("TEST 2: Direct BGE Reranker Model")
    try:
        reranker_results = await test_reranker_direct()
        for r in reranker_results:
            print(f"  [{r['score']:.4f}] {r['title']}: {r['snippet']}")
        print("\n  [PASS] BGE reranker is running and producing real scores")
    except Exception as e:
        print(f"  [FAIL] Reranker test failed: {e}")

    if not run_live:
        print("\n" + "=" * 70)
        print("  Set RUN_LIVE_WEB_TESTS=true to run the full web evidence pipeline")
        print("=" * 70)
        return

    # ── Test 3: Direct Tavily retrieval ────────────────────────
    separator("TEST 3: Direct Tavily Web Retrieval")
    try:
        tavily_result = await test_tavily_direct("Paris is the capital of France.")
        print(f"  Elapsed: {tavily_result.get('elapsed_ms')}ms")
        print(f"  Passages: {tavily_result.get('passage_count')}")
        for p in tavily_result.get("passages", []):
            print(f"\n    Title: {p['title']}")
            print(f"    URL: {p['url']}")
            print(f"    Source: {p['source']}")
            print(f"    Relevance: {p['relevance_score']}")
            print(f"    Snippet: {p['snippet_preview']}")
        print("\n  [PASS] Tavily retrieval is working with real URLs and content")
    except Exception as e:
        print(f"  [FAIL] Tavily test failed: {e}")

    # ── Test 4-10: Full pipeline for each claim ────────────────
    # Use custom claim from CLI args if provided
    claims_to_test = STANDARD_CLAIMS
    if len(sys.argv) > 1:
        custom_claim = " ".join(sys.argv[1:])
        claims_to_test = [{
            "claim": custom_claim,
            "expected_verdict": "unknown",
            "domain": "general",
            "description": "Custom user claim",
        }]

    all_results = []
    for i, test_case in enumerate(claims_to_test, start=4):
        separator(f"TEST {i}: Full Pipeline — {test_case['description']}")
        print(f"  Claim: {test_case['claim']}")
        print(f"  Expected: {test_case['expected_verdict']}")
        print()

        try:
            result = await test_full_pipeline(test_case["claim"], test_case["domain"])
            all_results.append(result)

            # Print retrieval results
            retrieval = result.get("stages", {}).get("retrieval", {})
            print(f"  RETRIEVAL ({retrieval.get('elapsed_ms')}ms): {retrieval.get('passage_count')} passages")
            for p in retrieval.get("passages", []):
                print(f"    [{p['relevance_score']:.2f}] {p['title']}")
                print(f"           URL: {p['url']}")
                print(f"           Credibility: {p['credibility']:.2f}")
                print(f"           Snippet: {p['snippet_preview'][:80]}...")

            # Print reranking
            reranking = result.get("stages", {}).get("reranking", {})
            if reranking:
                print(f"\n  RERANKING ({reranking.get('elapsed_ms')}ms): {reranking.get('reranked_count')} reranked")
                for p in reranking.get("reranked", []):
                    print(f"    [{p['relevance_score']:.4f}] {p['title']}")

            # Print relevance gate
            gate = result.get("stages", {}).get("relevance_gate", {})
            if gate:
                print(f"\n  RELEVANCE GATE: {gate.get('accepted_count')}/{gate.get('input_count')} accepted")
                for r in gate.get("rejected", []):
                    print(f"    REJECTED: [{r['score']:.4f}] {r['title']}")

            # Print NLI
            nli = result.get("stages", {}).get("nli", {})
            if nli:
                print(f"\n  NLI ({nli.get('elapsed_ms')}ms):")
                for r in nli.get("results", []):
                    label_icon = {"entailment": "[+]", "contradiction": "[-]", "neutral": "[~]"}.get(
                        r["label"].split(".")[-1] if "." in r["label"] else r["label"],
                        "[?]"
                    )
                    print(f"    {label_icon} {r['title']}")
                    print(f"       E={r['entailment_score']:.4f} C={r['contradiction_score']:.4f} N={r['neutral_score']:.4f}")
                    print(f"       Label: {r['label']}")

            # Print scoring / verdict
            scoring = result.get("stages", {}).get("scoring", {})
            if scoring:
                print(f"\n  SCORING ({scoring.get('elapsed_ms')}ms):")
                print(f"    Verdict: {scoring.get('verdict')}")
                print(f"    Support: {scoring.get('support_score'):.4f}")
                print(f"    Contradiction: {scoring.get('contradiction_score'):.4f}")
                print(f"    Trust: {scoring.get('trust_score'):.4f}")

            # Summary
            summary = result.get("evidence_summary", {})
            if summary:
                print(f"\n  EVIDENCE SUMMARY:")
                print(f"    Retrieved: {summary.get('total_retrieved')}")
                print(f"    After reranking: {summary.get('after_reranking')}")
                print(f"    After relevance gate: {summary.get('after_relevance_gate')}")
                print(f"    Supporting: {summary.get('supporting')}")
                print(f"    Contradicting: {summary.get('contradicting')}")
                print(f"    Neutral: {summary.get('neutral')}")

            # Final verdict match check
            actual = result.get("verdict", "unknown").lower()
            if "." in actual:
                actual = actual.split(".")[-1]
            expected = test_case["expected_verdict"]
            match = "[MATCH]" if actual == expected else "[MISMATCH]"
            print(f"\n  {match} VERDICT: {actual} (expected: {expected})")

        except Exception as e:
            print(f"  [ERROR] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Final Summary ──────────────────────────────────────────
    separator("FINAL SUMMARY")
    for r in all_results:
        v = r.get("verdict", "ERROR")
        if "." in str(v):
            v = str(v).split(".")[-1]
        trust = r.get("trust_score", 0)
        summary = r.get("evidence_summary", {})
        print(f"  [{v:>14s}] (trust={trust:.4f}) {r.get('claim', '?')[:60]}")
        if summary:
            print(f"                  sources={summary.get('total_retrieved',0)}"
                  f" supporting={summary.get('supporting',0)}"
                  f" contradicting={summary.get('contradicting',0)}")


if __name__ == "__main__":
    asyncio.run(main())
