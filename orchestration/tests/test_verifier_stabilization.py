"""
HalluciGuard Step 3 — Verifier Stabilization Regression Suite.

Verifies:
  1. SqliteCache auto-initialization (no 'no such table' errors).
  2. Wikipedia adapter evidence retrieval (non-zero passages returned).
  3. Single-claim end-to-end verification pipeline ("Paris is the capital of France.").
  4. DeBERTa NLI entailment scoring & ClaimReport contract integrity.
"""
from __future__ import annotations

import os
import sys
import unittest

# Setup import paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VERIFIER_DIR = os.path.join(PROJECT_ROOT, "agents", "verifier_agent")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if VERIFIER_DIR not in sys.path:
    sys.path.insert(0, VERIFIER_DIR)

from cache.sqlite_cache import SqliteCache
from adapters.general import GeneralAdapter
from api.pipeline import VerificationPipeline
from schemas.models import SuspiciousClaim, VerifierInputV2, VerdictLabel, EntailmentLabel


class TestVerifierStabilization(unittest.IsolatedAsyncioTestCase):

    async def test_sqlite_cache_auto_initialization(self) -> None:
        """Verify SqliteCache auto-initializes table without throwing 'no such table'."""
        db_path = os.path.join(PROJECT_ROOT, "scratch", "test_verification_cache.db")
        if os.path.exists(db_path):
            os.remove(db_path)

        cache = SqliteCache(db_path=db_path, ttl_seconds=60)
        res = await cache.get("general", "test_query_key")
        self.assertIsNone(res)

        await cache.set("general", "test_query_key", {"status": "ok"})
        cached = await cache.get("general", "test_query_key")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.get("status"), "ok")

        if os.path.exists(db_path):
            os.remove(db_path)

    async def test_wikipedia_adapter_retrieval(self) -> None:
        """Verify GeneralAdapter retrieves Wikipedia evidence passages for known claim."""
        adapter = GeneralAdapter()
        passages = await adapter.search("Paris is the capital of France.", k=3)
        self.assertGreaterEqual(len(passages), 1, "Wikipedia adapter should return at least 1 passage")

        p0 = passages[0]
        self.assertTrue(p0.url.startswith("https://en.wikipedia.org/wiki/"))
        self.assertTrue(len(p0.snippet) > 0)
        self.assertEqual(p0.source, "wikipedia")

    async def test_single_claim_end_to_end_verification(self) -> None:
        """Verify single simple claim 'Paris is the capital of France.' reaches NLI and produces VERIFIED verdict."""
        pipeline = VerificationPipeline()
        await pipeline.cache.invalidate("general", "Paris is the capital of France.")
        await pipeline.cache.invalidate("general", "c1")

        payload = VerifierInputV2(
            query_id="stabilization-test-fresh",
            domain="general",
            suspicious_claims=[
                SuspiciousClaim(claim_id="c1", text="Paris is the capital of France.")
            ]
        )

        result = await pipeline.verify(payload)

        self.assertEqual(result.domain, "general")
        self.assertGreaterEqual(result.retrieved_sources, 1, "Should retrieve at least 1 evidence source")
        self.assertEqual(len(result.claim_evidence), 1, "Should return 1 ClaimReport")

        claim_rep = result.claim_evidence[0]
        self.assertEqual(claim_rep.claim_text, "Paris is the capital of France.")
        self.assertEqual(claim_rep.verdict, VerdictLabel.VERIFIED)
        self.assertGreater(len(claim_rep.evidence), 0, "ClaimReport evidence list must not be empty")

        ev0 = claim_rep.evidence[0]
        self.assertIsNotNone(ev0.url)
        self.assertIsNotNone(ev0.snippet)
        self.assertGreater(len(ev0.snippet), 0)
        self.assertIn(ev0.entailment_label, [EntailmentLabel.ENTAILMENT, EntailmentLabel.NEUTRAL, EntailmentLabel.CONTRADICTION])
        self.assertGreaterEqual(ev0.entailment_score, 0.0)
        self.assertGreater(ev0.credibility_score, 0.0)


if __name__ == "__main__":
    unittest.main()
