"""Regression (audit H4/M5/M11): a text-keyed Verifier cache hit must rebind the
stored report's claim_id/claim_text to the CURRENT request, so the graph's exact
claim-coverage contract holds on warm / cross-pass hits and same-text duplicates.
"""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VERIFIER_DIR = PROJECT_ROOT / "agents" / "verifier_agent"
for _p in (str(PROJECT_ROOT), str(VERIFIER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from api.pipeline import VerificationPipeline  # noqa: E402
from schemas.models import (  # noqa: E402
    ClaimReport,
    VerifierInputV2,
    SuspiciousClaim,
    VerdictLabel,
)


def _stale_cached_report() -> dict:
    # A report as it would have been stored by an EARLIER run under a different id.
    return ClaimReport(
        claim_id="stale-id-from-prior-run",
        claim_text="Paris is the capital of France.",
        evidence=[],
        support_score=0.9,
        contradiction_score=0.0,
        trust_score=0.8,
        verdict=VerdictLabel.VERIFIED,
    ).model_dump()


def test_cache_hit_rebinds_claim_id_and_text_to_request(monkeypatch):
    pipe = VerificationPipeline()
    pipe.cache_enabled = True

    async def fake_get(domain, text):
        return _stale_cached_report()

    monkeypatch.setattr(pipe.cache, "get", fake_get)
    payload = VerifierInputV2(
        query_id="q", domain="general",
        suspicious_claims=[SuspiciousClaim(claim_id="c1", text="Paris is the capital of France.")],
    )
    res = asyncio.run(pipe.verify(payload))
    assert len(res.claim_evidence) == 1
    assert res.claim_evidence[0].claim_id == "c1"
    assert res.claim_evidence[0].claim_text == "Paris is the capital of France."


def test_same_text_duplicate_claims_keep_distinct_ids(monkeypatch):
    pipe = VerificationPipeline()
    pipe.cache_enabled = True

    async def fake_get(domain, text):
        return _stale_cached_report()

    monkeypatch.setattr(pipe.cache, "get", fake_get)
    payload = VerifierInputV2(
        query_id="q", domain="general",
        suspicious_claims=[
            SuspiciousClaim(claim_id="c1", text="Paris is the capital of France."),
            SuspiciousClaim(claim_id="c2", text="Paris is the capital of France."),
        ],
    )
    res = asyncio.run(pipe.verify(payload))
    assert [r.claim_id for r in res.claim_evidence] == ["c1", "c2"]
