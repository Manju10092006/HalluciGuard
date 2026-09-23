"""Live corrector smoke test: run CharacterRegenerator against real providers.

Prints NO secrets. Records provider_used, status, and failure_category.
"""
import asyncio
import os


def _load_env(path=".env"):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


_load_env()

from services.character_regenerator import CharacterRegenerator  # noqa: E402
from orchestration.schemas import (  # noqa: E402
    CorrectionRequest,
    ClaimReport,
    Evidence,
    VerdictLabel,
    EntailmentLabel,
)

ORIGINAL = "Java was created by Dennis Ritchie in 1972."


def _request():
    ev = Evidence(
        evidence_id="E1",
        title="Java history",
        source="Sun Microsystems",
        snippet="Java was developed by James Gosling at Sun Microsystems, first released in 1995.",
        entailment_label=EntailmentLabel.CONTRADICTION,
        entailment_score=0.95,
        credibility_score=0.95,
    )
    claim = ClaimReport(
        claim_id="c1",
        claim_text=ORIGINAL,
        verdict=VerdictLabel.CONTRADICTED,
        support_score=0.05,
        contradiction_score=0.95,
        confidence_score=0.95,
        evidence=[ev],
    )
    return CorrectionRequest(
        execution_id="exec-live-1",
        user_query="Who created Java?",
        original_response=ORIGINAL,
        claims_to_correct=[claim],
        claims_to_preserve=[],
        trusted_evidence=[],
        contradictory_evidence=[ev],
        correction_instructions="Repair the contradicted claim using evidence.",
    )


async def main():
    res = await CharacterRegenerator().regenerate(_request())
    print("status           :", res.status)
    print("provider_used    :", getattr(res, "provider_used", None))
    print("failure_category :", getattr(res, "failure_category", None))
    print("corrected[:200]  :", (getattr(res, "corrected_response", None) or "")[:200])


if __name__ == "__main__":
    asyncio.run(main())
