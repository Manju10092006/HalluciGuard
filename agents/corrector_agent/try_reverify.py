"""
try_reverify.py — exercise the ReVerifier stage IN ISOLATION.

The ReVerifier's job: after a correction, RE-EXTRACT the claims from the
CORRECTED answer and verify each against the evidence — so a NEWLY introduced
error (e.g. the creator got fixed but the year is now wrong) is caught, instead
of only re-checking the originally-flagged claim.

This tester skips the Corrector and feeds a corrected answer straight into the
ReVerifier, so you can watch it catch a secondary hallucination.

Run from agents/corrector_agent:

    # The corrected answer fixed the creator but INTRODUCED a wrong year (1991).
    # Evidence says 1995 -> ReVerifier should FAIL on the year.
    python try_reverify.py \
        --query "Who created Java?" \
        --corrected "Java was created by James Gosling in 1991." \
        --evidence "Java was created by James Gosling and his team at Sun Microsystems, first released in 1995."

    # A clean correction -> VERIFIED
    python try_reverify.py \
        --query "Who created Java?" \
        --corrected "Java was created by James Gosling in 1995." \
        --evidence "Java was created by James Gosling and his team at Sun Microsystems, first released in 1995."

This is offline/deterministic — no API key or network needed.
"""

from __future__ import annotations

import argparse
import json
import sys

from app.models import (
    JudgeVerificationPayload,
    AtomicClaim,
    ClaimStatus,
    EvidencePassage,
    CorrectionPlan,
)
from app.reverifier import ReVerifier


def main() -> None:
    ap = argparse.ArgumentParser(description="Test the ReVerifier stage in isolation")
    ap.add_argument("--query", default="", help="original user question")
    ap.add_argument("--corrected", required=True, help="the corrected answer to re-verify")
    ap.add_argument("--original", default="", help="the pre-correction draft (optional context)")
    ap.add_argument("--evidence", action="append", metavar="TEXT", default=[],
                    help="supporting evidence passage (repeatable)")
    ap.add_argument("--contradiction", action="append", metavar="TEXT", default=[],
                    help="contradiction evidence passage (repeatable)")
    ap.add_argument("--json", action="store_true", help="dump the raw contract dict")
    args = ap.parse_args()

    supporting = [
        EvidencePassage(id=f"E{i:03d}", sourceTitle="user", passageText=t, url="", relevanceScore=0.95)
        for i, t in enumerate(args.evidence, 1)
    ]
    contradiction = [
        EvidencePassage(id=f"X{i:03d}", sourceTitle="user", passageText=t, url="", relevanceScore=0.95)
        for i, t in enumerate(args.contradiction, 1)
    ]

    # Minimal payload carrying the original context + evidence the ReVerifier reuses.
    payload = JudgeVerificationPayload(
        query=args.query,
        originalResponse=args.original or args.corrected,
        claims=[AtomicClaim(id="C001", text=args.original or args.corrected,
                            status=ClaimStatus.CONTRADICTED, confidenceScore=0.9, evidenceIds=[])],
        supportingEvidence=supporting,
        contradictionEvidence=contradiction,
        trustScore=0.3,
        sourceMetadata=[],
        correctionInstructions="",
    )

    # Minimal empty plan — the ReVerifier derives its verdict from the corrected
    # TEXT + evidence, not from what the plan claims was done.
    plan = CorrectionPlan(
        preservedClaims=[], claimsToRewrite=[], unsupportedClaims=[], strategyNotes=""
    )

    reverifier = ReVerifier()
    # The ReVerifier re-extracts from the corrected answer and verifies vs evidence.
    result = reverifier.reverify(payload=payload, plan=plan, corrected_answer=args.corrected)
    contract = result.to_contract() if hasattr(result, "to_contract") else result

    if args.json:
        print(json.dumps(contract, indent=2, default=str))
        return

    print("=" * 74)
    print("REVERIFIER (isolated)")
    print("=" * 74)
    print(f"  query          : {args.query}")
    print(f"  corrected      : {args.corrected}")
    for e in supporting:
        print(f"  evidence(+)    : {e.passageText}")
    for e in contradiction:
        print(f"  evidence(-)    : {e.passageText}")
    print("-" * 74)
    status = contract.get("status")
    print(f"  STATUS         : {status}")
    print(f"  overall_conf   : {contract.get('overall_confidence')}")
    claims = contract.get("claims") or contract.get("failed_claims") or []
    print(f"  re-extracted claims ({len(claims)}):")
    for c in claims:
        print(f"    - [{str(c.get('verdict','?')):12}] conf={c.get('confidence','?')}  {c.get('text','')}")
    print("=" * 74)
    if status == "FAILED":
        print("  -> ReVerifier BLOCKED the answer (secondary error caught). Routes back to correction.")
    else:
        print("  -> ReVerifier ACCEPTED the corrected answer.")


if __name__ == "__main__":
    main()
