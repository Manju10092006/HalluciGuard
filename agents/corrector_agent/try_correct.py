"""
try_correct.py — quick manual tester for the Corrector + ReVerifier pipeline.

Lets you feed your OWN query / draft answer / claims / evidence on the command
line (or from a JSON file) and see the corrected answer plus the reverification
outcome, without hand-writing the full JudgeVerificationPayload.

Run from THIS directory (agents/corrector_agent) so `import app...` resolves and
doesn't collide with the repo-root Gradio app.py:

    cd agents/corrector_agent

    # 1) Simplest: one contradicted claim, one supporting-evidence passage
    python try_correct.py \
        --query "Who created Java?" \
        --answer "Java was created by Snehith in 1995." \
        --contradicted "Java was created by Snehith" \
        --evidence "Java was created by James Gosling and his team at Sun Microsystems; first released in 1995."

    # 2) Keep a verified claim while fixing a contradicted one (co-location case)
    python try_correct.py \
        --query "Who created Java?" \
        --answer "Java was created by Snehith in 1995." \
        --contradicted "Java was created by Snehith" \
        --verified "Java was released in 1995" \
        --evidence "Java was created by James Gosling and his team at Sun Microsystems in 1995."

    # 3) Full control: supply a complete JudgeVerificationPayload JSON
    python try_correct.py --payload my_payload.json

First run downloads Qwen2.5-1.5B-Instruct (~3GB) unless the fine-tuned
checkpoint under ./training/ exists; later runs are cached.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

from app.models import (
    JudgeVerificationPayload,
    AtomicClaim,
    ClaimStatus,
    EvidencePassage,
)
from app.orchestrator import CorrectorOrchestrator


def build_payload(args) -> JudgeVerificationPayload:
    if args.payload:
        with open(args.payload, encoding="utf-8") as fh:
            return JudgeVerificationPayload(**json.load(fh))

    if not args.answer:
        sys.exit("Provide --answer (the draft to correct), or --payload FILE.")

    claims = []
    evidence = []
    ev_ids = []

    # Evidence passages (shared as supporting evidence for the claims).
    for i, ev in enumerate(args.evidence or [], 1):
        eid = f"E{i:03d}"
        ev_ids.append(eid)
        evidence.append(EvidencePassage(
            id=eid, sourceTitle="user-supplied", passageText=ev,
            url="", relevanceScore=0.95,
        ))

    # Contradicted claims -> must be corrected.
    for i, c in enumerate(args.contradicted or [], 1):
        claims.append(AtomicClaim(
            id=f"C{i:03d}", text=c, status=ClaimStatus.CONTRADICTED,
            confidenceScore=0.95, evidenceIds=ev_ids,
        ))
    # Verified claims -> must be preserved.
    for i, c in enumerate(args.verified or [], 1):
        claims.append(AtomicClaim(
            id=f"V{i:03d}", text=c, status=ClaimStatus.VERIFIED,
            confidenceScore=0.95, evidenceIds=ev_ids,
        ))

    if not claims:
        sys.exit("Provide at least one --contradicted or --verified claim, or --payload FILE.")

    return JudgeVerificationPayload(
        query=args.query or "",
        originalResponse=args.answer,
        claims=claims,
        supportingEvidence=evidence,
        contradictionEvidence=[],
        trustScore=args.trust,
        sourceMetadata=[],
        correctionInstructions=args.instructions,
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description="Manual Corrector+ReVerifier tester")
    ap.add_argument("--query", help="original user question")
    ap.add_argument("--answer", help="draft answer to correct")
    ap.add_argument("--contradicted", action="append", metavar="CLAIM",
                    help="a claim judged CONTRADICTED (repeatable)")
    ap.add_argument("--verified", action="append", metavar="CLAIM",
                    help="a claim judged VERIFIED, must be preserved (repeatable)")
    ap.add_argument("--evidence", action="append", metavar="TEXT",
                    help="a supporting evidence passage (repeatable)")
    ap.add_argument("--instructions", default="Correct the contradicted claims using the supporting evidence; preserve verified content.",
                    help="correction instructions for the Corrector")
    ap.add_argument("--trust", type=float, default=0.3, help="initial trust score [0-1]")
    ap.add_argument("--payload", help="path to a full JudgeVerificationPayload JSON")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="dump the full result object as JSON")
    args = ap.parse_args()

    payload = build_payload(args)
    orch = CorrectorOrchestrator()
    result = await orch.executeCorrectionPipeline(payload, maxRetries=args.max_retries)

    if args.json:
        print(json.dumps(result.model_dump(), indent=2, default=str))
        return

    print("=" * 74)
    print("INPUT")
    print("=" * 74)
    print(f"  query   : {payload.query}")
    print(f"  draft   : {payload.originalResponse}")
    for c in payload.claims:
        print(f"  claim   : [{c.status.value:14}] {c.text}")
    for e in payload.supportingEvidence:
        print(f"  evidence: {e.passageText}")
    print()
    print("=" * 74)
    print("RESULT")
    print("=" * 74)
    print(f"  final answer      : {result.finalResponse}")
    print(f"  fully approved    : {result.isFullyApproved}")
    print(f"  unresolved        : {result.isTerminatedUnresolved}")
    print(f"  attempts          : {result.attemptsCount}")
    print(f"  trust  {result.initialTrustScore:.2f} -> {result.finalTrustScore:.2f}")
    print(f"  model             : {result.providerUsed}/{result.modelUsed}")
    print(f"  latency (ms)      : {result.totalLatencyMs:.0f}")
    if result.traceLogs:
        print("  trace stages      : " + " -> ".join(
            getattr(t, "stage", str(t)) if not isinstance(t, dict) else t.get("stage", "?")
            for t in result.traceLogs))
    print("=" * 74)


if __name__ == "__main__":
    asyncio.run(main())
