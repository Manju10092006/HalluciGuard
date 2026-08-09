import asyncio
from app.models import JudgeVerificationPayload, AtomicClaim, EvidencePassage, ClaimStatus
from app.orchestrator import CorrectorOrchestrator

async def test():
    orchestrator = CorrectorOrchestrator()
    payload = JudgeVerificationPayload(
        query="Who invented Python?",
        originalResponse="Elon Musk",
        claims=[
            AtomicClaim(
                id="c1",
                text="Elon Musk",
                status=ClaimStatus.HALLUCINATED,
                confidenceScore=0.9,
                evidenceIds=["e1"]
            )
        ],
        supportingEvidence=[
            EvidencePassage(
                id="e1",
                sourceTitle="Test",
                passageText="Guido"
            )
        ],
        trustScore=0.5,
        correctionInstructions="Fix it"
    )
    
    try:
        res = await orchestrator.executeCorrectionPipeline(payload)
        print("Success:", res)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
