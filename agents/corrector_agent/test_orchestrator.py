import asyncio
import sys
import os

# Add the parent directory to sys.path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import (
    JudgeVerificationPayload, AtomicClaim, ClaimStatus, EvidencePassage, JudgeVerificationResult
)
from app.orchestrator import CorrectorOrchestrator

# Mock the Judge to always fail, forcing the retry limit to be reached and triggering the fallback logic
class MockJudge:
    def verifyCorrectedResponse(self, payload, plan, processed_text, current_attempt):
        return JudgeVerificationResult(
            isApproved=False,
            trustScore=0.1,
            verifiedClaimsCount=0,
            remainingHallucinationsCount=len(payload.claims),
            feedback="Failed to incorporate evidence.",
            rejectionReasons=["LLM generated garbage."]
        )

# Mock Model Client to just return garbage
class MockModelClient:
    def generate_correction(self, prompt):
        return "Garbage LLM output"

async def run_tests():
    orchestrator = CorrectorOrchestrator()
    orchestrator.judge = MockJudge()
    orchestrator.model_client = MockModelClient()

    # Shared evidence
    ev_guido = EvidencePassage(
        id="ev1", sourceTitle="Python History", passageText="Python was created by Guido van Rossum, and first released on February 20, 1991."
    )
    ev_verified = EvidencePassage(
        id="ev2", sourceTitle="Fact Source", passageText="The sky is blue."
    )

    tests = [
        {
            "name": "1. Full Evidence (The Reported Bug)",
            "payload": JudgeVerificationPayload(
                query="Who created Python?",
                originalResponse="Python was created by Elon Musk.",
                claims=[
                    AtomicClaim(id="c1", text="Python was created by Elon Musk.", status=ClaimStatus.HALLUCINATED, confidenceScore=0.9, evidenceIds=["ev1"])
                ],
                supportingEvidence=[ev_guido],
                trustScore=0.5,
                correctionInstructions=""
            )
        },
        {
            "name": "2. Zero Evidence",
            "payload": JudgeVerificationPayload(
                query="What is the meaning of life?",
                originalResponse="The meaning of life is 42.",
                claims=[
                    AtomicClaim(id="c1", text="The meaning of life is 42.", status=ClaimStatus.INSUFFICIENT_EVIDENCE, confidenceScore=0.9, evidenceIds=[])
                ],
                supportingEvidence=[],
                trustScore=0.5,
                correctionInstructions=""
            )
        },
        {
            "name": "3. Mixed Multi-Claim (Verified + Rewritten)",
            "payload": JudgeVerificationPayload(
                query="Tell me about Python and the sky.",
                originalResponse="The sky is blue. Python was created by Elon Musk.",
                claims=[
                    AtomicClaim(id="c1", text="The sky is blue.", status=ClaimStatus.VERIFIED, confidenceScore=0.9, evidenceIds=["ev2"]),
                    AtomicClaim(id="c2", text="Python was created by Elon Musk.", status=ClaimStatus.HALLUCINATED, confidenceScore=0.9, evidenceIds=["ev1"])
                ],
                supportingEvidence=[ev_verified, ev_guido],
                trustScore=0.5,
                correctionInstructions=""
            )
        },
        {
            "name": "4. Mixed Multi-Claim (Evidence + Unsupported)",
            "payload": JudgeVerificationPayload(
                query="Who created Python and what is the meaning of life?",
                originalResponse="Python was created by Elon Musk. The meaning of life is 42.",
                claims=[
                    AtomicClaim(id="c1", text="Python was created by Elon Musk.", status=ClaimStatus.HALLUCINATED, confidenceScore=0.9, evidenceIds=["ev1"]),
                    AtomicClaim(id="c2", text="The meaning of life is 42.", status=ClaimStatus.INSUFFICIENT_EVIDENCE, confidenceScore=0.9, evidenceIds=[])
                ],
                supportingEvidence=[ev_guido],
                trustScore=0.5,
                correctionInstructions=""
            )
        },
        {
            "name": "5. Fully Verified",
            "payload": JudgeVerificationPayload(
                query="What color is the sky?",
                originalResponse="The sky is blue.",
                claims=[
                    AtomicClaim(id="c1", text="The sky is blue.", status=ClaimStatus.VERIFIED, confidenceScore=0.9, evidenceIds=["ev2"])
                ],
                supportingEvidence=[ev_verified],
                trustScore=0.5,
                correctionInstructions=""
            )
        }
    ]

    for test in tests:
        print(f"--- TEST CASE: {test['name']} ---")
        # Run with maxRetries=1 to save time
        result = await orchestrator.executeCorrectionPipeline(test["payload"], maxRetries=1)
        print(f"Final Corrected Response:\n\"{result.finalResponse}\"\n")

if __name__ == "__main__":
    asyncio.run(run_tests())
