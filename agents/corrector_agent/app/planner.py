from app.models import (
    CorrectionPlan, JudgeVerificationPayload, AtomicClaim, ClaimToEdit, ClaimStatus
)

class CorrectionPlanner:
    def planCorrection(self, payload: JudgeVerificationPayload) -> CorrectionPlan:
        preserved = []
        toRewrite = []
        unsupported = []

        evidenceMap = {ev.id: ev for ev in payload.supportingEvidence}
        contradictionMap = {ev.id: ev for ev in payload.contradictionEvidence}

        for claim in payload.claims:
            if claim.status == ClaimStatus.VERIFIED:
                preserved.append(claim)
            elif claim.status in [ClaimStatus.HALLUCINATED, ClaimStatus.CONTRADICTED]:
                matchedEvidence = [evidenceMap[eid] for eid in claim.evidenceIds if eid in evidenceMap]
                if not matchedEvidence:
                    matchedEvidence = payload.supportingEvidence
                    
                matchedContradictions = [contradictionMap[eid] for eid in claim.evidenceIds if eid in contradictionMap]
                if not matchedContradictions:
                    matchedContradictions = payload.contradictionEvidence

                if not matchedEvidence and not matchedContradictions:
                    unsupported.append(claim)
                else:
                    reason = "Claim directly contradicts verified evidence." if claim.status == ClaimStatus.CONTRADICTED else "Claim contains hallucinated/unverified assertions."
                    toRewrite.append(
                        ClaimToEdit(
                            claim=claim,
                            reason=reason,
                            matchedEvidence=matchedEvidence,
                            matchedContradictions=matchedContradictions
                        )
                    )
            elif claim.status in [ClaimStatus.INSUFFICIENT_EVIDENCE, ClaimStatus.UNCERTAIN]:
                unsupported.append(claim)

        strategyNotes = (
            f"Preserved {len(preserved)} verified claims. "
            f"Targeted {len(toRewrite)} claims for evidence-grounded rewriting. "
            f"Identified {len(unsupported)} claims with insufficient evidence for transparent disclaimer replacement."
        )

        return CorrectionPlan(
            preservedClaims=preserved,
            claimsToRewrite=toRewrite,
            unsupportedClaims=unsupported,
            strategyNotes=strategyNotes
        )
