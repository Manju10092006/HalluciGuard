package com.example.halluciguard.planner

import com.example.halluciguard.model.AtomicClaim
import com.example.halluciguard.model.ClaimStatus
import com.example.halluciguard.model.ClaimToEdit
import com.example.halluciguard.model.CorrectionPlan
import com.example.halluciguard.model.EvidencePassage
import com.example.halluciguard.model.JudgeVerificationPayload

class CorrectionPlanner {

    fun planCorrection(payload: JudgeVerificationPayload): CorrectionPlan {
        val preserved = mutableListOf<AtomicClaim>()
        val toRewrite = mutableListOf<ClaimToEdit>()
        val unsupported = mutableListOf<AtomicClaim>()

        val evidenceMap = payload.supportingEvidence.associateBy { it.id }
        val contradictionMap = payload.contradictionEvidence.associateBy { it.id }

        for (claim in payload.claims) {
            when (claim.status) {
                ClaimStatus.VERIFIED -> {
                    preserved.add(claim)
                }
                ClaimStatus.HALLUCINATED, ClaimStatus.CONTRADICTED -> {
                    val matchedEvidence = claim.evidenceIds.mapNotNull { evidenceMap[it] }.ifEmpty {
                        payload.supportingEvidence
                    }
                    val matchedContradictions = claim.evidenceIds.mapNotNull { contradictionMap[it] }.ifEmpty {
                        payload.contradictionEvidence
                    }

                    if (matchedEvidence.isEmpty() && matchedContradictions.isEmpty()) {
                        unsupported.add(claim)
                    } else {
                        toRewrite.add(
                            ClaimToEdit(
                                claim = claim,
                                reason = if (claim.status == ClaimStatus.CONTRADICTED) {
                                    "Claim directly contradicts verified evidence."
                                } else {
                                    "Claim contains hallucinated/unverified assertions."
                                },
                                matchedEvidence = matchedEvidence,
                                matchedContradictions = matchedContradictions
                            )
                        )
                    }
                }
                ClaimStatus.INSUFFICIENT_EVIDENCE, ClaimStatus.UNCERTAIN -> {
                    unsupported.add(claim)
                }
            }
        }

        val strategyNotes = buildString {
            append("Preserved ${preserved.size} verified claims. ")
            append("Targeted ${toRewrite.size} claims for evidence-grounded rewriting. ")
            append("Identified ${unsupported.size} claims with insufficient evidence for transparent disclaimer replacement.")
        }

        return CorrectionPlan(
            preservedClaims = preserved,
            claimsToRewrite = toRewrite,
            unsupportedClaims = unsupported,
            strategyNotes = strategyNotes
        )
    }
}
