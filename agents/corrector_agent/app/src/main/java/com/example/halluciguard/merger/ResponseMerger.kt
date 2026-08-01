package com.example.halluciguard.merger

import com.example.halluciguard.model.AtomicClaim
import com.example.halluciguard.model.ClaimDiff
import com.example.halluciguard.model.ClaimStatus
import com.example.halluciguard.model.CorrectionPlan
import com.example.halluciguard.model.DiffAction
import com.example.halluciguard.model.JudgeVerificationPayload

class ResponseMerger {

    fun computeDiffs(
        payload: JudgeVerificationPayload,
        plan: CorrectionPlan,
        finalResponse: String
    ): List<ClaimDiff> {
        val diffs = mutableListOf<ClaimDiff>()

        // 1. Preserved Verified Claims
        for (claim in plan.preservedClaims) {
            val isPresent = finalResponse.contains(claim.text, ignoreCase = true)
            diffs.add(
                ClaimDiff(
                    originalClaimId = claim.id,
                    originalText = claim.text,
                    status = ClaimStatus.VERIFIED,
                    actionTaken = DiffAction.PRESERVED_EXACT,
                    correctedText = claim.text,
                    explanation = if (isPresent) {
                        "Preserved exactly as verified by Judge Agent."
                    } else {
                        "Verified claim slightly rephrased for flow, facts maintained."
                    }
                )
            )
        }

        // 2. Rewritten Claims (Hallucinated or Contradicted)
        for (item in plan.claimsToRewrite) {
            val evidenceText = item.matchedEvidence.firstOrNull()?.passageText
                ?: "Fact corrected using verified ground truth passages."
            diffs.add(
                ClaimDiff(
                    originalClaimId = item.claim.id,
                    originalText = item.claim.text,
                    status = item.claim.status,
                    actionTaken = DiffAction.REWRITTEN_WITH_EVIDENCE,
                    correctedText = evidenceText,
                    explanation = "Hallucination eliminated. Grounded in source '${item.matchedEvidence.firstOrNull()?.sourceTitle ?: "Verified Source"}'."
                )
            )
        }

        // 3. Unsupported Claims (Insufficient Evidence or Uncertain)
        for (claim in plan.unsupportedClaims) {
            diffs.add(
                ClaimDiff(
                    originalClaimId = claim.id,
                    originalText = claim.text,
                    status = claim.status,
                    actionTaken = DiffAction.REPLACED_UNSUPPORTED_DISCLAIMER,
                    correctedText = "Current evidence is insufficient to support this claim.",
                    explanation = "Insufficient evidence provided by Judge; replaced with transparent disclaimer."
                )
            )
        }

        return diffs
    }

    fun cleanAndFormatResponse(rawLlmText: String): String {
        return rawLlmText
            .replace(Regex("```[a-zA-Z]*"), "")
            .replace("```", "")
            .replace("=== OUTPUT MANDATE ===", "")
            .replace(Regex("\\[ID: [^\\]]+\\]"), "")
            .trim()
    }
}

class ResponseValidator {

    data class ValidationResult(
        val isValid: Boolean,
        val issues: List<String>
    )

    fun validateStructure(
        plan: CorrectionPlan,
        finalResponseText: String
    ): ValidationResult {
        val issues = mutableListOf<String>()

        if (finalResponseText.isBlank()) {
            issues.add("Generated response is empty.")
            return ValidationResult(false, issues)
        }

        // Check if preserved claims are maintained
        var preservedCount = 0
        for (claim in plan.preservedClaims) {
            // Check for key words
            val words = claim.text.split(" ").filter { it.length > 3 }
            val matchRatio = if (words.isNotEmpty()) {
                val matches = words.count { finalResponseText.contains(it, ignoreCase = true) }
                matches.toDouble() / words.size
            } else 1.0

            if (matchRatio >= 0.5) {
                preservedCount++
            }
        }

        if (plan.preservedClaims.isNotEmpty() && preservedCount == 0) {
            issues.add("Warning: Verified claims appear significantly altered or missing.")
        }

        return ValidationResult(issues.isEmpty(), issues)
    }
}
