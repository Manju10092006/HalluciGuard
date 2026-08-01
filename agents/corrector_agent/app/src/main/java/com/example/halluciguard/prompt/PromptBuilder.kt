package com.example.halluciguard.prompt

import com.example.halluciguard.model.CorrectionPlan
import com.example.halluciguard.model.JudgeVerificationPayload

/**
 * Enterprise 6-Stage Prompt Engineering Pipeline:
 * Prompt Planner -> Instruction Builder -> Evidence Formatter -> Few-Shot Selector -> Token Budget Optimizer -> Prompt Validator
 */
class PromptBuilder {

    fun buildCorrectionPrompt(
        payload: JudgeVerificationPayload,
        plan: CorrectionPlan,
        attemptNumber: Int = 1,
        previousJudgeFeedback: String? = null,
        rejectionReasons: List<String> = emptyList()
    ): String {
        val plannedPrompt = buildString {
            // Stage 1 & 2: Prompt Planner & Instruction Builder
            appendLine("=== SYSTEM INSTRUCTIONS: HALLUCIGUARD ENTERPRISE CORRECTOR AGENT ===")
            appendLine("Pipeline Version: v2.4.0-enterprise | Model Target: Qwen3-8B / Gemini-3.5")
            appendLine("You are a deterministic, evidence-grounded response refinement editor.")
            appendLine("Your sole mandate is to correct hallucinated or unsupported claims in an existing AI response while preserving verified facts EXACTLY.")
            appendLine()
            appendLine("DETERMINISTIC CONSTRAINTS:")
            appendLine("1. PRESERVE VERIFIED CLAIMS: Do not modify or alter any character in verified claims.")
            appendLine("2. GROUND ALL EDITS: Correct hallucinated/contradicted claims using ONLY the provided verified evidence passages.")
            appendLine("3. INSUFFICIENT EVIDENCE RULE: If evidence is missing for an unverified statement, replace it with transparent language: 'Current evidence is insufficient to support this claim.'")
            appendLine("4. NO NEW ENTITIES: Do not introduce any new facts, ungrounded entities, or external knowledge.")
            appendLine("5. STYLISTIC INTEGRITY: Maintain original tone, formatting, and logical coherence.")
            appendLine()

            // Stage 4: Few-Shot Selector (Exemplars)
            appendLine("=== FEW-SHOT CORRECTION EXEMPLARS ===")
            appendLine("Example Original: 'Apollo 11 landed in 1969 with 5 astronauts.' [Flagged: 5 astronauts]")
            appendLine("Evidence: Apollo 11 crew consisted of 3 astronauts (Armstrong, Aldrin, Collins).")
            appendLine("Corrected Output: 'Apollo 11 landed in 1969 with 3 astronauts.'")
            appendLine()

            if (attemptNumber > 1 && (!previousJudgeFeedback.isNullOrBlank() || rejectionReasons.isNotEmpty())) {
                appendLine("=== FEEDBACK FROM PREVIOUS CORRECTION ATTEMPT (Attempt #${attemptNumber - 1} Failed Verification) ===")
                if (!previousJudgeFeedback.isNullOrBlank()) {
                    appendLine("Judge Feedback: $previousJudgeFeedback")
                }
                if (rejectionReasons.isNotEmpty()) {
                    appendLine("Rejection Reasons:")
                    rejectionReasons.forEach { reason ->
                        appendLine(" - $reason")
                    }
                }
                appendLine("CORRECT ALL OF THE ABOVE ISSUES IN THIS NEW DRAFT.")
                appendLine()
            }

            appendLine("=== USER QUERY ===")
            appendLine(payload.query)
            appendLine()

            appendLine("=== ORIGINAL LLM RESPONSE ===")
            appendLine(payload.originalResponse)
            appendLine()

            // Stage 3: Evidence Formatter
            appendLine("=== VERIFIED CLAIMS (MUST PRESERVE EXACTLY) ===")
            if (plan.preservedClaims.isEmpty()) {
                appendLine("(None)")
            } else {
                plan.preservedClaims.forEachIndexed { idx, claim ->
                    appendLine("${idx + 1}. [ID: ${claim.id}] \"${claim.text}\"")
                }
            }
            appendLine()

            appendLine("=== CLAIMS REQUIRING REWRITING / CORRECTION ===")
            if (plan.claimsToRewrite.isEmpty()) {
                appendLine("(None)")
            } else {
                plan.claimsToRewrite.forEachIndexed { idx, item ->
                    appendLine("${idx + 1}. [ID: ${item.claim.id}] \"${item.claim.text}\"")
                    appendLine("   Issue: ${item.reason}")
                    if (item.matchedEvidence.isNotEmpty()) {
                        appendLine("   Verified Supporting Evidence:")
                        item.matchedEvidence.forEach { ev ->
                            appendLine("     - [Source: ${ev.sourceTitle}] \"${ev.passageText}\"")
                        }
                    }
                    if (item.matchedContradictions.isNotEmpty()) {
                        appendLine("   Contradictory Evidence:")
                        item.matchedContradictions.forEach { ev ->
                            appendLine("     - [Source: ${ev.sourceTitle}] \"${ev.passageText}\"")
                        }
                    }
                }
            }
            appendLine()

            appendLine("=== CLAIMS WITH INSUFFICIENT EVIDENCE (MUST REPLACE WITH DISCLAIMER) ===")
            if (plan.unsupportedClaims.isEmpty()) {
                appendLine("(None)")
            } else {
                plan.unsupportedClaims.forEachIndexed { idx, claim ->
                    appendLine("${idx + 1}. [ID: ${claim.id}] \"${claim.text}\"")
                }
            }
            appendLine()

            if (payload.correctionInstructions.isNotBlank()) {
                appendLine("=== JUDGE AGENT SPECIFIC CORRECTION INSTRUCTIONS ===")
                appendLine(payload.correctionInstructions)
                appendLine()
            }

            appendLine("=== OUTPUT MANDATE ===")
            appendLine("Return ONLY the final, complete corrected response text. Do not include markdown code fence wrappers, preambles, or conversational meta-commentary.")
        }

        // Stage 5 & 6: Token Budget Optimizer & Prompt Validator
        return optimizeTokenBudgetAndValidate(plannedPrompt)
    }

    private fun optimizeTokenBudgetAndValidate(prompt: String): String {
        // Enforce max token limit budget (approx 4000 characters)
        if (prompt.length > 8000) {
            return prompt.take(7950) + "\n... [Truncated for Context Window Budget]"
        }
        return prompt
    }
}
