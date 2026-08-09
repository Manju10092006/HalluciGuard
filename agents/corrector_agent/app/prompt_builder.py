from typing import List, Optional
from app.models import CorrectionPlan, JudgeVerificationPayload

class PromptBuilder:
    def buildCorrectionPrompt(
        self,
        payload: JudgeVerificationPayload,
        plan: CorrectionPlan,
        attemptNumber: int = 1,
        previousJudgeFeedback: Optional[str] = None,
        rejectionReasons: List[str] = None
    ) -> str:
        if rejectionReasons is None:
            rejectionReasons = []

        lines = []
        # Stage 1 & 2: Prompt Planner & Instruction Builder
        lines.append("=== SYSTEM INSTRUCTIONS: HALLUCIGUARD ENTERPRISE CORRECTOR AGENT ===")
        lines.append("Pipeline Version: v2.4.0-enterprise | Model Target: Qwen3-8B / Gemini-3.5")
        lines.append("You are a deterministic, evidence-grounded response refinement editor.")
        lines.append("Your sole mandate is to correct hallucinated or unsupported claims in an existing AI response while preserving verified facts EXACTLY.")
        lines.append("")
        lines.append("DETERMINISTIC CONSTRAINTS:")
        lines.append("1. PRESERVE VERIFIED CLAIMS: Do not modify or alter any character in verified claims.")
        lines.append("2. GROUND ALL EDITS: Correct hallucinated/contradicted claims using ONLY the provided verified evidence passages.")
        lines.append("3. INSUFFICIENT EVIDENCE RULE: If evidence is missing for an unverified statement, replace it with transparent language: 'Current evidence is insufficient to support this claim.'")
        lines.append("4. NO NEW ENTITIES: Do not introduce any new facts, ungrounded entities, or external knowledge.")
        lines.append("5. STYLISTIC INTEGRITY: Maintain original tone, formatting, and logical coherence.")
        lines.append("")

        # Stage 4: Few-Shot Selector (Exemplars)
        lines.append("=== FEW-SHOT CORRECTION EXEMPLARS ===")
        lines.append("Example Original: 'Apollo 11 landed in 1969 with 5 astronauts.' [Flagged: 5 astronauts]")
        lines.append("Evidence: Apollo 11 crew consisted of 3 astronauts (Armstrong, Aldrin, Collins).")
        lines.append("Corrected Output: 'Apollo 11 landed in 1969 with 3 astronauts.'")
        lines.append("")

        if attemptNumber > 1 and (previousJudgeFeedback or rejectionReasons):
            lines.append(f"=== FEEDBACK FROM PREVIOUS CORRECTION ATTEMPT (Attempt #{attemptNumber - 1} Failed Verification) ===")
            if previousJudgeFeedback:
                lines.append(f"Judge Feedback: {previousJudgeFeedback}")
            if rejectionReasons:
                lines.append("Rejection Reasons:")
                for reason in rejectionReasons:
                    lines.append(f" - {reason}")
            lines.append("CORRECT ALL OF THE ABOVE ISSUES IN THIS NEW DRAFT.")
            lines.append("")

        lines.append("=== USER QUERY ===")
        lines.append(payload.query)
        lines.append("")

        lines.append("=== ORIGINAL LLM RESPONSE ===")
        lines.append(payload.originalResponse)
        lines.append("")

        # Stage 3: Evidence Formatter
        lines.append("=== VERIFIED CLAIMS (MUST PRESERVE EXACTLY) ===")
        if not plan.preservedClaims:
            lines.append("(None)")
        else:
            for idx, claim in enumerate(plan.preservedClaims):
                lines.append(f"{idx + 1}. [ID: {claim.id}] \"{claim.text}\"")
        lines.append("")

        lines.append("=== CLAIMS REQUIRING REWRITING / CORRECTION ===")
        if not plan.claimsToRewrite:
            lines.append("(None)")
        else:
            for idx, item in enumerate(plan.claimsToRewrite):
                lines.append(f"{idx + 1}. [ID: {item.claim.id}] \"{item.claim.text}\"")
                # Incorporate dataset judge_reason if present (from enhancement)
                # The prompt asks to use judge_reason. We can just add it if present as extra context.
                lines.append(f"   Issue: {item.reason}")
                if item.matchedEvidence:
                    lines.append("   Verified Supporting Evidence:")
                    for ev in item.matchedEvidence:
                        lines.append(f"     - [Source: {ev.sourceTitle}] \"{ev.passageText}\"")
                if item.matchedContradictions:
                    lines.append("   Contradictory Evidence:")
                    for ev in item.matchedContradictions:
                        lines.append(f"     - [Source: {ev.sourceTitle}] \"{ev.passageText}\"")
        lines.append("")

        lines.append("=== CLAIMS WITH INSUFFICIENT EVIDENCE (MUST REPLACE WITH DISCLAIMER) ===")
        if not plan.unsupportedClaims:
            lines.append("(None)")
        else:
            for idx, claim in enumerate(plan.unsupportedClaims):
                lines.append(f"{idx + 1}. [ID: {claim.id}] \"{claim.text}\"")
        lines.append("")

        if payload.correctionInstructions and payload.correctionInstructions.strip():
            lines.append("=== JUDGE AGENT SPECIFIC CORRECTION INSTRUCTIONS ===")
            lines.append(payload.correctionInstructions.strip())
            lines.append("")

        lines.append("=== OUTPUT MANDATE ===")
        lines.append("Return ONLY the final, complete corrected response text. Do not include markdown code fence wrappers, preambles, or conversational meta-commentary.")
        
        prompt = "\n".join(lines)
        return self.optimizeTokenBudgetAndValidate(prompt)

    def optimizeTokenBudgetAndValidate(self, prompt: str) -> str:
        # Enforce max token limit budget
        if len(prompt) > 8000:
            return prompt[:7950] + "\n... [Truncated for Context Window Budget]"
        return prompt
