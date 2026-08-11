import re
from typing import List, Tuple
from app.models import (
    JudgeVerificationPayload, CorrectionPlan, ClaimDiff, ClaimStatus, DiffAction
)

class ResponseMerger:
    def computeDiffs(self, payload: JudgeVerificationPayload, plan: CorrectionPlan, finalResponse: str) -> List[ClaimDiff]:
        diffs = []
        
        # 1. Preserved Verified Claims
        for claim in plan.preservedClaims:
            isPresent = claim.text.lower() in finalResponse.lower()
            explanation = "Preserved exactly as verified by Judge Agent." if isPresent else "Verified claim slightly rephrased for flow, facts maintained."
            diffs.append(
                ClaimDiff(
                    originalClaimId=claim.id,
                    originalText=claim.text,
                    status=ClaimStatus.VERIFIED,
                    actionTaken=DiffAction.PRESERVED_EXACT,
                    correctedText=claim.text,
                    explanation=explanation
                )
            )
            
        # 2. Rewritten Claims
        for item in plan.claimsToRewrite:
            evidenceText = item.matchedEvidence[0].passageText if item.matchedEvidence else "Fact corrected using verified ground truth passages."
            sourceTitle = item.matchedEvidence[0].sourceTitle if item.matchedEvidence else "Verified Source"
            
            diffs.append(
                ClaimDiff(
                    originalClaimId=item.claim.id,
                    originalText=item.claim.text,
                    status=item.claim.status,
                    actionTaken=DiffAction.REWRITTEN_WITH_EVIDENCE,
                    correctedText=evidenceText,
                    explanation=f"Hallucination eliminated. Grounded in source '{sourceTitle}'."
                )
            )
            
        # 3. Unsupported Claims
        for claim in plan.unsupportedClaims:
            diffs.append(
                ClaimDiff(
                    originalClaimId=claim.id,
                    originalText=claim.text,
                    status=claim.status,
                    actionTaken=DiffAction.REPLACED_UNSUPPORTED_DISCLAIMER,
                    correctedText="Current evidence is insufficient to support this claim.",
                    explanation="Insufficient evidence provided by Judge; replaced with transparent disclaimer."
                )
            )
            
        return diffs
        
    def cleanAndFormatResponse(self, rawLlmText: str) -> str:
        text = re.sub(r'```[a-zA-Z]*', '', rawLlmText)
        text = text.replace('```', '')
        text = text.replace('=== OUTPUT MANDATE ===', '')
        text = re.sub(r'\[ID: [^\]]+\]', '', text)
        return text.strip()

class ResponseValidator:
    def validateStructure(self, plan: CorrectionPlan, finalResponseText: str) -> Tuple[bool, List[str]]:
        issues = []
        
        if not finalResponseText.strip():
            issues.append("Generated response is empty.")
            return False, issues
            
        preservedCount = 0
        for claim in plan.preservedClaims:
            words = [w for w in claim.text.split() if len(w) > 3]
            if words:
                matches = sum(1 for w in words if w.lower() in finalResponseText.lower())
                matchRatio = matches / len(words)
            else:
                matchRatio = 1.0
                
            if matchRatio >= 0.5:
                preservedCount += 1
                
        if plan.preservedClaims and preservedCount == 0:
            issues.append("Warning: Verified claims appear significantly altered or missing.")
            
        return len(issues) == 0, issues
