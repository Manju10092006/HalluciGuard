"""
HalluciGuard - Judge Agent Decision Engine
Determines the final trust action (ACCEPT, VERIFY_AGAIN, CORRECT, REJECT, ABSTAIN),
generates natural language reasoning, and constructs the structured output payload for the Corrector Agent.
"""

from typing import Dict, List, Any
from config import JudgeConfig, DEFAULT_CONFIG

class DecisionEngine:
    def __init__(self, config: JudgeConfig = DEFAULT_CONFIG):
        self.config = config

    def evaluate_decision(
        self,
        calibration_results: Dict[str, Any],
        evaluated_pairs: List[Dict[str, Any]],
        user_query: str,
        draft_response: str
    ) -> Dict[str, Any]:
        """
        Executes decision rules based on calibrated metrics and claim-evidence NLI outputs.
        """
        calibrated_conf = calibration_results.get("calibrated_confidence", 0.0)
        overall_contradiction = calibration_results.get("overall_contradiction", 0.0)
        overall_entailment = calibration_results.get("overall_entailment", 0.0)
        evidence_strength = calibration_results.get("evidence_strength", 0.0)
        severity = calibration_results.get("severity", "LOW")

        # Separate verified vs hallucinated / suspicious claims
        verified_claims = []
        hallucinated_claims = []
        trusted_evidence = []

        for pair in evaluated_pairs:
            claim = pair.get("claim", "")
            evidence = pair.get("evidence", "")
            nli = pair.get("nli_scores", {})
            top_rel = pair.get("top_relation", "neutral")

            if evidence:
                trusted_evidence.append({
                    "claim": claim,
                    "evidence_snippet": evidence,
                    "evidence_source": pair.get("source", "Verifier Knowledge Base"),
                    "confidence": pair.get("evidence_confidence", 0.8),
                    "nli_relation": top_rel
                })

            if top_rel == "contradiction" or nli.get("contradiction", 0) > 0.40:
                hallucinated_claims.append({
                    "claim": claim,
                    "contradiction_score": nli.get("contradiction", 0.0),
                    "reason": f"Contradicted by retrieved evidence with score {nli.get('contradiction', 0.0):.2f}"
                })
            elif top_rel == "entailment" or nli.get("entailment", 0) > 0.60:
                verified_claims.append(claim)

        # Decision Logic Core
        decision = "ABSTAIN"
        next_action = "NONE"
        reason = ""
        explanation = ""

        # Case 1: No evidence found at all
        if not evaluated_pairs or evidence_strength == 0.0:
            if calibrated_conf < 0.30:
                decision = "ABSTAIN"
                next_action = "Request domain specialist or external search fallback"
                reason = "Insufficient evidence retrieved to verify response claims."
                explanation = "The framework could not locate authoritative ground-truth evidence to confirm or refute the response."
            else:
                decision = "VERIFY_AGAIN"
                next_action = "Re-route to Verifier Agent with expanded search parameters"
                reason = "Low verifier evidence coverage requires secondary verification pass."
                explanation = "The response confidence is borderline and missing external evidence support."

        # Case 2: Direct Contradictions detected (Hallucination)
        elif hallucinated_claims or overall_contradiction >= self.config.correct_contradiction_threshold:
            if overall_contradiction >= self.config.reject_contradiction_threshold or severity == "CRITICAL":
                decision = "REJECT"
                next_action = "Block draft response; issue immediate warning or safe default reply"
                reason = f"Severe hallucination detected with contradiction index {overall_contradiction:.2f}."
                explanation = f"Critical factual conflicts were identified across {len(hallucinated_claims)} key claims."
            else:
                decision = "CORRECT"
                next_action = "Forward hallucinated claims and trusted evidence to Corrector Agent"
                reason = f"Hallucinations identified in {len(hallucinated_claims)} claim(s); correction required."
                explanation = "The response contains specific factual inconsistencies that can be minimally edited using verified evidence."

        # Case 3: High Confidence & Strong Entailment (Factually Faithful)
        elif calibrated_conf >= self.config.accept_confidence_threshold and overall_entailment >= 0.50:
            decision = "ACCEPT"
            next_action = "Pass response to user directly without modification"
            reason = f"Response is fully verified with calibrated confidence {calibrated_conf:.2f}."
            explanation = "All key atomic claims are strongly supported by ground-truth evidence."

        # Case 4: Moderate Confidence / Needs Re-Verification
        elif self.config.verify_again_confidence_range[0] <= calibrated_conf <= self.config.verify_again_confidence_range[1]:
            decision = "VERIFY_AGAIN"
            next_action = "Request targeted sub-claim verification from Verifier Agent"
            reason = f"Calibrated confidence {calibrated_conf:.2f} is in verification boundary zone."
            explanation = "Some claims have neutral or partial evidence alignment requiring deeper domain search."

        # Default fallback
        else:
            decision = "CORRECT" if hallucinated_claims else "VERIFY_AGAIN"
            next_action = "Forward to Corrector Agent for safety review" if decision == "CORRECT" else "Re-verify claims"
            reason = f"Calibrated score {calibrated_conf:.2f} warrants precautionary processing."
            explanation = "Automated rules flagged minor faithfulness uncertainty."

        # Package payload for downstream Corrector Agent & Observability
        corrector_payload = {
            "judge_decision": decision,
            "severity": severity,
            "reason": reason,
            "explanation": explanation,
            "next_action": next_action,
            "hallucinated_claims": hallucinated_claims,
            "verified_claims": verified_claims,
            "trusted_evidence": trusted_evidence,
            "original_draft_response": draft_response,
            "user_query": user_query,
            "corrector_guidance": (
                f"Replace or edit {len(hallucinated_claims)} hallucinated claim(s) using provided trusted evidence. "
                "Preserve original non-hallucinated content."
            ) if decision == "CORRECT" else "No correction needed."
        }

        return corrector_payload
