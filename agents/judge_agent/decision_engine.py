"""
HalluciGuard - Enterprise Decision Intelligence Engine
Synthesizes Domain Policy, Evidence Intelligence, Source Consensus, Contradiction Taxonomy,
Claim Criticality, and Calibrated Confidence to generate arbitration decisions and payloads.
"""

from typing import Dict, List, Any
from config import JudgeConfig, DEFAULT_CONFIG
from domain_policies import DomainPolicy
from contradiction_analyzer import ContradictionTaxonomyAnalyzer
from negotiation_engine import MultiAgentNegotiationEngine
from audit_trail import DecisionAuditEngine

class DecisionIntelligenceEngine:
    def __init__(self, config: JudgeConfig = DEFAULT_CONFIG):
        self.config = config
        self.contradiction_analyzer = ContradictionTaxonomyAnalyzer()
        self.negotiation_engine = MultiAgentNegotiationEngine()
        self.audit_engine = DecisionAuditEngine()

    def evaluate_decision(
        self,
        calibration_results: Dict[str, Any],
        evidence_intel: Dict[str, Any],
        consensus_data: Dict[str, Any],
        memory_data: Dict[str, Any],
        criticality_data: Dict[str, Any],
        domain_policy: DomainPolicy,
        user_query: str,
        draft_response: str
    ) -> Dict[str, Any]:
        """
        Synthesizes multi-dimensional evidence signals into final arbitration decision.
        """
        calibrated_conf = calibration_results.get("calibrated_confidence", 0.0)
        overall_contradiction = calibration_results.get("overall_contradiction", 0.0)
        overall_entailment = calibration_results.get("overall_entailment", 0.0)
        severity = calibration_results.get("severity", "LOW")

        processed_pairs = evidence_intel.get("processed_pairs", [])
        completeness = evidence_intel.get("evidence_completeness", "MISSING_ALL_EVIDENCE")
        authority = evidence_intel.get("overall_authority", 0.0)

        # Analyze Contradiction & Claim-Level Hallucination Scores
        hallucinated_claims = []
        verified_claims = []
        trusted_evidence = []
        all_claim_scores = []
        primary_contradiction = {"has_contradiction": False, "taxonomy_type": "NO_CONTRADICTION", "risk_weight": 0.0}

        for pair in processed_pairs:
            claim = pair.get("claim", "")
            evidence = pair.get("evidence", pair.get("evidence_snippet", ""))
            nli = pair.get("nli_scores", {})
            top_rel = pair.get("top_relation", "neutral")

            entail_score = nli.get("entailment", 0.0)
            contra_score = nli.get("contradiction", 0.0)
            neutral_score = nli.get("neutral", 0.0)

            # Claim-level hallucination score formulation:
            # H_score = 0.50 * Contradiction + 0.30 * (1 - Entailment) + 0.20 * Detector_Prob
            claim_hallucination_score = round(min(1.0, max(0.0, 
                0.50 * contra_score + 0.30 * (1.0 - entail_score) + 0.20 * calibration_results.get("metrics", {}).get("detector_hallucination_prob", 0.3)
            )), 4)

            claim_info = {
                "claim": claim,
                "hallucination_score": claim_hallucination_score,
                "faithfulness_score": round(1.0 - claim_hallucination_score, 4),
                "entailment_score": entail_score,
                "contradiction_score": contra_score,
                "neutral_score": neutral_score,
                "nli_relation": top_rel
            }
            all_claim_scores.append(claim_info)

            if evidence:
                trusted_evidence.append({
                    "claim": claim,
                    "evidence_snippet": evidence,
                    "evidence_source": pair.get("source", "Verifier Knowledge Base"),
                    "authority_score": pair.get("authority_score", 0.8),
                    "freshness_score": pair.get("freshness_score", 0.85),
                    "confidence": pair.get("evidence_confidence", 0.8),
                    "nli_relation": top_rel
                })

            taxonomy = self.contradiction_analyzer.classify_contradiction(claim, evidence, nli)
            
            if taxonomy["has_contradiction"] or claim_hallucination_score >= 0.35:
                if taxonomy["risk_weight"] > primary_contradiction["risk_weight"]:
                    primary_contradiction = taxonomy

                hallucinated_claims.append({
                    "claim": claim,
                    "hallucination_score": claim_hallucination_score,
                    "contradiction_score": contra_score,
                    "taxonomy_type": taxonomy.get("taxonomy_type", "UNVERIFIED_HALLUCINATION"),
                    "risk_weight": max(taxonomy.get("risk_weight", 0.5), claim_hallucination_score),
                    "explanation": taxonomy.get("explanation", f"Claim flagged with hallucination score {claim_hallucination_score:.2f} due to weak evidence entailment.")
                })
            elif top_rel == "entailment" or entail_score >= 0.55:
                verified_claims.append(claim)

        # ---------------------------------------------------------
        # Decision Arbitration Logic Core
        # ---------------------------------------------------------
        decision = "ABSTAIN"
        next_action = "NONE"
        reason = ""
        explanation = ""
        negotiation_protocol = None

        # Case 1: Missing Evidence / Insufficient Authority
        if not processed_pairs or completeness == "MISSING_ALL_EVIDENCE" or authority < domain_policy.min_required_authority * 0.5:
            decision = "ABSTAIN"
            next_action = "Request external domain specialist or ground-truth knowledge repository fallback."
            reason = f"Insufficient grounding in {domain_policy.domain_name} domain (Authority: {authority:.2f} < Min: {domain_policy.min_required_authority:.2f})."
            explanation = "Grounding evidence was either absent or failed domain authority requirements."

            negotiation_protocol = self.negotiation_engine.generate_verifier_instruction(
                domain_policy=domain_policy,
                evidence_completeness=completeness,
                unverified_claims=[draft_response] if draft_response else [],
                current_sources=[p.get("source", "Unknown") for p in processed_pairs]
            )

        # Case 2: Direct or High Risk Contradiction Detected (Hallucination)
        elif hallucinated_claims or overall_contradiction >= domain_policy.correct_contradiction_threshold:
            if overall_contradiction >= domain_policy.reject_contradiction_threshold or primary_contradiction["taxonomy_type"] == "DIRECT_SAFETY_CONTRADICTION" or severity == "CRITICAL":
                decision = "REJECT"
                next_action = "Block response immediately; output safety alert or default safe response."
                reason = f"Critical {primary_contradiction['taxonomy_type']} detected under {domain_policy.domain_name} policy."
                explanation = f"Response refuted by verified authoritative ground-truth ({primary_contradiction['explanation']})."
            else:
                decision = "CORRECT"
                next_action = "Forward hallucinated claims and trusted evidence payload to Corrector Agent."
                reason = f"Hallucination identified ({primary_contradiction['taxonomy_type']}); fixable via minimal edit."
                explanation = f"Specific claims conflict with verified evidence; forwarding {len(hallucinated_claims)} claim(s) to Corrector Agent."

        # Case 3: High Confidence & Entailment (Factually Accurate)
        elif calibrated_conf >= domain_policy.accept_confidence_threshold and overall_entailment >= 0.50 and authority >= domain_policy.min_required_authority:
            decision = "ACCEPT"
            next_action = "Pass response directly to user without modification."
            reason = f"Response verified under {domain_policy.domain_name} policy with calibrated confidence {calibrated_conf:.2f}."
            explanation = "All key claims are strongly supported by high-authority, fresh, consistent ground-truth evidence."

        # Case 4: Moderate Confidence / Borderline Evidence -> VERIFY_AGAIN with Negotiation Protocol
        else:
            decision = "VERIFY_AGAIN"
            next_action = "Execute structured negotiation instruction; request targeted re-verification from Verifier Agent."
            reason = f"Calibrated confidence ({calibrated_conf:.2f}) falls below {domain_policy.domain_name} acceptance threshold ({domain_policy.accept_confidence_threshold:.2f})."
            explanation = "Evidence coverage is partial or requires secondary domain verification pass."

            unverified = [p.get("claim", draft_response) for p in processed_pairs if p.get("top_relation") != "entailment"]
            negotiation_protocol = self.negotiation_engine.generate_verifier_instruction(
                domain_policy=domain_policy,
                evidence_completeness=completeness,
                unverified_claims=unverified or [draft_response],
                current_sources=[p.get("source", "Unknown") for p in processed_pairs]
            )

        # ---------------------------------------------------------
        # Package Corrector Payload & Audit Record
        # ---------------------------------------------------------
        corrector_payload = {
            "judge_decision": decision,
            "severity": severity,
            "reason": reason,
            "explanation": explanation,
            "next_action": next_action,
            "claim_level_hallucination_scores": all_claim_scores,
            "hallucinated_claims": hallucinated_claims,
            "verified_claims": verified_claims,
            "trusted_evidence": trusted_evidence,
            "original_draft_response": draft_response,
            "user_query": user_query,
            "corrector_guidance": (
                f"Replace or repair {len(hallucinated_claims)} hallucinated claim(s) using trusted evidence. "
                "Preserve original non-hallucinated content."
            ) if decision == "CORRECT" else "No correction needed."
        }

        # Build Decision Audit Record
        audit_record = self.audit_engine.build_audit_record(
            decision=decision,
            severity=severity,
            calibrated_conf=calibrated_conf,
            domain_policy=domain_policy,
            evidence_intel=evidence_intel,
            consensus_data=consensus_data,
            contradiction_data=primary_contradiction,
            detector_output={},
            user_query=user_query,
            draft_response=draft_response
        )

        return {
            "judge_decision": decision,
            "severity": severity,
            "reason": reason,
            "explanation": explanation,
            "next_action": next_action,
            "corrector_payload": corrector_payload,
            "negotiation_protocol": negotiation_protocol,
            "audit_record": audit_record,
            "contradiction_taxonomy": primary_contradiction
        }
