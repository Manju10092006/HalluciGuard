"""
HalluciGuard - Canonical Judge Agent
The Chief Decision Officer of HalluciGuard.

The Judge receives VerifierResult from the Verifier and decides what the system should do next:
  - ACCEPT: Release draft response verbatim
  - CORRECT: Deliver targeted CorrectionRequest payload to Snehith's Corrector Agent
  - VERIFY_AGAIN: Request expanded verification pass (if retries available)
  - REJECT: Block response due to critical safety/contradiction risk
  - ABSTAIN: Insufficient evidence or unresolvable pipeline degradation

The Judge does NOT perform independent fact-checking or NLI model inference.
It relies on the authoritative factual investigation produced by the Verifier.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union

try:
    from agents.judge_agent.config import JudgeConfig, DEFAULT_CONFIG
except ImportError:
    from config import JudgeConfig, DEFAULT_CONFIG

try:
    from agents.judge_agent.domain_policies import DomainPolicyRegistry, DEFAULT_DOMAIN_REGISTRY, DomainPolicy
except ImportError:
    from domain_policies import DomainPolicyRegistry, DEFAULT_DOMAIN_REGISTRY, DomainPolicy
from orchestration.schemas import (
    JudgeResult,
    CorrectionRequest,
    ReverificationResult,
    VerifierResult,
    ClaimReport,
    Evidence,
    DetectorResult,
    JudgeDecision,
    SeverityLevel,
    VerdictLabel,
    EntailmentLabel,
    ExecutionStatus,
)

logger = logging.getLogger("HalluciGuard.JudgeAgent")


class JudgeAgent:
    """
    Canonical Judge Agent.
    Evaluates VerifierResult and emits canonical JudgeResult.
    """

    def __init__(self, config: Optional[JudgeConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.domain_registry = DEFAULT_DOMAIN_REGISTRY
        self._consecutive_errors = 0
        self._circuit_open = False
        logger.info("HalluciGuard Canonical Judge Agent initialized.")

    def evaluate(
        self,
        verifier_result: Union[VerifierResult, Dict[str, Any]],
        detector_result: Optional[Union[DetectorResult, Dict[str, Any]]] = None,
        user_query: str = "",
        original_response: str = "",
        draft_response: str = "",
        domain: str = "",
        reverification_result: Optional[Union[ReverificationResult, Dict[str, Any]]] = None,
        retry_count: int = 0
    ) -> JudgeResult:
        """
        Main decision arbitration entry point.
        """
        response_text = original_response or draft_response or ""

        # -------------------------------------------------------------------
        # 1. Post-Correction Re-verification Evaluation (Phase J5 Loop)
        # -------------------------------------------------------------------
        if reverification_result is not None:
            return self._evaluate_reverification(reverification_result, user_query, response_text)

        # -------------------------------------------------------------------
        # 2. Input Normalization & Controlled Failure Handling
        # -------------------------------------------------------------------
        normalized_verifier = self._normalize_verifier_result(verifier_result, domain)
        if normalized_verifier is None:
            logger.warning("Judge received empty or unparseable VerifierResult. Returning ABSTAIN.")
            return JudgeResult(
                decision=JudgeDecision.ABSTAIN,
                severity=SeverityLevel.HIGH,
                reason="Invalid or missing VerifierResult payload.",
                explanation="Grounding evidence was absent or failed schema validation. Unsafe to proceed.",
                confidence=0.0,
                correction_request=None,
                status=ExecutionStatus.FAILED
            )

        normalized_detector = self._normalize_detector_result(detector_result)
        domain_name = normalized_verifier.domain or domain or "General Knowledge"
        policy = self.domain_registry.get_policy(domain_name)

        # -------------------------------------------------------------------
        # 3. Claim-Level Decision Processing (No NLI re-verification)
        # -------------------------------------------------------------------
        claim_reports = normalized_verifier.claim_reports

        claims_to_correct: List[ClaimReport] = []
        claims_to_preserve: List[ClaimReport] = []
        trusted_evidence: List[Evidence] = []
        contradictory_evidence: List[Evidence] = []
        unverified_claims: List[ClaimReport] = []
        conflicted_claims: List[ClaimReport] = []

        for claim in claim_reports:
            verdict_str = str(claim.verdict).lower()
            if verdict_str == VerdictLabel.CONTRADICTED.value:
                claims_to_correct.append(claim)
                for ev in claim.evidence:
                    contradictory_evidence.append(ev)
            elif verdict_str == VerdictLabel.VERIFIED.value:
                claims_to_preserve.append(claim)
                for ev in claim.evidence:
                    trusted_evidence.append(ev)
            elif verdict_str == VerdictLabel.CONFLICTED.value:
                conflicted_claims.append(claim)
            elif verdict_str == VerdictLabel.UNVERIFIED.value:
                unverified_claims.append(claim)
            else:
                if claim.contradiction_score >= 0.5:
                    claims_to_correct.append(claim)
                    for ev in claim.evidence:
                        contradictory_evidence.append(ev)
                elif claim.support_score >= 0.5:
                    claims_to_preserve.append(claim)
                    for ev in claim.evidence:
                        trusted_evidence.append(ev)
                else:
                    unverified_claims.append(claim)

        for ev in normalized_verifier.evidence:
            if ev not in trusted_evidence and ev not in contradictory_evidence:
                trusted_evidence.append(ev)

        # -------------------------------------------------------------------
        # 4. Apply Policy Decision Governance Tree
        # -------------------------------------------------------------------
        det_prob = normalized_detector.hallucination_probability if normalized_detector else 0.0

        has_contradictions = len(claims_to_correct) > 0
        has_preservations = len(claims_to_preserve) > 0
        has_unverified = len(unverified_claims) > 0
        has_conflicted = len(conflicted_claims) > 0
        total_claims = len(claim_reports)

        decision: JudgeDecision = JudgeDecision.ABSTAIN
        severity: SeverityLevel = SeverityLevel.LOW
        reason: str = ""
        explanation: str = ""
        correction_req: Optional[CorrectionRequest] = None

        # Rule A: Critical / Direct Safety Contradictions -> REJECT (or CORRECT if non-critical)
        if has_contradictions:
            is_critical_domain = policy.strictness_level in ["VERY_STRICT", "STRICT"]
            is_high_contradiction = any(c.contradiction_score >= policy.reject_contradiction_threshold for c in claims_to_correct)

            if is_critical_domain and is_high_contradiction:
                decision = JudgeDecision.REJECT
                severity = SeverityLevel.CRITICAL
                reason = f"Critical factual contradiction detected in {policy.domain_name} domain."
                explanation = f"Claim(s) strongly refuting ground-truth. Rejected to prevent safety/compliance risk."
            else:
                decision = JudgeDecision.CORRECT
                severity = SeverityLevel.MEDIUM
                reason = f"Identified {len(claims_to_correct)} contradicted claim(s) requiring evidence-grounded repair."
                explanation = f"Response contains fixable factual errors. Directing Corrector to repair flagged claims while preserving verified claims."

                instructions = (
                    f"Modify only the {len(claims_to_correct)} claim(s) flagged in claims_to_correct using "
                    f"contradictory_evidence and trusted_evidence. "
                    f"Preserve all {len(claims_to_preserve)} claim(s) in claims_to_preserve without altering facts."
                )

                correction_req = CorrectionRequest(
                    execution_id=f"exec-{int(time.time())}",
                    user_query=user_query,
                    original_response=response_text,
                    claims_to_correct=claims_to_correct,
                    claims_to_preserve=claims_to_preserve,
                    trusted_evidence=trusted_evidence,
                    contradictory_evidence=contradictory_evidence,
                    correction_instructions=instructions
                )

        # Rule B: Absent evidence (0 claims evaluated)
        elif total_claims == 0:
            if det_prob >= 0.70:
                decision = JudgeDecision.REJECT
                severity = SeverityLevel.HIGH
                reason = f"High hallucination risk ({det_prob:.2f}) with zero supporting evidence."
                explanation = "Response flagged as high risk by Detector without grounding evidence."
            elif retry_count < self.config.max_verification_retries:
                decision = JudgeDecision.VERIFY_AGAIN
                severity = SeverityLevel.MEDIUM
                reason = "No verification claims/evidence provided. Requesting retrieval pass."
                explanation = "Verifier produced empty evidence set. Retrying verification."
            else:
                decision = JudgeDecision.ABSTAIN
                severity = SeverityLevel.HIGH
                reason = f"Insufficient grounding evidence in {policy.domain_name} domain."
                explanation = "Grounding evidence was absent and retries exhausted."

        # Rule C: All evaluated claims verified -> ACCEPT
        elif not has_contradictions and has_preservations:
            decision = JudgeDecision.ACCEPT
            severity = SeverityLevel.LOW
            reason = "All claims verified against authoritative ground-truth evidence."
            explanation = f"Response is fully grounded in {policy.domain_name} sources with overall confidence {normalized_verifier.overall_confidence:.2f}."

        # Rule D: Unverified or Conflicted claims (UNVERIFIED != CONTRADICTED)
        elif has_unverified or has_conflicted:
            if retry_count < self.config.max_verification_retries:
                decision = JudgeDecision.VERIFY_AGAIN
                severity = SeverityLevel.MEDIUM
                reason = f"Unverified or conflicted claims present. Triggering verification retry pass {retry_count + 1}."
                explanation = f"Evidence was insufficient or conflicted for {len(unverified_claims) + len(conflicted_claims)} claim(s). Requesting expanded retrieval."
            elif policy.strictness_level in ["VERY_STRICT", "STRICT"]:
                decision = JudgeDecision.ABSTAIN
                severity = SeverityLevel.HIGH
                reason = f"Insufficient grounding evidence under strict {policy.domain_name} policy."
                explanation = "Verification retries exhausted without sufficient authoritative grounding."
            else:
                decision = JudgeDecision.ACCEPT
                severity = SeverityLevel.LOW
                reason = f"Unverified claim accepted under relaxed {policy.domain_name} policy baseline."
                explanation = "Low-risk conversational domain allows release of unverified non-safety claim."

        confidence = round(min(1.0, max(0.0, normalized_verifier.overall_confidence * (1.0 - 0.2 * det_prob))), 4)

        return JudgeResult(
            decision=decision,
            severity=severity,
            reason=reason,
            explanation=explanation,
            confidence=confidence,
            correction_request=correction_req,
            status=ExecutionStatus.COMPLETED
        )

    def _evaluate_reverification(
        self,
        reverification_result: Union[ReverificationResult, Dict[str, Any]],
        user_query: str,
        response_text: str
    ) -> JudgeResult:
        """
        Phase J5 — Evaluates post-correction ReverificationResult.
        """
        if isinstance(reverification_result, dict):
            try:
                rev_res = ReverificationResult.model_validate(reverification_result)
            except Exception:
                passed = reverification_result.get("passed", False)
                rem_cnt = reverification_result.get("remaining_contradictions", 0)
                if passed and rem_cnt == 0:
                    return JudgeResult(
                        decision=JudgeDecision.ACCEPT,
                        severity=SeverityLevel.LOW,
                        reason="Post-correction re-verification passed. Safe to release.",
                        explanation="Corrected text verified with 0 remaining contradictions.",
                        confidence=0.90,
                        correction_request=None,
                        status=ExecutionStatus.COMPLETED
                    )
                else:
                    return JudgeResult(
                        decision=JudgeDecision.REJECT,
                        severity=SeverityLevel.HIGH,
                        reason="Post-correction re-verification failed with remaining contradictions.",
                        explanation="Correction introduced or retained factual contradictions. Rolling back.",
                        confidence=0.30,
                        correction_request=None,
                        status=ExecutionStatus.COMPLETED
                    )
        else:
            rev_res = reverification_result

        if rev_res.passed and rev_res.remaining_contradictions == 0:
            return JudgeResult(
                decision=JudgeDecision.ACCEPT,
                severity=SeverityLevel.LOW,
                reason="Post-correction re-verification passed successfully. Safe to commit.",
                explanation="Refined text verified by Verifier with zero remaining contradictions.",
                confidence=0.92,
                correction_request=None,
                status=ExecutionStatus.COMPLETED
            )
        else:
            return JudgeResult(
                decision=JudgeDecision.REJECT,
                severity=SeverityLevel.HIGH,
                reason=f"Post-correction re-verification failed with {rev_res.remaining_contradictions} remaining contradiction(s).",
                explanation="Correction failed re-verification gate. Rolling back to safe response.",
                confidence=0.20,
                correction_request=None,
                status=ExecutionStatus.COMPLETED
            )

    def _normalize_verifier_result(
        self,
        verifier_result: Union[VerifierResult, Dict[str, Any]],
        fallback_domain: str
    ) -> Optional[VerifierResult]:
        """Normalizes dict or VerifierResult into canonical VerifierResult Pydantic model."""
        if verifier_result is None:
            return None
        if isinstance(verifier_result, VerifierResult):
            return verifier_result

        if isinstance(verifier_result, dict):
            try:
                return VerifierResult.model_validate(verifier_result)
            except Exception as e:
                logger.debug(f"Direct Pydantic parsing failed ({e}), normalizing dictionary schema...")

            query_id = verifier_result.get("query_id", "Q-001")
            domain = verifier_result.get("domain", fallback_domain or "General Knowledge")
            overall_conf = verifier_result.get("overall_confidence", verifier_result.get("confidence_score", 0.8))

            claim_reports: List[ClaimReport] = []

            if "claim_reports" in verifier_result:
                for c in verifier_result["claim_reports"]:
                    if isinstance(c, ClaimReport):
                        claim_reports.append(c)
                    elif isinstance(c, dict):
                        try:
                            claim_reports.append(ClaimReport.model_validate(c))
                        except Exception:
                            pass

            elif "claims" in verifier_result:
                raw_claims = verifier_result["claims"]
                for i, c in enumerate(raw_claims):
                    if isinstance(c, dict):
                        c_id = c.get("claim_id", f"C{i+1}")
                        c_text = c.get("claim_text", c.get("claim", ""))
                        v_str = str(c.get("verdict", "unverified")).lower()
                        if "verified" in v_str:
                            verdict = VerdictLabel.VERIFIED
                        elif "contradict" in v_str:
                            verdict = VerdictLabel.CONTRADICTED
                        elif "conflict" in v_str:
                            verdict = VerdictLabel.CONFLICTED
                        else:
                            verdict = VerdictLabel.UNVERIFIED

                        ev_list: List[Evidence] = []
                        for j, ev_data in enumerate(c.get("evidence", [])):
                            if isinstance(ev_data, dict):
                                ev_list.append(Evidence(
                                    evidence_id=ev_data.get("evidence_id", f"E{j+1}"),
                                    title=ev_data.get("title", ""),
                                    source=ev_data.get("source", "Unknown"),
                                    url=ev_data.get("url"),
                                    snippet=ev_data.get("snippet", ev_data.get("evidence_snippet", "")),
                                    entailment_label=EntailmentLabel.CONTRADICTION if verdict == VerdictLabel.CONTRADICTED else EntailmentLabel.ENTAILMENT,
                                    entailment_score=ev_data.get("entailment_score", 0.8),
                                    credibility_score=ev_data.get("credibility_score", 0.8)
                                ))

                        claim_reports.append(ClaimReport(
                            claim_id=c_id,
                            claim_text=c_text,
                            verdict=verdict,
                            support_score=0.9 if verdict == VerdictLabel.VERIFIED else 0.1,
                            contradiction_score=0.9 if verdict == VerdictLabel.CONTRADICTED else 0.1,
                            confidence_score=c.get("confidence_score", 0.8),
                            evidence=ev_list
                        ))
                    elif isinstance(c, str):
                        claim_reports.append(ClaimReport(
                            claim_id=f"C{i+1}",
                            claim_text=c,
                            verdict=VerdictLabel.UNVERIFIED,
                            support_score=0.5,
                            contradiction_score=0.0,
                            confidence_score=0.5,
                            evidence=[]
                        ))

            elif "claim_evidence_pairs" in verifier_result:
                pairs = verifier_result["claim_evidence_pairs"]
                for i, pair in enumerate(pairs):
                    c_text = pair.get("claim", "")
                    ev_text = pair.get("evidence", pair.get("evidence_snippet", ""))
                    src = pair.get("source", "Unknown")
                    rel = pair.get("nli_relation", pair.get("top_relation", "")).lower()
                    ev_lower = ev_text.lower()
                    c_lower = c_text.lower()
                    
                    refutation_keywords = [
                        "contraindicated", "refutes", "mismatch", "false", "incorrect",
                        "prohibited", "fatal", "is not", "does not", "not directly",
                        "interpreted", "refuted", "denied", "contrary"
                    ]
                    is_refutation = any(k in ev_lower for k in refutation_keywords)

                    # Simple regex numeric mismatch check
                    import re
                    c_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', c_lower))
                    ev_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', ev_lower))
                    is_num_mismatch = bool(c_nums and ev_nums and not c_nums.intersection(ev_nums))

                    if "contra" in rel or pair.get("contradiction_score", 0) > 0.4 or is_refutation or is_num_mismatch:
                        verdict = VerdictLabel.CONTRADICTED
                    elif "entail" in rel or pair.get("entailment_score", 0) > 0.5 or (ev_text and not rel):
                        verdict = VerdictLabel.VERIFIED
                    else:
                        verdict = VerdictLabel.UNVERIFIED

                    ev = Evidence(
                        evidence_id=f"E{i+1}",
                        title=src,
                        source=src,
                        snippet=ev_text,
                        entailment_label=EntailmentLabel.CONTRADICTION if verdict == VerdictLabel.CONTRADICTED else EntailmentLabel.ENTAILMENT,
                        entailment_score=0.85,
                        credibility_score=0.80
                    )

                    claim_reports.append(ClaimReport(
                        claim_id=f"C{i+1}",
                        claim_text=c_text,
                        verdict=verdict,
                        support_score=0.85 if verdict == VerdictLabel.VERIFIED else 0.1,
                        contradiction_score=0.85 if verdict == VerdictLabel.CONTRADICTED else 0.1,
                        confidence_score=0.85,
                        evidence=[ev] if ev_text else []
                    ))

            return VerifierResult(
                query_id=query_id,
                domain=domain,
                claim_reports=claim_reports,
                evidence=[],
                overall_confidence=overall_conf,
                status=ExecutionStatus.COMPLETED
            )

        return None

    @staticmethod
    def _normalize_verifier_output(verifier_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Legacy static normalization helper for claim_evidence or claim_evidence_pairs.
        """
        if not verifier_output or not isinstance(verifier_output, dict):
            return []

        if "claim_evidence_pairs" in verifier_output:
            return verifier_output["claim_evidence_pairs"]

        if "claim_evidence" in verifier_output:
            pairs = []
            for item in verifier_output["claim_evidence"]:
                claim_text = item.get("claim_text", item.get("claim", ""))
                verdict = item.get("verdict", "")
                trust_score = item.get("trust_score", 0.0)
                evidence_list = item.get("evidence", [])

                if not evidence_list:
                    continue

                for ev in evidence_list:
                    if isinstance(ev, dict):
                        snippet = ev.get("snippet", ev.get("evidence_snippet", ""))
                        source = ev.get("source", "unknown")
                        pub_date = ev.get("publication_date")
                        entail_score = ev.get("entailment_score", 0.8)
                        cred_score = ev.get("credibility_score", 0.8)

                        pairs.append({
                            "claim": claim_text,
                            "evidence": snippet,
                            "source": source,
                            "publication_date": pub_date,
                            "verifier_verdict": verdict,
                            "verifier_trust_score": trust_score,
                            "evidence_confidence": float(entail_score * cred_score)
                        })
            return pairs

        return []

    def _normalize_detector_result(
        self,
        detector_result: Optional[Union[DetectorResult, Dict[str, Any]]]
    ) -> Optional[DetectorResult]:
        """Normalizes dict or DetectorResult into canonical DetectorResult model."""
        if detector_result is None:
            return None
        if isinstance(detector_result, DetectorResult):
            return detector_result
        if isinstance(detector_result, dict):
            try:
                return DetectorResult.model_validate(detector_result)
            except Exception:
                prob = float(detector_result.get("hallucination_probability", 0.0))
                conf = float(detector_result.get("confidence_score", 0.8))
                from orchestration.schemas import RiskLevel, NextAction
                risk = RiskLevel.HIGH if prob >= 0.7 else (RiskLevel.MEDIUM if prob >= 0.4 else RiskLevel.LOW)
                return DetectorResult(
                    hallucination_probability=prob,
                    confidence_score=conf,
                    risk_level=risk,
                    next_action=NextAction.VERIFY if prob >= 0.3 else NextAction.ACCEPT,
                    status=ExecutionStatus.COMPLETED
                )
        return None
