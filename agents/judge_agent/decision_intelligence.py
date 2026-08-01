"""
HalluciGuard - Decision Intelligence Engine
The Chief Decision Officer. Orchestrates all subsystems to produce a final
governance decision. Does NOT calculate confidence scores. REASONS about trust.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

from config import Decision, Severity, EvidenceQuality, SystemHealth
from decision_policies import DomainPolicy, DomainPolicyRegistry, DEFAULT_POLICY_REGISTRY
from evidence_governance import EvidenceGovernanceEngine, EvidenceSetGovernanceReport
from coverage_analyzer import CoverageAnalyzer, CoverageReport
from nli_engine import NLIEngine
from runtime_inspector import RuntimeInspector, RuntimeInspectionReport
from conflict_resolver import ConflictResolver, ConflictReport
from consensus_analyzer import ConsensusAnalyzer, ConsensusReport
from risk_intelligence import RiskIntelligenceEngine, RiskAssessment
from memory_intelligence import MemoryIntelligenceLayer, MemoryInsight
from workflow_orchestrator import WorkflowOrchestrator, WorkflowAction
from explainability import ExplainabilityEngine, ReasoningChain, AuditRecord

logger = logging.getLogger("HalluciGuard.DecisionIntelligence")


@dataclass
class JudgeVerdict:
    """The complete output of the Judge Agent's decision process."""
    decision: Decision
    severity: Severity
    reasoning_chain: List[str]
    workflow_action: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    evidence_governance: Dict[str, Any]
    coverage: Dict[str, Any]
    conflicts: List[Dict[str, Any]]
    consensus: Dict[str, Any]
    runtime_health: Dict[str, Any]
    memory_insight: Dict[str, Any]
    audit_record: Dict[str, Any]
    claim_verdicts: List[Dict[str, Any]]
    detector_signal: Dict[str, Any]
    alternatives_rejected: Dict[str, str]


class DecisionIntelligenceEngine:
    """
    The Chief Decision Officer of HalluciGuard.
    Orchestrates the entire verification governance process.
    """

    def __init__(self, policy_registry: Optional[DomainPolicyRegistry] = None):
        self.policy_registry = policy_registry or DEFAULT_POLICY_REGISTRY
        self.nli = NLIEngine(use_hf=True)
        self.evidence_gov = EvidenceGovernanceEngine()
        self.coverage = CoverageAnalyzer()
        self.runtime = RuntimeInspector()
        self.conflicts = ConflictResolver()
        self.consensus = ConsensusAnalyzer()
        self.risk = RiskIntelligenceEngine()
        self.memory = MemoryIntelligenceLayer()
        self.workflow = WorkflowOrchestrator()
        self.explain = ExplainabilityEngine()

    def evaluate(
        self,
        user_query: str,
        draft_response: str,
        detector_output: Dict[str, Any],
        verifier_output: Dict[str, Any],
        domain: str = "",
        memory_context: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> JudgeVerdict:
        """
        Execute the full decision intelligence pipeline.
        This is the single entry point for all Judge Agent operations.
        """
        logger.info(f"=== JUDGE AGENT DECISION PIPELINE INITIATED ===")
        logger.info(f"Domain: {domain or 'Auto-detect'} | Retry: {retry_count}")

        # ---- Phase 1: Governance Context ----
        policy = self.policy_registry.get_policy(domain)
        logger.info(f"Governing Policy: {policy.domain_name} ({policy.strictness_level})")

        # ---- Phase 1b: Dynamic Claim Extraction & Evidence Alignment ----
        pairs = verifier_output.get("claim_evidence_pairs", [])
        pairs = self._extract_and_align_claims(draft_response, pairs)

        # ---- Phase 2: Runtime Inspection ----
        nli_results = self.nli.batch_predict(pairs)
        runtime_report = self.runtime.inspect_pipeline(detector_output, verifier_output, nli_results)
        logger.info(f"Pipeline Health: {runtime_report.system_health.value}")

        # ---- Phase 3: Evidence Governance ----
        evidence_report = self.evidence_gov.assess_evidence_set(pairs, policy)
        logger.info(f"Evidence Quality: {evidence_report.overall_quality.value}")

        # ---- Phase 4: Coverage Analysis ----
        coverage_report = self.coverage.analyze_coverage(nli_results)
        logger.info(f"Coverage: {coverage_report.coverage_status}")

        # ---- Phase 5: Conflict Resolution ----
        conflict_reports = self.conflicts.analyze_conflicts(nli_results, policy)
        active_conflicts = [c for c in conflict_reports if c.has_conflict]
        logger.info(f"Conflicts: {len(active_conflicts)} active")

        # ---- Phase 6: Consensus Analysis ----
        consensus_report = self.consensus.evaluate_consensus(nli_results)
        logger.info(f"Consensus: {consensus_report.consensus_status}")

        # ---- Phase 7: Memory Intelligence ----
        all_sources = [p.get("source", p.get("evidence_source", "")) for p in pairs]
        memory_insight = self.memory.query_memory(draft_response, all_sources, memory_context)

        # ---- Phase 8: Risk Assessment ----
        risk_assessment = self.risk.assess_risk(
            policy, evidence_report, coverage_report,
            conflict_reports, consensus_report, runtime_report
        )
        logger.info(f"Risk: {risk_assessment.risk_level.value} | Safe: {risk_assessment.is_safe_to_release}")

        # ---- Phase 9: Decision Reasoning (NOT SCORING) ----
        decision, severity = self._reason_decision(
            detector_output, policy, evidence_report, coverage_report,
            conflict_reports, consensus_report, runtime_report,
            risk_assessment, memory_insight, retry_count
        )
        logger.info(f"DECISION: {decision.value} | SEVERITY: {severity.value}")

        # ---- Phase 10: Workflow Orchestration ----
        action = self.workflow.determine_next_action(
            decision, risk_assessment, evidence_report, conflict_reports, policy, retry_count
        )

        # ---- Phase 11: Explainability ----
        reasoning = self.explain.build_reasoning_chain(
            decision, severity, policy, evidence_report, coverage_report,
            conflict_reports, consensus_report, runtime_report, risk_assessment, memory_insight
        )
        audit = self.explain.build_audit_record(
            decision, severity, reasoning, policy, evidence_report,
            runtime_report, risk_assessment, user_query, draft_response
        )

        # ---- Phase 12: Claim-Level Verdicts ----
        claim_verdicts = self._build_claim_verdicts(nli_results, conflict_reports, detector_output)

        # ---- Assemble Verdict ----
        return JudgeVerdict(
            decision=decision, severity=severity,
            reasoning_chain=reasoning.steps,
            workflow_action={"type": action.action_type, "target": action.target_agent,
                             "instructions": action.instructions, "priority": action.priority,
                             "reasoning": action.reasoning},
            risk_assessment={"level": risk_assessment.risk_level.value,
                             "safe_to_release": risk_assessment.is_safe_to_release,
                             "human_review": risk_assessment.requires_human_review,
                             "factors": risk_assessment.risk_factors,
                             "mitigating": risk_assessment.mitigating_factors,
                             "reasoning": risk_assessment.reasoning},
            evidence_governance={"quality": evidence_report.overall_quality.value,
                                 "total_items": evidence_report.total_items,
                                 "authoritative": evidence_report.authoritative_count,
                                 "sufficient": evidence_report.is_sufficient_for_decision,
                                 "concerns": evidence_report.governance_concerns,
                                 "reasoning": evidence_report.reasoning},
            coverage={"status": coverage_report.coverage_status,
                      "covered": coverage_report.claims_covered,
                      "uncovered": coverage_report.claims_uncovered,
                      "partial": coverage_report.claims_partially_covered,
                      "total": coverage_report.total_claims},
            conflicts=[{"type": c.conflict_type.value, "claim": c.affected_claim[:100],
                        "implication": c.implication, "safety_critical": c.is_safety_critical,
                        "immediate_action": c.requires_immediate_action} for c in conflict_reports],
            consensus={"status": consensus_report.consensus_status,
                       "agreeing": consensus_report.agreeing_sources,
                       "disagreeing": consensus_report.disagreeing_sources,
                       "reasoning": consensus_report.reasoning},
            runtime_health={"health": runtime_report.system_health.value,
                            "trustworthy": runtime_report.is_pipeline_trustworthy,
                            "warnings": runtime_report.degradation_warnings,
                            "reasoning": runtime_report.reasoning},
            memory_insight={"historical_context": memory_insight.has_historical_context,
                            "known_pattern": memory_insight.known_hallucination_pattern,
                            "concerns": memory_insight.memory_concerns,
                            "recommendations": memory_insight.memory_recommendations},
            audit_record={"id": audit.audit_id, "timestamp": audit.timestamp,
                          "decision": audit.final_decision, "severity": audit.severity,
                          "reasoning_chain": audit.reasoning_chain},
            claim_verdicts=claim_verdicts,
            detector_signal={"hallucination_probability": detector_output.get("hallucination_probability", 0),
                             "confidence_score": detector_output.get("confidence_score", 0),
                             "flagged_claims": detector_output.get("flagged_claims", [])},
            alternatives_rejected=reasoning.alternatives_considered
        )

    def _reason_decision(
        self,
        detector_output, policy, evidence_report, coverage_report,
        conflict_reports, consensus_report, runtime_report,
        risk_assessment, memory_insight, retry_count
    ) -> tuple:
        """
        Core reasoning engine. Produces a decision through GOVERNANCE LOGIC, not scoring.
        Follows a structured decision tree based on safety, evidence, and risk factors.
        """
        det_prob = detector_output.get("hallucination_probability", 0.0)
        has_safety_conflict = any(c.is_safety_critical for c in conflict_reports if c.has_conflict)
        has_any_conflict = any(c.has_conflict for c in conflict_reports)
        has_immediate = any(c.requires_immediate_action for c in conflict_reports if c.has_conflict)

        # ═══════════════════════════════════════════════════════════════
        # RULE 1: Safety-critical conflicts → REJECT or ESCALATE
        # ═══════════════════════════════════════════════════════════════
        if has_safety_conflict or has_immediate:
            if policy.escalate_on_safety_conflict:
                return Decision.ESCALATE_HUMAN, Severity.CRITICAL
            return Decision.REJECT, Severity.CRITICAL

        # ═══════════════════════════════════════════════════════════════
        # RULE 2: Pipeline critically failed → ABSTAIN
        # ═══════════════════════════════════════════════════════════════
        if runtime_report.system_health == SystemHealth.CRITICAL_FAILURE:
            return Decision.ABSTAIN, Severity.HIGH

        # ═══════════════════════════════════════════════════════════════
        # RULE 3: No evidence at all → depends on Detector signal
        # ═══════════════════════════════════════════════════════════════
        if evidence_report.overall_quality == EvidenceQuality.ABSENT:
            if det_prob >= 0.7:
                return Decision.REJECT, Severity.HIGH
            if policy.retry_on_insufficient_evidence and retry_count < 2:
                return Decision.VERIFY_AGAIN, Severity.MEDIUM
            return Decision.ABSTAIN, Severity.MEDIUM

        # ═══════════════════════════════════════════════════════════════
        # RULE 4: Direct conflicts exist → CORRECT if fixable, else REJECT
        # ═══════════════════════════════════════════════════════════════
        if has_any_conflict:
            conflict_types = set(c.conflict_type for c in conflict_reports if c.has_conflict)
            # If conflicts are severe and in strict domain
            if policy.strictness_level == "VERY_STRICT" and len([c for c in conflict_reports if c.has_conflict]) >= 2:
                return Decision.REJECT, Severity.HIGH
            # Conflicts are fixable (correction candidate)
            return Decision.CORRECT, Severity.MEDIUM

        # ═══════════════════════════════════════════════════════════════
        # RULE 5: Detector signals high hallucination but no evidence conflicts
        # ═══════════════════════════════════════════════════════════════
        if det_prob >= 0.7:
            # Evidence exists but Detector is alarmed → respect both signals
            if evidence_report.is_sufficient_for_decision and consensus_report.has_consensus:
                # Evidence overrides Detector concern
                return Decision.ACCEPT, Severity.LOW
            # Detector is alarmed and evidence is not conclusive
            if policy.retry_on_insufficient_evidence and retry_count < 2:
                return Decision.VERIFY_AGAIN, Severity.MEDIUM
            return Decision.CORRECT, Severity.MEDIUM

        # ═══════════════════════════════════════════════════════════════
        # RULE 6: Detector signals moderate concern
        # ═══════════════════════════════════════════════════════════════
        if det_prob >= 0.4:
            if evidence_report.is_sufficient_for_decision:
                if consensus_report.has_consensus:
                    return Decision.ACCEPT, Severity.INFORMATIONAL
                if coverage_report.coverage_status == "FULLY_COVERED":
                    return Decision.ACCEPT, Severity.LOW
            if policy.retry_on_insufficient_evidence and retry_count < 2:
                return Decision.VERIFY_AGAIN, Severity.LOW
            return Decision.CORRECT, Severity.LOW

        # ═══════════════════════════════════════════════════════════════
        # RULE 7: Detector signals low concern + evidence supports
        # ═══════════════════════════════════════════════════════════════
        if evidence_report.is_sufficient_for_decision:
            return Decision.ACCEPT, Severity.INFORMATIONAL

        # ═══════════════════════════════════════════════════════════════
        # RULE 8: Low Detector concern but insufficient evidence
        # ═══════════════════════════════════════════════════════════════
        if policy.strictness_level in ("VERY_STRICT", "STRICT"):
            if policy.retry_on_insufficient_evidence and retry_count < 2:
                return Decision.VERIFY_AGAIN, Severity.LOW
            return Decision.CORRECT, Severity.LOW

        # Default: Accept with informational note
        return Decision.ACCEPT, Severity.INFORMATIONAL

    def _extract_and_align_claims(
        self,
        draft_response: str,
        provided_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract sentences/claims from draft_response if provided, and align them
        with any matching evidence provided in provided_pairs.
        """
        if not draft_response or not draft_response.strip():
            return provided_pairs

        # Split response into clean sentences using regex (avoiding splitting on decimals like 96.8)
        import re
        raw_sentences = [s.strip() for s in re.split(r'(?<!\d)\.(?!\d)|[!\n\?]', draft_response) if s.strip()]
        if not raw_sentences:
            return provided_pairs

        aligned = []
        used_provided_indices = set()

        for stmt in raw_sentences:
            if len(stmt) < 5:  # skip trivial fragments
                continue

            # Try to match with provided evidence pairs
            matched_pair = None
            best_overlap = 0.0

            stmt_words = set(w.strip(".,!?:;\"'").lower() for w in stmt.split() if len(w.strip(".,!?:;\"'")) > 2)

            for idx, p in enumerate(provided_pairs):
                p_claim = p.get("claim", "")
                p_words = set(w.strip(".,!?:;\"'").lower() for w in p_claim.split() if len(w.strip(".,!?:;\"'")) > 2)
                if stmt_words and p_words:
                    overlap = len(stmt_words & p_words) / len(stmt_words)
                    if overlap > best_overlap and overlap > 0.2:
                        best_overlap = overlap
                        matched_pair = p
                        used_provided_indices.add(idx)

            if matched_pair:
                aligned.append({
                    "claim": stmt,
                    "evidence": matched_pair.get("evidence", ""),
                    "source": matched_pair.get("source", matched_pair.get("evidence_source", "Provided Evidence"))
                })
            else:
                aligned.append({
                    "claim": stmt,
                    "evidence": "",
                    "source": "None (Unverified)"
                })

        # Also include any remaining provided pairs that weren't matched
        for idx, p in enumerate(provided_pairs):
            if idx not in used_provided_indices and p.get("claim"):
                aligned.append(p)

        return aligned if aligned else provided_pairs

    def _build_claim_verdicts(
        self,
        nli_results: List[Dict[str, Any]],
        conflict_reports: List[ConflictReport],
        detector_output: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Build per-claim verdict breakdown with explicit hallucination scores."""
        verdicts = []
        det_prob = (detector_output or {}).get("hallucination_probability", 0.0)

        for i, nli_pair in enumerate(nli_results):
            claim = nli_pair.get("claim", "")
            evidence = nli_pair.get("evidence", "")
            scores = nli_pair.get("nli_scores", {})
            relation = nli_pair.get("top_relation", "neutral")

            conflict = conflict_reports[i] if i < len(conflict_reports) else None

            entail = scores.get("entailment", 0.0)
            contra = scores.get("contradiction", 0.0)
            neutral = scores.get("neutral", 1.0)

            # Calculate explicit Claim Hallucination Score (0.0 to 1.0)
            if conflict and conflict.has_conflict and conflict.conflict_type.name != "NO_CONFLICT":
                claim_hallucination_prob = max(contra, det_prob, 0.85)
            elif relation == "contradiction":
                claim_hallucination_prob = max(contra, det_prob)
            elif not evidence or not evidence.strip():
                # No evidence provided for claim -> hallucination risk driven by Detector probability & lack of grounding
                claim_hallucination_prob = max(det_prob, 0.40 if det_prob > 0.3 else 0.15)
            elif relation == "entailment":
                claim_hallucination_prob = round(max(0.0, (1.0 - entail) * det_prob), 4)
            else:
                # Neutral / partial evidence
                claim_hallucination_prob = round(0.5 * det_prob + 0.5 * (1.0 - entail), 4)

            verdict = {
                "claim_index": i,
                "claim_text": claim,
                "evidence_text": evidence[:200] if evidence else "No grounding evidence provided",
                "source": nli_pair.get("source", nli_pair.get("evidence_source", "Unknown")),
                "nli_entailment": entail,
                "nli_neutral": neutral,
                "nli_contradiction": contra,
                "hallucination_score": round(claim_hallucination_prob, 4),
                "hallucination_pct": f"{round(claim_hallucination_prob * 100, 1)}%",
                "top_relation": relation,
                "has_conflict": conflict.has_conflict if conflict else False,
                "conflict_type": conflict.conflict_type.value if conflict and conflict.has_conflict else "NONE",
                "conflict_implication": conflict.implication if conflict and conflict.has_conflict else "No conflict.",
                "is_safe": not (conflict.is_safety_critical if conflict else False),
            }

            # Determine per-claim status
            if relation == "entailment" and not (conflict and conflict.has_conflict):
                verdict["status"] = "VERIFIED"
                verdict["status_label"] = "✅ Verified"
            elif relation == "contradiction" or (conflict and conflict.has_conflict) or claim_hallucination_prob >= 0.70:
                verdict["status"] = "HALLUCINATED"
                verdict["status_label"] = "❌ Hallucinated"
            else:
                verdict["status"] = "UNVERIFIED"
                verdict["status_label"] = "⚠️ Unverified"

            verdicts.append(verdict)
        return verdicts
