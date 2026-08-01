"""
HalluciGuard - Decision Explainability & Audit Engine
Produces reproducible reasoning chains and complete audit trails.
Every decision is explainable, traceable, and defensible.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from config import Decision, Severity
from decision_policies import DomainPolicy
from evidence_governance import EvidenceSetGovernanceReport
from coverage_analyzer import CoverageReport
from conflict_resolver import ConflictReport
from consensus_analyzer import ConsensusReport
from runtime_inspector import RuntimeInspectionReport
from risk_intelligence import RiskAssessment
from memory_intelligence import MemoryInsight


@dataclass
class ReasoningChain:
    steps: List[str]
    alternatives_considered: Dict[str, str]
    governing_policy: str
    final_verdict: str


@dataclass
class AuditRecord:
    audit_id: str
    timestamp: str
    user_query: str
    draft_response: str
    domain: str
    final_decision: str
    severity: str
    reasoning_chain: List[str]
    alternatives_rejected: Dict[str, str]
    evidence_summary: Dict[str, Any]
    runtime_health: str
    risk_summary: Dict[str, Any]


class ExplainabilityEngine:
    def build_reasoning_chain(
        self,
        decision: Decision,
        severity: Severity,
        domain_policy: DomainPolicy,
        evidence_report: EvidenceSetGovernanceReport,
        coverage: CoverageReport,
        conflicts: List[ConflictReport],
        consensus: ConsensusReport,
        runtime: RuntimeInspectionReport,
        risk: RiskAssessment,
        memory: MemoryInsight
    ) -> ReasoningChain:
        steps = []

        # Step 1: Domain governance
        steps.append(f"1. GOVERNANCE CONTEXT: Evaluated under '{domain_policy.domain_name}' domain policy "
                      f"(Strictness: {domain_policy.strictness_level}). {domain_policy.description}")

        # Step 2: Runtime health
        steps.append(f"2. PIPELINE HEALTH: System is {runtime.system_health.value}. {runtime.reasoning}")

        # Step 3: Evidence governance
        steps.append(f"3. EVIDENCE GOVERNANCE: Evidence quality assessed as {evidence_report.overall_quality.value}. "
                      f"{evidence_report.reasoning}")
        if evidence_report.governance_concerns:
            steps.append(f"   CONCERNS: {'; '.join(evidence_report.governance_concerns)}")

        # Step 4: Coverage
        steps.append(f"4. COVERAGE ANALYSIS: {coverage.coverage_status}. "
                      f"{coverage.claims_covered} covered, {coverage.claims_uncovered} uncovered, "
                      f"{coverage.claims_partially_covered} partially covered out of {coverage.total_claims} total.")

        # Step 5: Conflicts
        active_conflicts = [c for c in conflicts if c.has_conflict]
        if active_conflicts:
            for i, c in enumerate(active_conflicts):
                steps.append(f"5.{i+1}. CONFLICT [{c.conflict_type.value}]: {c.implication}")
        else:
            steps.append("5. CONFLICT ANALYSIS: No evidence conflicts detected.")

        # Step 6: Consensus
        steps.append(f"6. SOURCE CONSENSUS: {consensus.consensus_status}. {consensus.reasoning}")

        # Step 7: Memory
        if memory.has_historical_context:
            if memory.memory_concerns:
                steps.append(f"7. MEMORY INTELLIGENCE: {'; '.join(memory.memory_concerns)}")
            else:
                steps.append("7. MEMORY INTELLIGENCE: No historical concerns flagged.")
        else:
            steps.append("7. MEMORY INTELLIGENCE: No historical context available from Memory Agent.")

        # Step 8: Risk
        steps.append(f"8. RISK ASSESSMENT: {risk.risk_level.value}. {risk.reasoning}")

        # Step 9: Final decision
        steps.append(f"9. FINAL DECISION: {decision.value} (Severity: {severity.value}).")

        # Build alternatives
        alts = {}
        if decision != Decision.ACCEPT:
            alts["ACCEPT"] = "Rejected: Evidence, coverage, or risk level insufficient for safe release under current domain policy."
        if decision != Decision.CORRECT:
            alts["CORRECT"] = "Not selected: No fixable hallucinated claims identified, or conflicts are too severe for correction."
        if decision != Decision.REJECT:
            alts["REJECT"] = "Not selected: Conflicts are not safety-critical or evidence provides some grounding."
        if decision != Decision.VERIFY_AGAIN:
            alts["VERIFY_AGAIN"] = "Not selected: Evidence is either conclusive or entirely absent (retry would not help)."
        if decision != Decision.ABSTAIN:
            alts["ABSTAIN"] = "Not selected: Sufficient information exists to make an actionable determination."

        return ReasoningChain(
            steps=steps, alternatives_considered=alts,
            governing_policy=f"{domain_policy.domain_name} ({domain_policy.strictness_level})",
            final_verdict=f"{decision.value} — {severity.value}"
        )

    def build_audit_record(
        self,
        decision: Decision,
        severity: Severity,
        reasoning: ReasoningChain,
        domain_policy: DomainPolicy,
        evidence_report: EvidenceSetGovernanceReport,
        runtime: RuntimeInspectionReport,
        risk: RiskAssessment,
        user_query: str,
        draft_response: str
    ) -> AuditRecord:
        return AuditRecord(
            audit_id=f"AUDIT_{uuid.uuid4().hex[:12].upper()}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            user_query=user_query,
            draft_response=draft_response,
            domain=domain_policy.domain_name,
            final_decision=decision.value,
            severity=severity.value,
            reasoning_chain=reasoning.steps,
            alternatives_rejected=reasoning.alternatives_considered,
            evidence_summary={
                "quality": evidence_report.overall_quality.value,
                "total_items": evidence_report.total_items,
                "authoritative_count": evidence_report.authoritative_count,
                "sufficient": evidence_report.is_sufficient_for_decision,
                "concerns": evidence_report.governance_concerns
            },
            runtime_health=runtime.system_health.value,
            risk_summary={
                "level": risk.risk_level.value,
                "safe_to_release": risk.is_safe_to_release,
                "human_review_required": risk.requires_human_review,
                "factors": risk.risk_factors,
                "mitigating": risk.mitigating_factors
            }
        )
