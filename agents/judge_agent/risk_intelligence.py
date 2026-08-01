"""
HalluciGuard - Risk Intelligence Engine
Reasons about risk from a governance perspective. No numeric formulas.
Considers domain sensitivity, conflict severity, evidence gaps, and system health.
"""

from dataclasses import dataclass, field
from typing import List
from config import Severity, ConflictType, EvidenceQuality, SystemHealth
from decision_policies import DomainPolicy
from evidence_governance import EvidenceSetGovernanceReport
from coverage_analyzer import CoverageReport
from conflict_resolver import ConflictReport
from consensus_analyzer import ConsensusReport
from runtime_inspector import RuntimeInspectionReport


@dataclass
class RiskAssessment:
    risk_level: Severity
    risk_factors: List[str]
    mitigating_factors: List[str]
    is_safe_to_release: bool
    requires_human_review: bool
    reasoning: str


class RiskIntelligenceEngine:
    def assess_risk(
        self,
        domain_policy: DomainPolicy,
        evidence_report: EvidenceSetGovernanceReport,
        coverage_report: CoverageReport,
        conflicts: List[ConflictReport],
        consensus: ConsensusReport,
        runtime_report: RuntimeInspectionReport
    ) -> RiskAssessment:
        risk_factors = []
        mitigating = []

        # --- Collect risk factors ---
        has_safety_conflict = any(c.is_safety_critical for c in conflicts if c.has_conflict)
        has_any_conflict = any(c.has_conflict for c in conflicts)
        has_immediate_action = any(c.requires_immediate_action for c in conflicts if c.has_conflict)

        if has_safety_conflict:
            risk_factors.append(f"Safety-critical conflict detected in {domain_policy.domain_name} domain.")
        if has_any_conflict and not has_safety_conflict:
            conflict_types = [c.conflict_type.value for c in conflicts if c.has_conflict]
            risk_factors.append(f"Evidence conflicts detected: {', '.join(conflict_types)}.")

        if evidence_report.overall_quality in (EvidenceQuality.ABSENT, EvidenceQuality.INSUFFICIENT):
            risk_factors.append("Evidence is absent or insufficient for decision-making.")
        elif evidence_report.overall_quality == EvidenceQuality.WEAK:
            risk_factors.append("Evidence quality is weak — does not meet domain authority requirements.")

        if not evidence_report.has_required_source_types:
            risk_factors.append(f"Missing required source types: {evidence_report.missing_required_source_types}.")

        if coverage_report.coverage_status in ("NO_COVERAGE", "INSUFFICIENTLY_COVERED"):
            risk_factors.append(f"Evidence coverage is insufficient: {coverage_report.claims_uncovered} claim(s) uncovered.")

        if consensus.consensus_status == "DISAGREEMENT":
            risk_factors.append("Independent sources disagree about the claim.")

        if runtime_report.system_health != SystemHealth.HEALTHY:
            risk_factors.append(f"Verification pipeline is in {runtime_report.system_health.value} state: {runtime_report.reasoning}")

        if domain_policy.strictness_level == "VERY_STRICT":
            risk_factors.append(f"Domain '{domain_policy.domain_name}' enforces VERY_STRICT governance policy.")

        # --- Collect mitigating factors ---
        if evidence_report.overall_quality in (EvidenceQuality.AUTHORITATIVE, EvidenceQuality.STRONG):
            mitigating.append("Evidence quality is authoritative/strong and meets policy requirements.")

        if consensus.has_consensus and consensus.consensus_status in ("STRONG_AGREEMENT", "MAJORITY_AGREEMENT"):
            mitigating.append(f"Sources demonstrate {consensus.consensus_status.lower().replace('_', ' ')}.")

        if coverage_report.coverage_status == "FULLY_COVERED":
            mitigating.append("All claims are fully covered by retrieved evidence.")

        if runtime_report.is_pipeline_trustworthy:
            mitigating.append("Verification pipeline executed without degradation.")

        if not has_any_conflict:
            mitigating.append("No evidence conflicts detected.")

        # --- Determine risk level through reasoning ---
        if has_safety_conflict or has_immediate_action:
            level = Severity.CRITICAL
            safe = False
            human_review = domain_policy.escalate_on_safety_conflict
            reasoning = (f"CRITICAL RISK: Safety-critical conflict detected under {domain_policy.domain_name} policy. "
                         f"Response must not be released without correction or human review.")
        elif len(risk_factors) >= 4 or (has_any_conflict and domain_policy.strictness_level == "VERY_STRICT"):
            level = Severity.HIGH
            safe = False
            human_review = domain_policy.strictness_level in ("VERY_STRICT", "STRICT")
            reasoning = (f"HIGH RISK: Multiple risk factors identified ({len(risk_factors)}). "
                         f"Evidence and conflict analysis indicate response is not safe under {domain_policy.domain_name} governance.")
        elif len(risk_factors) >= 2:
            level = Severity.MEDIUM
            safe = len(mitigating) > len(risk_factors)
            human_review = False
            reasoning = (f"MEDIUM RISK: {len(risk_factors)} risk factor(s) identified but partially mitigated by "
                         f"{len(mitigating)} positive signal(s).")
        elif len(risk_factors) == 1:
            level = Severity.LOW
            safe = True
            human_review = False
            reasoning = f"LOW RISK: Single minor risk factor identified. Mitigating factors outweigh concerns."
        else:
            level = Severity.INFORMATIONAL
            safe = True
            human_review = False
            reasoning = "MINIMAL RISK: No significant risk factors. Evidence supports safe release."

        return RiskAssessment(
            risk_level=level, risk_factors=risk_factors, mitigating_factors=mitigating,
            is_safe_to_release=safe, requires_human_review=human_review, reasoning=reasoning
        )
