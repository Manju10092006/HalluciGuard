"""
HalluciGuard - Workflow Orchestrator
Dynamically controls what happens NEXT based on the Judge's reasoning.
Generates structured actions targeting specific agents.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from config import Decision, Severity
from decision_policies import DomainPolicy
from evidence_governance import EvidenceSetGovernanceReport
from conflict_resolver import ConflictReport
from risk_intelligence import RiskAssessment


@dataclass
class WorkflowAction:
    """Recommended workflow action from the orchestrator."""
    action_type: str
    target_agent: str
    instructions: Dict[str, Any]
    priority: str
    reasoning: str


class WorkflowOrchestrator:
    """Orchestrates workflow decisions based on judge outcomes."""
    # Domain -> required authoritative sources
    _DOMAIN_SOURCES = {
        "Healthcare": ["PubMed", "FDA", "WHO", "NIH", "Cochrane"],
        "Cybersecurity": ["MITRE ATT&CK", "NVD", "CVE", "CISA"],
        "Finance": ["SEC EDGAR", "Bloomberg", "Annual 10-K Reports"],
        "Law": ["Government Acts", "Court Judgements", "Statutory Repositories"],
        "Scientific Research": ["IEEE Xplore", "arXiv", "Nature / Science"],
    }

    def determine_next_action(
        self,
        decision: Decision,
        risk: RiskAssessment,
        evidence_report: EvidenceSetGovernanceReport,
        conflicts: List[ConflictReport],
        domain_policy: DomainPolicy,
        retry_count: int = 0
    ) -> WorkflowAction:
        """Determine the next workflow action based on the current state."""

        if decision == Decision.ACCEPT:
            return WorkflowAction(
                action_type="ACCEPT_RELEASE", target_agent="USER",
                instructions={"action": "Release verified response to user."},
                priority="NORMAL",
                reasoning="Response verified. Evidence is sufficient, no conflicts detected, and risk is acceptable."
            )

        if decision == Decision.REJECT:
            return WorkflowAction(
                action_type="REJECT_BLOCK", target_agent="USER",
                instructions={
                    "action": "Block draft response.",
                    "safe_fallback": "The system could not verify this response. Please consult an authoritative source.",
                    "blocked_reason": risk.reasoning
                },
                priority="CRITICAL",
                reasoning=f"Response blocked due to: {risk.reasoning}"
            )

        if decision == Decision.CORRECT:
            hallucinated = [c.affected_claim for c in conflicts if c.has_conflict]
            trusted = [c.conflicting_evidence for c in conflicts if c.has_conflict and c.conflicting_evidence]
            return WorkflowAction(
                action_type="REQUEST_CORRECTION", target_agent="CORRECTOR",
                instructions={
                    "action": "Repair hallucinated claims using trusted evidence.",
                    "hallucinated_claims": hallucinated,
                    "trusted_evidence": trusted,
                    "guidance": "Replace contradicted claims with evidence-grounded alternatives. Preserve non-hallucinated content."
                },
                priority="HIGH",
                reasoning="Response contains fixable hallucinations. Forwarding specific claims and evidence to Corrector Agent."
            )

        if decision == Decision.VERIFY_AGAIN:
            can_retry = retry_count < 2 and domain_policy.retry_on_insufficient_evidence
            if not can_retry:
                return WorkflowAction(
                    action_type="TERMINATE_ABSTAIN", target_agent="USER",
                    instructions={"action": "Retries exhausted. Unable to verify.", "retry_count": retry_count},
                    priority="HIGH",
                    reasoning=f"Re-verification retries exhausted ({retry_count}). Cannot safely release unverified response."
                )

            required_sources = self._DOMAIN_SOURCES.get(domain_policy.domain_name, ["Enterprise KB"])
            current_sources = [a.source_name for a in evidence_report.individual_assessments]
            missing = [s for s in required_sources if not any(s.lower() in c.lower() for c in current_sources)]

            return WorkflowAction(
                action_type="REQUEST_REVERIFICATION", target_agent="VERIFIER",
                instructions={
                    "action": "Perform targeted re-verification with expanded sources.",
                    "required_sources": missing or required_sources,
                    "search_depth": "deep" if domain_policy.strictness_level in ("VERY_STRICT", "STRICT") else "standard",
                    "max_results": 20,
                    "priority": "high",
                    "domain_policy": domain_policy.domain_name,
                    "retry_number": retry_count + 1,
                    "evidence_gaps": evidence_report.missing_required_source_types
                },
                priority="HIGH",
                reasoning=f"Evidence insufficient under {domain_policy.domain_name} policy. Requesting expanded retrieval from {missing or required_sources}."
            )

        if decision == Decision.ESCALATE_HUMAN:
            return WorkflowAction(
                action_type="ESCALATE_HUMAN", target_agent="HUMAN_REVIEWER",
                instructions={
                    "action": "Request human expert review.",
                    "reason": risk.reasoning,
                    "risk_level": risk.risk_level.value,
                    "domain": domain_policy.domain_name,
                    "escalation_policy": domain_policy.human_review_description
                },
                priority="CRITICAL",
                reasoning=f"Automated verification cannot resolve this safely. Escalating to human reviewer per {domain_policy.domain_name} policy."
            )

        # ABSTAIN fallback
        return WorkflowAction(
            action_type="TERMINATE_ABSTAIN", target_agent="USER",
            instructions={"action": "Unable to verify response.", "reason": "Insufficient evidence and context."},
            priority="NORMAL",
            reasoning="Judge abstains from decision due to insufficient grounding for any actionable determination."
        )
