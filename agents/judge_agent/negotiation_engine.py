"""
HalluciGuard - Negotiation Engine
Generates structured, traceable negotiation messages for other agents.
Every cross-agent communication is identified, timestamped, and auditable.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from decision_policies import DomainPolicy
from risk_intelligence import RiskAssessment


@dataclass
class NegotiationMessage:
    """Message exchanged between agents during negotiation."""
    message_id: str
    sender: str
    recipient: str
    action_requested: str
    context: Dict[str, Any]
    requirements: Dict[str, Any]
    priority: str
    reasoning: str
    timestamp: str


class NegotiationEngine:
    """Engine for inter-agent negotiation and coordination."""
    def request_reverification(
        self,
        domain_policy: DomainPolicy,
        missing_sources: List[str],
        unverified_claims: List[str]
    ) -> NegotiationMessage:
        """Request re-verification of a claim from the verifier."""
        return NegotiationMessage(
            message_id=f"NEG_{uuid.uuid4().hex[:10].upper()}",
            sender="JUDGE_AGENT", recipient="VERIFIER_AGENT",
            action_requested="REVERIFICATION",
            context={
                "domain": domain_policy.domain_name,
                "strictness": domain_policy.strictness_level,
                "unverified_claims": unverified_claims[:10],
                "claim_count": len(unverified_claims)
            },
            requirements={
                "target_sources": missing_sources,
                "search_depth": "deep",
                "max_results_per_claim": 10,
                "require_primary_sources": domain_policy.requires_primary_sources,
                "freshness_constraint": domain_policy.freshness_sensitivity
            },
            priority="HIGH",
            reasoning=f"Evidence insufficient for {domain_policy.domain_name} governance. "
                      f"Requesting targeted retrieval from {missing_sources}.",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )

    def request_correction(
        self,
        hallucinated_claims: List[str],
        trusted_evidence: List[str],
        correction_guidance: str = ""
    ) -> NegotiationMessage:
        """Request correction of hallucinated content."""
        return NegotiationMessage(
            message_id=f"NEG_{uuid.uuid4().hex[:10].upper()}",
            sender="JUDGE_AGENT", recipient="CORRECTOR_AGENT",
            action_requested="CORRECTION",
            context={
                "hallucinated_claims": hallucinated_claims,
                "claim_count": len(hallucinated_claims)
            },
            requirements={
                "trusted_evidence": trusted_evidence,
                "strategy": "Replace contradicted claims with evidence-grounded alternatives.",
                "preserve_non_hallucinated": True,
                "additional_guidance": correction_guidance
            },
            priority="HIGH",
            reasoning="Judge identified fixable hallucinations. Corrector must replace specific claims with evidence-grounded text.",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )

    def request_memory_lookup(
        self,
        claim: str,
        query_type: str = "hallucination_history"
    ) -> NegotiationMessage:
        """Request additional context from the memory agent."""
        return NegotiationMessage(
            message_id=f"NEG_{uuid.uuid4().hex[:10].upper()}",
            sender="JUDGE_AGENT", recipient="MEMORY_AGENT",
            action_requested="MEMORY_LOOKUP",
            context={"claim": claim, "query_type": query_type},
            requirements={"return_format": "structured", "include_reliability_scores": True},
            priority="NORMAL",
            reasoning=f"Requesting historical context for claim: '{claim[:100]}...'",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )

    def escalate_to_human(
        self,
        full_context: Dict[str, Any],
        risk: RiskAssessment,
        escalation_reason: str
    ) -> NegotiationMessage:
        """Escalate a decision to human review."""
        return NegotiationMessage(
            message_id=f"NEG_{uuid.uuid4().hex[:10].upper()}",
            sender="JUDGE_AGENT", recipient="HUMAN_REVIEWER",
            action_requested="HUMAN_ESCALATION",
            context=full_context,
            requirements={
                "risk_level": risk.risk_level.value,
                "risk_factors": risk.risk_factors,
                "requires_response": True
            },
            priority="CRITICAL",
            reasoning=escalation_reason,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
