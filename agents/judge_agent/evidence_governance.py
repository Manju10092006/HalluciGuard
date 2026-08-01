"""
HalluciGuard - Evidence Governance Engine
Evaluates evidence from a GOVERNANCE perspective. Does NOT score evidence numerically.
Answers: Is evidence authoritative? Independent? Diverse? Does it cover the claim? Is critical evidence missing?
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from config import EvidenceQuality, SourceTier
from decision_policies import DomainPolicy
from source_reliability import SourceReliabilityAnalyzer


@dataclass
class EvidenceAssessment:
    source_name: str
    source_tier: SourceTier
    is_authoritative: bool
    is_acceptable_for_policy: bool
    acceptance_reasoning: str


@dataclass
class EvidenceSetGovernanceReport:
    overall_quality: EvidenceQuality
    total_items: int
    authoritative_count: int
    independent_source_count: int
    has_required_source_types: bool
    missing_required_source_types: List[str]
    individual_assessments: List[EvidenceAssessment]
    governance_concerns: List[str]
    is_sufficient_for_decision: bool
    reasoning: str


class EvidenceGovernanceEngine:
    def __init__(self):
        self._source_analyzer = SourceReliabilityAnalyzer()

    def assess_evidence_set(
        self,
        claim_evidence_pairs: List[Dict[str, Any]],
        domain_policy: DomainPolicy
    ) -> EvidenceSetGovernanceReport:
        if not claim_evidence_pairs:
            return EvidenceSetGovernanceReport(
                overall_quality=EvidenceQuality.ABSENT,
                total_items=0, authoritative_count=0, independent_source_count=0,
                has_required_source_types=False,
                missing_required_source_types=[t.value for t in domain_policy.required_source_tiers],
                individual_assessments=[], governance_concerns=["No evidence was provided by the Verifier Agent."],
                is_sufficient_for_decision=False,
                reasoning="Cannot make a governance determination without any evidence. The Verifier returned no grounding artifacts."
            )

        assessments = []
        concerns = []
        sources = []
        authoritative_count = 0

        for pair in claim_evidence_pairs:
            evidence_text = pair.get("evidence", pair.get("evidence_snippet", ""))
            source_name = pair.get("source", pair.get("evidence_source", "Unknown"))
            sources.append(source_name)

            if not evidence_text or not evidence_text.strip():
                concerns.append(f"Evidence item from '{source_name}' contains no substantive text.")
                continue

            tier = self._source_analyzer.classify_source(source_name)
            is_acceptable, reason = self._source_analyzer.is_source_acceptable(source_name, domain_policy)
            is_auth = tier in (SourceTier.OFFICIAL_STANDARD, SourceTier.PEER_REVIEWED)
            if is_auth:
                authoritative_count += 1

            assessments.append(EvidenceAssessment(
                source_name=source_name, source_tier=tier,
                is_authoritative=is_auth, is_acceptable_for_policy=is_acceptable,
                acceptance_reasoning=reason
            ))
            if not is_acceptable:
                concerns.append(reason)

        # Check source independence
        independence = self._source_analyzer.assess_source_independence(sources)
        independent_count = independence["unique_providers"]

        # Check required source types — policy lists ACCEPTABLE tiers (OR logic)
        # If ANY of the required tiers is present, the requirement is met
        present_tiers = set(a.source_tier for a in assessments)
        has_any_required = bool(present_tiers & set(domain_policy.required_source_tiers))
        missing_all_required = not has_any_required
        missing_required_list = [t.value for t in domain_policy.required_source_tiers if t not in present_tiers]

        if missing_all_required and domain_policy.strictness_level in ("VERY_STRICT", "STRICT"):
            concerns.append(f"{domain_policy.domain_name} policy requires sources from {[t.value for t in domain_policy.required_source_tiers]}, but none were present.")

        if not independence["is_diverse"] and domain_policy.strictness_level in ("VERY_STRICT", "STRICT") and len(assessments) >= 2:
            concerns.append("Evidence lacks diversity — all items originate from a single source category.")

        if independent_count < domain_policy.minimum_independent_sources:
            concerns.append(f"Policy requires at least {domain_policy.minimum_independent_sources} independent source(s), found {independent_count}.")

        # Determine overall quality through reasoning
        total = len(assessments)
        if total == 0:
            quality = EvidenceQuality.ABSENT
            sufficient = False
            reasoning = "No usable evidence items were provided."
        elif authoritative_count >= 2 and has_any_required and independence["is_diverse"]:
            quality = EvidenceQuality.AUTHORITATIVE
            sufficient = True
            reasoning = f"Evidence is authoritative ({authoritative_count} authoritative sources), meets required source types, and is independently diverse."
        elif authoritative_count >= 1 and has_any_required:
            quality = EvidenceQuality.STRONG
            sufficient = True
            reasoning = f"Evidence includes {authoritative_count} authoritative source(s) from an accepted tier."
        elif has_any_required:
            quality = EvidenceQuality.MODERATE
            sufficient = True
            reasoning = "Evidence meets required source type(s) but lacks authoritative classification."
        elif authoritative_count >= 1:
            quality = EvidenceQuality.MODERATE
            sufficient = domain_policy.strictness_level in ("MODERATE", "RELAXED")
            reasoning = "Evidence includes authoritative source(s) but not from the specifically required tier(s)."
        elif total >= 1:
            quality = EvidenceQuality.WEAK
            sufficient = domain_policy.strictness_level == "RELAXED"
            reasoning = "Evidence exists but does not meet domain authority requirements."
        else:
            quality = EvidenceQuality.INSUFFICIENT
            sufficient = False
            reasoning = "Evidence is insufficient for governance determination."

        return EvidenceSetGovernanceReport(
            overall_quality=quality, total_items=total,
            authoritative_count=authoritative_count,
            independent_source_count=independent_count,
            has_required_source_types=has_any_required,
            missing_required_source_types=missing_required_list,
            individual_assessments=assessments,
            governance_concerns=concerns,
            is_sufficient_for_decision=sufficient,
            reasoning=reasoning
        )
