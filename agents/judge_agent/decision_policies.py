"""
HalluciGuard - Domain Decision Policies
Defines governance BEHAVIOR per domain, not numeric thresholds.
Each policy describes what evidence is required, what sources are mandatory,
when to retry, when to escalate, and what constitutes sufficient coverage.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from config import SourceTier


@dataclass
class DomainPolicy:
    domain_name: str
    strictness_level: str  # VERY_STRICT, STRICT, MODERATE, RELAXED
    required_source_tiers: List[SourceTier]
    minimum_independent_sources: int
    requires_primary_sources: bool
    allows_community_evidence: bool
    requires_temporal_relevance: bool
    freshness_sensitivity: str  # HIGH, MEDIUM, LOW
    critical_claim_categories: List[str]
    retry_on_insufficient_evidence: bool
    escalate_on_safety_conflict: bool
    human_review_description: str
    description: str


class DomainPolicyRegistry:
    def __init__(self):
        self._policies: Dict[str, DomainPolicy] = {
            "Healthcare": DomainPolicy(
                domain_name="Healthcare",
                strictness_level="VERY_STRICT",
                required_source_tiers=[SourceTier.OFFICIAL_STANDARD, SourceTier.PEER_REVIEWED],
                minimum_independent_sources=1,
                requires_primary_sources=True,
                allows_community_evidence=False,
                requires_temporal_relevance=True,
                freshness_sensitivity="HIGH",
                critical_claim_categories=["dosage", "contraindicated", "side effect", "drug interaction", "treatment", "diagnosis", "fatal", "pediatric", "pregnancy"],
                retry_on_insufficient_evidence=True,
                escalate_on_safety_conflict=True,
                human_review_description="Any drug safety, dosage, or contraindication conflict requires immediate human clinical review.",
                description="Zero tolerance for unverified medical claims. Official or peer-reviewed sources mandatory. Community evidence rejected for safety claims."
            ),
            "Cybersecurity": DomainPolicy(
                domain_name="Cybersecurity",
                strictness_level="STRICT",
                required_source_tiers=[SourceTier.OFFICIAL_STANDARD, SourceTier.ENTERPRISE_VENDOR],
                minimum_independent_sources=1,
                requires_primary_sources=True,
                allows_community_evidence=True,
                requires_temporal_relevance=True,
                freshness_sensitivity="HIGH",
                critical_claim_categories=["cve", "exploit", "vulnerability", "remote code execution", "zero-day", "patch", "ransomware"],
                retry_on_insufficient_evidence=True,
                escalate_on_safety_conflict=True,
                human_review_description="Unverified exploit or vulnerability claims should be flagged for SOC analyst review.",
                description="Rapid freshness decay. CVE and exploit claims require MITRE/NVD/vendor confirmation. Community evidence accepted as supplementary only."
            ),
            "Finance": DomainPolicy(
                domain_name="Finance",
                strictness_level="STRICT",
                required_source_tiers=[SourceTier.OFFICIAL_STANDARD, SourceTier.ENTERPRISE_VENDOR],
                minimum_independent_sources=1,
                requires_primary_sources=True,
                allows_community_evidence=False,
                requires_temporal_relevance=True,
                freshness_sensitivity="MEDIUM",
                critical_claim_categories=["revenue", "net income", "ebitda", "sec", "10-k", "bankruptcy", "stock price", "dividend", "earnings"],
                retry_on_insufficient_evidence=True,
                escalate_on_safety_conflict=False,
                human_review_description="Financial metric discrepancies should be escalated to compliance review.",
                description="Financial data requires SEC filing or vendor terminal confirmation. Community evidence rejected."
            ),
            "Law": DomainPolicy(
                domain_name="Law",
                strictness_level="STRICT",
                required_source_tiers=[SourceTier.OFFICIAL_STANDARD, SourceTier.PEER_REVIEWED],
                minimum_independent_sources=1,
                requires_primary_sources=True,
                allows_community_evidence=False,
                requires_temporal_relevance=False,
                freshness_sensitivity="LOW",
                critical_claim_categories=["statute", "penalty", "jurisdiction", "indictment", "liability", "precedent"],
                retry_on_insufficient_evidence=True,
                escalate_on_safety_conflict=False,
                human_review_description="Legal interpretation disputes should be escalated to legal counsel.",
                description="Statutory and case law claims require official government or academic legal sources."
            ),
            "Scientific Research": DomainPolicy(
                domain_name="Scientific Research",
                strictness_level="MODERATE",
                required_source_tiers=[SourceTier.PEER_REVIEWED, SourceTier.OFFICIAL_STANDARD],
                minimum_independent_sources=1,
                requires_primary_sources=True,
                allows_community_evidence=True,
                requires_temporal_relevance=True,
                freshness_sensitivity="MEDIUM",
                critical_claim_categories=["constant", "formula", "theorem", "experiment", "publication"],
                retry_on_insufficient_evidence=True,
                escalate_on_safety_conflict=False,
                human_review_description="Conflicting research claims should be flagged for domain expert review.",
                description="Peer-reviewed publication grounding required. Community evidence accepted as supplementary."
            ),
            "General Knowledge": DomainPolicy(
                domain_name="General Knowledge",
                strictness_level="MODERATE",
                required_source_tiers=[SourceTier.COMMUNITY, SourceTier.REPUTABLE_NEWS],
                minimum_independent_sources=1,
                requires_primary_sources=False,
                allows_community_evidence=True,
                requires_temporal_relevance=False,
                freshness_sensitivity="LOW",
                critical_claim_categories=[],
                retry_on_insufficient_evidence=False,
                escalate_on_safety_conflict=False,
                human_review_description="Not typically required.",
                description="Balanced thresholds. Community and news sources accepted."
            ),
            "Entertainment": DomainPolicy(
                domain_name="Entertainment",
                strictness_level="RELAXED",
                required_source_tiers=[SourceTier.COMMUNITY, SourceTier.UNVERIFIED],
                minimum_independent_sources=0,
                requires_primary_sources=False,
                allows_community_evidence=True,
                requires_temporal_relevance=False,
                freshness_sensitivity="LOW",
                critical_claim_categories=[],
                retry_on_insufficient_evidence=False,
                escalate_on_safety_conflict=False,
                human_review_description="Not required.",
                description="Relaxed governance for low-risk conversational and entertainment queries."
            ),
        }

    def get_policy(self, domain_name: str) -> DomainPolicy:
        if not domain_name:
            return self._policies["General Knowledge"]
        for key, policy in self._policies.items():
            if key.lower() in domain_name.lower() or domain_name.lower() in key.lower():
                return policy
        return self._policies["General Knowledge"]


DEFAULT_POLICY_REGISTRY = DomainPolicyRegistry()
