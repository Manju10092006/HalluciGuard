"""
HalluciGuard - Domain-Aware Decision Policies
Defines domain-specific risk expectations, decision thresholds, minimum source authority requirements,
and temporal decay policies for Healthcare, Cybersecurity, Finance, Law, Research, and General domains.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DomainPolicy:
    domain_name: str
    accept_confidence_threshold: float
    correct_contradiction_threshold: float
    reject_contradiction_threshold: float
    min_required_authority: float
    freshness_half_life_days: float
    strictness_level: str  # "VERY_STRICT", "STRICT", "MODERATE", "RELAXED"
    description: str

class DomainPolicyRegistry:
    def __init__(self):
        self._policies: Dict[str, DomainPolicy] = {
            "Healthcare": DomainPolicy(
                domain_name="Healthcare",
                accept_confidence_threshold=0.88,
                correct_contradiction_threshold=0.35,
                reject_contradiction_threshold=0.60,
                min_required_authority=0.80, # Grounding requires peer-reviewed or govt authority
                freshness_half_life_days=365.0,
                strictness_level="VERY_STRICT",
                description="Zero tolerance for unverified medical claims; high authority required."
            ),
            "Cybersecurity": DomainPolicy(
                domain_name="Cybersecurity",
                accept_confidence_threshold=0.85,
                correct_contradiction_threshold=0.40,
                reject_contradiction_threshold=0.65,
                min_required_authority=0.75,
                freshness_half_life_days=30.0, # Rapid vulnerability decay
                strictness_level="STRICT",
                description="High freshness decay sensitivity; requires CVE/NVD/vendor authority."
            ),
            "Finance": DomainPolicy(
                domain_name="Finance",
                accept_confidence_threshold=0.85,
                correct_contradiction_threshold=0.40,
                reject_contradiction_threshold=0.65,
                min_required_authority=0.80,
                freshness_half_life_days=90.0,
                strictness_level="STRICT",
                description="Strict numeric and temporal accuracy; quarterly filing compliance."
            ),
            "Law": DomainPolicy(
                domain_name="Law",
                accept_confidence_threshold=0.87,
                correct_contradiction_threshold=0.38,
                reject_contradiction_threshold=0.62,
                min_required_authority=0.85,
                freshness_half_life_days=730.0,
                strictness_level="STRICT",
                description="High precedent & statutory authority requirements."
            ),
            "Scientific Research": DomainPolicy(
                domain_name="Scientific Research",
                accept_confidence_threshold=0.82,
                correct_contradiction_threshold=0.45,
                reject_contradiction_threshold=0.70,
                min_required_authority=0.70,
                freshness_half_life_days=365.0,
                strictness_level="MODERATE",
                description="Requires peer-reviewed publication grounding."
            ),
            "General Knowledge": DomainPolicy(
                domain_name="General Knowledge",
                accept_confidence_threshold=0.75,
                correct_contradiction_threshold=0.50,
                reject_contradiction_threshold=0.78,
                min_required_authority=0.40,
                freshness_half_life_days=1825.0,
                strictness_level="MODERATE",
                description="Balanced thresholds for general factual knowledge."
            ),
            "Entertainment": DomainPolicy(
                domain_name="Entertainment",
                accept_confidence_threshold=0.68,
                correct_contradiction_threshold=0.58,
                reject_contradiction_threshold=0.85,
                min_required_authority=0.20,
                freshness_half_life_days=365.0,
                strictness_level="RELAXED",
                description="Relaxed baseline for low-risk conversational queries."
            )
        }

    def get_policy(self, domain_name: str) -> DomainPolicy:
        """
        Returns policy for specified domain with fallback to General Knowledge.
        """
        normalized_name = domain_name.strip() if domain_name else "General Knowledge"
        for key in self._policies:
            if key.lower() in normalized_name.lower():
                return self._policies[key]
        return self._policies["General Knowledge"]

DEFAULT_DOMAIN_REGISTRY = DomainPolicyRegistry()
