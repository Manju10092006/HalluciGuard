"""
HalluciGuard - Source Reliability Analyzer
Classifies sources into governance tiers and assesses independence/diversity.
"""

from typing import List, Tuple, Dict
from config import SourceTier
from decision_policies import DomainPolicy


class SourceReliabilityAnalyzer:
    _TIER_KEYWORDS = {
        SourceTier.OFFICIAL_STANDARD: [
            "pubmed", "nih", "fda", "who", "cdc", "sec edgar", "mitre", "nvd",
            "cve", "cisa", "gov", "cochrane", "court acts", "statute", "government"
        ],
        SourceTier.PEER_REVIEWED: [
            "nature", "lancet", "ieee", "arxiv", "journal", "academic",
            "university", "doi", "researchgate", "springer", "elsevier"
        ],
        SourceTier.ENTERPRISE_VENDOR: [
            "10-k", "annual report", "vendor", "microsoft docs", "aws docs",
            "google cloud", "official documentation", "enterprise"
        ],
        SourceTier.REPUTABLE_NEWS: [
            "reuters", "bloomberg", "wall street journal", "bbc",
            "associated press", "financial times", "wsj"
        ],
        SourceTier.COMMUNITY: [
            "wikipedia", "stackoverflow", "github", "reddit", "forum",
            "wiki", "medium", "quora"
        ],
    }

    def classify_source(self, source_name: str) -> SourceTier:
        if not source_name:
            return SourceTier.UNVERIFIED
        src = source_name.lower()
        for tier, keywords in self._TIER_KEYWORDS.items():
            if any(kw in src for kw in keywords):
                return tier
        return SourceTier.UNVERIFIED

    def assess_source_independence(self, sources: List[str]) -> Dict:
        tiers = [self.classify_source(s) for s in sources]
        unique_tiers = set(tiers)
        unique_providers = set(s.strip().lower() for s in sources if s)
        return {
            "unique_tiers": len(unique_tiers),
            "unique_providers": len(unique_providers),
            "is_diverse": len(unique_tiers) >= 2 or len(unique_providers) >= 2,
            "tier_breakdown": {t.value: tiers.count(t) for t in unique_tiers}
        }

    def is_source_acceptable(self, source_name: str, policy: DomainPolicy) -> Tuple[bool, str]:
        tier = self.classify_source(source_name)
        if tier in policy.required_source_tiers:
            return True, f"Source '{source_name}' classified as {tier.value}, meets {policy.domain_name} policy."
        if tier == SourceTier.COMMUNITY and not policy.allows_community_evidence:
            return False, f"Source '{source_name}' is community content, rejected under {policy.domain_name} policy."
        if tier == SourceTier.UNVERIFIED and policy.strictness_level in ("VERY_STRICT", "STRICT"):
            return False, f"Source '{source_name}' is unverified, rejected under {policy.strictness_level} policy."
        return True, f"Source '{source_name}' ({tier.value}) accepted as supplementary under {policy.domain_name} policy."
