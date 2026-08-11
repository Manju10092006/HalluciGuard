from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import yaml
from models.domain_intelligence import get_domain_intelligence_registry

logger = logging.getLogger(__name__)


class SourceReliabilityManager:
    """Manages source credibility scores loaded from config/credibility.yaml and domain intelligence."""

    def __init__(self) -> None:
        self.credibility_db: Dict[str, Dict[str, float]] = {}
        self.default_credibility = 0.80
        self._load_config()

    def _load_config(self) -> None:
        config_path = Path(__file__).parent.parent / "config" / "credibility.yaml"
        if not config_path.exists():
            logger.warning(f"Credibility config not found at {config_path}")
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.credibility_db = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load credibility config {config_path}: {e}")

    def get_credibility(self, domain: str, source_id: str) -> float:
        """
        Get the credibility score for a source in a given domain.
        First checks credibility.yaml, then DomainIntelligenceRegistry, then defaults.
        """
        domain_key = domain.lower()
        source_key = source_id.lower()

        # 1. Direct match in credibility.yaml
        domain_sources = self.credibility_db.get(domain_key, {})
        if source_key in domain_sources:
            return float(domain_sources[source_key])

        # Substring / Prefix match in credibility.yaml (e.g., "pubmed_central" -> "pubmed")
        for k, val in domain_sources.items():
            if k in source_key or source_key in k:
                return float(val)

        # 2. Check DomainIntelligenceRegistry
        try:
            registry = get_domain_intelligence_registry()
            profile = registry.get_profile(domain)
            for api_src in profile.api_sources:
                if api_src.id.lower() in source_key or source_key in api_src.id.lower():
                    return float(api_src.credibility)
        except Exception as e:
            logger.debug("DomainIntelligence lookup failed for %s/%s: %s", domain, source_id, e)

        # 3. Known authoritative source defaults
        if any(s in source_key for s in ["pubmed", "pmc", "openfda", "sec", "nvd", "cisa", "mitre", "clinicaltrials"]):
            return 0.95

        return self.default_credibility

    def get_domain_sources(self, domain: str) -> Dict[str, float]:
        """Get all source credibility scores for a domain."""
        return self.credibility_db.get(domain.lower(), {})

    def update_credibility(self, domain: str, source_id: str, new_score: float) -> None:
        """Update credibility score (in-memory only)."""
        domain_key = domain.lower()
        if domain_key not in self.credibility_db:
            self.credibility_db[domain_key] = {}
        self.credibility_db[domain_key][source_id.lower()] = max(0.0, min(1.0, new_score))

    def compute_recency_factor(self, publication_date: str) -> float:
        """
        Compute weight adjustment based on age.
        Formula: max(0.6, 1.0 - (years_since_publication * 0.05))
        """
        try:
            date_str = str(publication_date)[:10]
            pub_date = datetime.strptime(date_str, "%Y-%m-%d")
            age_in_years = (datetime.now() - pub_date).days / 365.25
            return max(0.6, 1.0 - (age_in_years * 0.05))
        except (ValueError, TypeError):
            return 0.85

    def compute_domain_weight(self, domain: str, source_id: str, publication_date: str) -> float:
        """Compute combined weight from credibility and recency."""
        credibility = self.get_credibility(domain, source_id)
        recency = self.compute_recency_factor(publication_date)
        return credibility * recency
