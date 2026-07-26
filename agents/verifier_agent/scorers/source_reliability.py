from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict
from datetime import datetime

import yaml

class SourceReliabilityManager:
    """Manages source credibility scores loaded from config/credibility.yaml."""

    def __init__(self) -> None:
        self.credibility_db: Dict[str, Dict[str, float]] = {}
        self.default_credibility = 0.5
        self._load_config()

    def _load_config(self) -> None:
        config_path = Path(__file__).parent.parent / "config" / "credibility.yaml"
        if not config_path.exists():
            logging.warning(f"Credibility config not found at {config_path}")
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.credibility_db = yaml.safe_load(f) or {}
        except Exception as e:
            logging.warning(f"Failed to load credibility config {config_path}: {e}")

    def get_credibility(self, domain: str, source_id: str) -> float:
        """Get the credibility score for a source in a given domain."""
        domain_sources = self.credibility_db.get(domain.lower(), {})
        return domain_sources.get(source_id.lower(), self.default_credibility)

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
        Formula: max(0.5, 1.0 - (years_since_publication * 0.1))
        """
        try:
            date_str = str(publication_date)[:10]
            pub_date = datetime.strptime(date_str, '%Y-%m-%d')
            age_in_years = (datetime.now() - pub_date).days / 365.25
            return max(0.5, 1.0 - (age_in_years * 0.1))
        except (ValueError, TypeError):
            return 1.0

    def compute_domain_weight(self, domain: str, source_id: str, publication_date: str) -> float:
        """Compute combined weight from credibility and recency."""
        credibility = self.get_credibility(domain, source_id)
        recency = self.compute_recency_factor(publication_date)
        return credibility * recency
