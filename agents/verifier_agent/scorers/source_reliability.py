from __future__ import annotations
from typing import Dict
from datetime import datetime

class SourceReliabilityManager:
    """Manages source credibility scores."""

    def __init__(self) -> None:
        # In a real scenario, this would load from config/credibility.yaml
        self.credibility_db: Dict[str, Dict[str, float]] = {
            'healthcare': {
                'pubmed': 0.95,
                'cdc': 0.98,
                'openfda': 0.95
            },
            'finance': {
                'sec': 0.98,
                'bloomberg': 0.90
            }
        }
        self.default_credibility = 0.5

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
            # Handle standard ISO format or YYYY-MM-DD
            # Truncate to 10 chars to safely parse YYYY-MM-DD
            date_str = publication_date[:10]
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
