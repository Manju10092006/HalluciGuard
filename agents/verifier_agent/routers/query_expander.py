from __future__ import annotations
from typing import Dict, List
import re

class QueryExpander:
    """Expands queries with domain-specific synonyms and terms."""

    def __init__(self) -> None:
        # In a real setup, loads from config/query_expansion/*.json
        self.synonyms: Dict[str, Dict[str, str]] = {
            'healthcare': {
                't2dm': 'type 2 diabetes mellitus',
                'bp': 'blood pressure',
                'mi': 'myocardial infarction'
            }
        }
        self.domain_terms: Dict[str, List[str]] = {
            'healthcare': ['clinical trial', 'medical study', 'research']
        }

    def expand(self, query: str, domain: str) -> str:
        """
        Expand a query using abbreviations and domain terms.
        
        Args:
            query: The original query.
            domain: The domain of the query.
            
        Returns:
            The expanded query string.
        """
        domain_key = domain.lower()
        if domain_key not in self.synonyms:
            return query
            
        expanded_query = query
        
        # Check abbreviations
        for abbr, expansion in self.synonyms[domain_key].items():
            pattern = re.compile(rf'\b{re.escape(abbr)}\b', re.IGNORECASE)
            if pattern.search(query):
                expanded_query += f" {expansion}"
                
        # Add relevant domain terms (just one for simplicity, or random choice)
        if domain_key in self.domain_terms and self.domain_terms[domain_key]:
            term = self.domain_terms[domain_key][0]
            expanded_query += f" {term}"
            
        # Clean up whitespace
        expanded_query = ' '.join(expanded_query.split())
        return expanded_query
