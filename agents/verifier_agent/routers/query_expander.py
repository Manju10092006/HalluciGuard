from __future__ import annotations
import json
import logging
from pathlib import Path
import re
from typing import Dict, List

class QueryExpander:
    """Expands queries with domain-specific synonyms and terms from config JSON files."""

    def __init__(self) -> None:
        self.synonyms: Dict[str, Dict[str, str]] = {}
        self.domain_terms: Dict[str, List[str]] = {}
        self._load_configs()

    def _load_configs(self) -> None:
        config_dir = Path(__file__).parent.parent / "config" / "query_expansion"
        if not config_dir.exists():
            logging.warning(f"Query expansion config dir not found at {config_dir}")
            return

        for json_file in config_dir.glob("*.json"):
            domain = json_file.stem.lower()
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.synonyms[domain] = data.get("synonyms", {})
                    self.domain_terms[domain] = data.get("domain_terms", [])
            except Exception as e:
                logging.warning(f"Failed to load query expansion config {json_file}: {e}")

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

        # Add domain terms if available
        if domain_key in self.domain_terms and self.domain_terms[domain_key]:
            term = self.domain_terms[domain_key][0]
            expanded_query += f" {term}"

        # Clean up whitespace
        return ' '.join(expanded_query.split())
