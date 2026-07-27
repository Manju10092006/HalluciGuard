from __future__ import annotations
import json
import logging
from pathlib import Path
import re
from typing import Dict, List, Tuple

from claims.entity_resolver import EntityResolver, EntityResolution


class QueryExpander:
    """Expands queries with entity resolution, domain-specific synonyms, and terms."""

    def __init__(self) -> None:
        self.synonyms: Dict[str, Dict[str, str]] = {}
        self.domain_terms: Dict[str, List[str]] = {}
        self.entity_resolver = EntityResolver()
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

    def resolve_and_expand(self, query: str, domain: str) -> Tuple[str, EntityResolution]:
        """
        Extract entities and produce an entity-aware expanded query string.

        Args:
            query: The input sub-claim or query string.
            domain: Target domain.

        Returns:
            Tuple of (expanded_query, entity_resolution)
        """
        resolution = self.entity_resolver.resolve(query, domain)
        domain_key = domain.lower()

        # If a canonical entity query was resolved (e.g. CVE-2021-44228 or Metformin type 2 diabetes),
        # start with the canonical query!
        if resolution.canonical_query and resolution.primary_entity:
            base_query = resolution.canonical_query
        else:
            base_query = query

        expanded_query = base_query

        # Check domain abbreviation lookups
        if domain_key in self.synonyms:
            for abbr, expansion in self.synonyms[domain_key].items():
                pattern = re.compile(rf"\b{re.escape(abbr)}\b", re.IGNORECASE)
                if pattern.search(query):
                    expanded_query += f" {expansion}"

        # Clean up whitespace
        clean_query = " ".join(expanded_query.split())
        return clean_query, resolution

    def expand(self, query: str, domain: str) -> str:
        """Helper returning just the expanded query string for backward compatibility."""
        expanded, _ = self.resolve_and_expand(query, domain)
        return expanded
