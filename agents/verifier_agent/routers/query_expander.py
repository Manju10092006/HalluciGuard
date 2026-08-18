from __future__ import annotations
import json
import logging
from pathlib import Path
import re
from typing import Dict, List, Tuple

from claims.entity_resolver import EntityResolver, EntityResolution

logger = logging.getLogger(__name__)


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
            logger.warning(f"Query expansion config dir not found at {config_dir}")
            return

        for json_file in config_dir.glob("*.json"):
            domain = json_file.stem.lower()
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.synonyms[domain] = data.get("synonyms", {})
                    self.domain_terms[domain] = data.get("domain_terms", [])
            except Exception as e:
                logger.warning(f"Failed to load query expansion config {json_file}: {e}")

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

        # Canonical entities improve source matching, but must not replace the
        # rest of a factual claim: predicates, conditions, and outcomes carry
        # the verification intent.
        if resolution.canonical_query and resolution.primary_entity:
            canonical = resolution.canonical_query.strip()
            normalized_query = " ".join(query.split())
            if canonical.lower() in normalized_query.lower():
                base_query = normalized_query
            else:
                base_query = f"{canonical} {normalized_query}"
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

    def generate_search_queries(self, query: str, domain: str) -> List[str]:
        """
        Generate primary and expanded search queries preserving the semantic subject.
        Returns a list of 1-3 distinct search query strings.
        """
        clean_q = " ".join((query or "").split())
        if not clean_q:
            return []

        queries = [clean_q]

        resolution = self.entity_resolver.resolve(clean_q, domain)
        if resolution.canonical_query and resolution.canonical_query.lower() != clean_q.lower():
            queries.append(resolution.canonical_query)

        # Subject-predicate expansion for relationship claims (e.g. 'X was created by Y' -> 'X creator', 'X created by')
        created_match = re.search(
            r"^([A-Za-z0-9\s]+?)\s+(?:was\s+)?(?:created|developed|invented|founded|written|authored|discovered)\s+by\s+",
            clean_q,
            re.IGNORECASE,
        )
        if created_match:
            subj = created_match.group(1).strip()
            if subj and len(subj) > 2:
                queries.append(f"{subj} created by")

        capital_match = re.search(
            r"^([A-Za-z0-9\s]+?)\s+is\s+the\s+capital\s+of\s+([A-Za-z0-9\s]+)",
            clean_q,
            re.IGNORECASE,
        )
        if capital_match:
            country = capital_match.group(2).strip()
            queries.append(f"capital of {country}")

        location_match = re.search(
            r"^([A-Za-z0-9\s]+?)\s+is\s+(?:located\s+in|in)\s+([A-Za-z0-9\s]+)",
            clean_q,
            re.IGNORECASE,
        )
        if location_match:
            landmark = location_match.group(1).strip()
            queries.append(f"{landmark} location")

        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            qn = q.strip().lower()
            if qn and qn not in seen:
                seen.add(qn)
                unique_queries.append(q.strip())

        return unique_queries[:3]

    def expand(self, query: str, domain: str) -> str:
        """Helper returning just the expanded query string for backward compatibility."""
        expanded, _ = self.resolve_and_expand(query, domain)
        return expanded
