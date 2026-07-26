from __future__ import annotations

import json
import logging
import os
from typing import List

from schemas.models import Passage, AdapterMetadata
from .registry import DomainAdapter

logger = logging.getLogger(__name__)

class MockAdapter:
    """Mock adapter for offline/CI testing."""
    def __init__(self, wrapped_adapter: DomainAdapter) -> None:
        self.wrapped = wrapped_adapter
        self.name = wrapped_adapter.name
        self._fixture_path = os.path.join(
            os.path.dirname(__file__), "fixtures", f"{self.name}.json"
        )

    @property
    def metadata(self) -> AdapterMetadata:
        return self.wrapped.metadata

    async def search(self, query: str, k: int = 5) -> List[Passage]:
        try:
            if os.path.exists(self._fixture_path):
                with open(self._fixture_path, "r") as f:
                    data = json.load(f)
                    passages_data = data.get("passages", [])
                    
                    # Simple mock search filtering by query if possible
                    query_lower = query.lower()
                    filtered = [p for p in passages_data if query_lower in p.get("text", "").lower()]
                    
                    if not filtered:
                        filtered = passages_data # Fallback to all if no match
                        
                    return [
                        Passage(
                            text=p["text"],
                            source_id=p["source_id"],
                            source_url=p["source_url"],
                            relevance_score=p.get("relevance_score", 0.9)
                        ) for p in filtered[:k]
                    ]
        except Exception as e:
            logger.error(f"Error reading mock fixture for {self.name}: {e}")
        return []

    def credibility_of(self, source_id: str) -> float:
        try:
            if os.path.exists(self._fixture_path):
                with open(self._fixture_path, "r") as f:
                    data = json.load(f)
                    scores = data.get("credibility_scores", {})
                    if source_id in scores:
                        return float(scores[source_id])
        except Exception as e:
            logger.error(f"Error reading credibility from fixture {self.name}: {e}")
        return 0.8
