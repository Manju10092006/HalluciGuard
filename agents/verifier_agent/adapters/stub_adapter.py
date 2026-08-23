from __future__ import annotations

import logging
from typing import List

from schemas.models import Passage, AdapterMetadata

logger = logging.getLogger(__name__)

class StubAdapter:
    def __init__(self, domain_name: str) -> None:
        self.name = domain_name

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self.name,
            version="1.0.0",
            supported_domains=[],
            supports_live_search=False,
            cacheable=False,
            priority=0,
            max_results=0,
            is_stub=True
        )

    async def search(self, query: str, k: int = 5, **kwargs) -> List[Passage]:
        logger.info(f"{self.name} adapter not yet implemented")
        return []

    def credibility_of(self, source_id: str) -> float:
        return 0.50


STUB_DOMAINS = [
    "programming", "scientific", "education", "government", "news", 
    "mathematics", "physics", "chemistry", "biology", "space", 
    "history", "geography", "economics", "climate", "sports", 
    "business", "manufacturing", "pharmaceuticals"
]

def register_all_stubs(registry: 'AdapterRegistry') -> None:
    """Registers all stub domains."""
    for domain in STUB_DOMAINS:
        registry.register(StubAdapter(domain))
