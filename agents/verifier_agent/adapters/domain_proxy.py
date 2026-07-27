from __future__ import annotations

from typing import List

from models.domain_intelligence import DomainProfile
from schemas.models import AdapterMetadata, Passage


class DomainProxyAdapter:
    """Domain-specific adapter facade backed by a production authoritative adapter."""

    def __init__(self, profile: DomainProfile, delegate: object) -> None:
        self.name = profile.domain
        self.profile = profile
        self.delegate = delegate

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self.name,
            version="2.0.0",
            supported_domains=self.profile.source_ids,
            supports_live_search=True,
            cacheable=True,
            priority=8,
            max_results=50,
            is_stub=False,
        )

    async def search(self, query: str, k: int = 5) -> List[Passage]:
        search = getattr(self.delegate, "search")
        return await search(query, k)

    def credibility_of(self, source_id: str) -> float:
        for source in self.profile.api_sources:
            if source_id.lower().startswith(source.id.lower()):
                return source.credibility
        if hasattr(self.delegate, "credibility_of"):
            return float(getattr(self.delegate, "credibility_of")(source_id))
        return 0.75
