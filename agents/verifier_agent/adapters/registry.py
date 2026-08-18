from __future__ import annotations

import os
from typing import Any, Dict, List, Protocol, Optional

from models.domain_intelligence import get_domain_intelligence_registry
from schemas.models import Passage, AdapterMetadata


class DomainAdapter(Protocol):
    """Protocol defining the interface for domain adapters."""
    name: str

    @property
    def metadata(self) -> AdapterMetadata:
        """Returns the metadata for this adapter."""
        ...

    async def search(self, query: str, k: int = 5) -> List[Passage]:
        """Search the domain sources for passages matching the query."""
        ...

    def credibility_of(self, source_id: str) -> float:
        """Returns the credibility score of a source in this domain."""
        ...


class AdapterRegistry:
    """Registry for managing domain adapters."""
    def __init__(self) -> None:
        self._adapters: Dict[str, DomainAdapter] = {}

    def register(self, adapter: DomainAdapter) -> None:
        """Registers a domain adapter."""
        self._adapters[adapter.name] = adapter

    def get_adapter(self, domain: str) -> DomainAdapter:
        """Gets an adapter by domain name, respecting domain specialization without silent general fallback."""
        domain_registry = get_domain_intelligence_registry()
        canonical_domain = domain_registry.canonicalize(domain)
        if canonical_domain in self._adapters:
            return self._adapters[canonical_domain]
        if domain in self._adapters:
            return self._adapters[domain]

        # Specialized domain explicitly requested but missing adapter -> return StubAdapter for observable reporting
        if domain.lower() in ("healthcare", "ai_research", "finance", "cybersecurity", "legal"):
            from .stub_adapter import StubAdapter
            return StubAdapter(domain)

        adapter = self._adapters.get("general")
        if adapter is None:
            from .stub_adapter import StubAdapter
            return StubAdapter(domain)
        return adapter

    def list_domains(self) -> List[str]:
        """Lists all registered domain names."""
        return list(self._adapters.keys())

    def get_domain_statistics(self, domain: str) -> Dict[str, Any]:
        """Gets statistics for a specific domain adapter."""
        domain_registry = get_domain_intelligence_registry()
        canonical_domain = domain_registry.canonicalize(domain)
        adapter = self.get_adapter(canonical_domain)
        if not adapter:
            return {}
        profile = domain_registry.get_profile(canonical_domain)
        return {
            "domain": canonical_domain,
            "sources": profile.source_ids,
            "status": "active" if not adapter.metadata.is_stub else "stub",
            "credibility_scores": {source.id: source.credibility for source in profile.api_sources},
            "is_implemented": not adapter.metadata.is_stub
        }


    def get_all_statistics(self) -> List[Dict[str, Any]]:
        """Gets statistics for all domain adapters."""
        return [self.get_domain_statistics(domain) for domain in self.list_domains()]


_REGISTRY: Optional[AdapterRegistry] = None

def get_registry() -> AdapterRegistry:
    """Gets the singleton adapter registry, initializing and auto-registering if needed."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = AdapterRegistry()
        
        # Auto-register adapters
        from .general import GeneralAdapter
        from .healthcare import HealthcareAdapter
        from .cybersecurity import CybersecurityAdapter
        from .finance import FinanceAdapter
        from .legal_general import LegalGeneralAdapter
        from .ai_research import AiResearchAdapter
        from .domain_proxy import DomainProxyAdapter
        
        _REGISTRY.register(GeneralAdapter())
        _REGISTRY.register(HealthcareAdapter())
        _REGISTRY.register(CybersecurityAdapter())
        _REGISTRY.register(FinanceAdapter())
        _REGISTRY.register(LegalGeneralAdapter())
        _REGISTRY.register(AiResearchAdapter())

        domain_registry = get_domain_intelligence_registry()
        for profile in domain_registry.all_profiles():
            if profile.domain in _REGISTRY._adapters:
                continue
            delegate = _REGISTRY._adapters.get(profile.adapter) or _REGISTRY._adapters["general"]
            _REGISTRY.register(DomainProxyAdapter(profile, delegate))

    # ── Dynamic Web-Enhanced wrapping (opt-in via TAVILY_API_KEY) ──────
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if tavily_key:
        from .web_enhanced import WebEnhancedAdapter
        for domain_name in list(_REGISTRY._adapters.keys()):
            original = _REGISTRY._adapters[domain_name]
            if isinstance(original, WebEnhancedAdapter):
                continue
            if getattr(getattr(original, "metadata", None), "is_stub", False):
                continue
            _REGISTRY._adapters[domain_name] = WebEnhancedAdapter(
                primary_adapter=original,
                tavily_api_key=tavily_key,
            )

    return _REGISTRY
