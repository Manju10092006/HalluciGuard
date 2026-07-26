from __future__ import annotations

import logging
from typing import Any, Dict, List, Protocol, Optional

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
        """Gets an adapter by domain name, falling back to general."""
        if domain in self._adapters:
            return self._adapters[domain]
        return self._adapters.get("general")  # type: ignore

    def list_domains(self) -> List[str]:
        """Lists all registered domain names."""
        return list(self._adapters.keys())

    def get_domain_statistics(self, domain: str) -> Dict[str, Any]:
        """Gets statistics for a specific domain adapter."""
        adapter = self.get_adapter(domain)
        if not adapter:
            return {}
        return {
            "sources": adapter.metadata.supported_domains,
            "status": "active" if not adapter.metadata.is_stub else "stub",
            "credibility_scores": {}, # In real scenario, would be detailed
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
        from .stub_adapter import register_all_stubs
        
        _REGISTRY.register(GeneralAdapter())
        _REGISTRY.register(HealthcareAdapter())
        _REGISTRY.register(CybersecurityAdapter())
        _REGISTRY.register(FinanceAdapter())
        _REGISTRY.register(LegalGeneralAdapter())
        _REGISTRY.register(AiResearchAdapter())
        
        register_all_stubs(_REGISTRY)
        
    return _REGISTRY
