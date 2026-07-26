from __future__ import annotations
import pytest
from adapters.registry import Registry, get_registry

def test_register_adapter():
    registry = Registry()
    class DummyAdapter:
        pass
    registry.register("dummy", DummyAdapter())
    adapter = registry.get_adapter("dummy")
    assert isinstance(adapter, DummyAdapter)

def test_get_nonexistent_domain():
    registry = get_registry()
    adapter = registry.get_adapter("nonexistent_domain_123")
    # Should fallback to general
    assert adapter is not None

def test_list_domains():
    registry = get_registry()
    domains = registry.list_domains()
    assert "healthcare" in domains
    assert "general" in domains

def test_domain_statistics():
    registry = get_registry()
    adapter = registry.get_adapter("general")
    stats = adapter.get_health_status()
    assert "status" in stats
