"""Domain adapters for multi-source evidence retrieval."""
from __future__ import annotations

from .registry import AdapterRegistry, get_registry

__all__ = ["AdapterRegistry", "get_registry"]
