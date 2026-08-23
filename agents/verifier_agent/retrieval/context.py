"""
Request-local retrieval context for concurrent-safe adapter diagnostics.

Singleton adapters must NOT store per-request mutable state on ``self``.
All per-request retrieval diagnostics live in a ``RetrievalContext`` bound to
the current async task via ``contextvars``.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import List, Optional

from schemas.retrieval_trace import RetrievalTrace


@dataclass
class RetrievalContext:
    """Per-request retrieval diagnostics and trace state."""

    sources_attempted: List[str] = field(default_factory=list)
    sources_succeeded: List[str] = field(default_factory=list)
    sources_failed: List[str] = field(default_factory=list)
    generated_queries: List[str] = field(default_factory=list)
    last_retrieval_trace: Optional[RetrievalTrace] = None

    def reset(self) -> None:
        self.sources_attempted = []
        self.sources_succeeded = []
        self.sources_failed = []
        self.generated_queries = []
        self.last_retrieval_trace = None

    def record_attempt(self, source: str) -> None:
        if source and source not in self.sources_attempted:
            self.sources_attempted.append(source)

    def record_success(self, source: str) -> None:
        if source and source not in self.sources_succeeded:
            self.sources_succeeded.append(source)

    def record_failure(self, source: str) -> None:
        if source and source not in self.sources_failed:
            self.sources_failed.append(source)


_retrieval_context_var: ContextVar[Optional[RetrievalContext]] = ContextVar(
    "retrieval_context", default=None
)


def get_retrieval_context() -> RetrievalContext:
    """Return the active request-local retrieval context, creating one if needed."""
    ctx = _retrieval_context_var.get()
    if ctx is None:
        ctx = RetrievalContext()
        _retrieval_context_var.set(ctx)
    return ctx


def reset_retrieval_context() -> RetrievalContext:
    """Reset and bind a fresh retrieval context for a new verification request."""
    ctx = RetrievalContext()
    _retrieval_context_var.set(ctx)
    return ctx
