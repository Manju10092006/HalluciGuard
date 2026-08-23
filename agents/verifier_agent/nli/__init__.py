"""Natural Language Inference engine.

Canonical pipeline engine: ``nli.robust_entailment.NLIEngine`` — the production-safe
implementation (confidence-margin decision policy + batch-alignment guard) that the
verification pipeline imports via ``from nli import NLIEngine``. It exposes the §15/§26
``diagnostics()`` execution-proof contract that ``api/pipeline.py`` records into a
``ModelExecutionTrace`` and that certification mode enforces (§28).

``nli.entailment.NLIEngine`` is retained as a separately-tested twin (imported directly
by its own regression tests) and must NOT be wired into the pipeline: keeping a single
canonical engine here is what prevents the diagnostics import/implementation mismatch.
"""
from .robust_entailment import NLIEngine

__all__ = ["NLIEngine"]
