"""
HalluciGuard Verifier — Certification Mode (§28/§29/§30).

Certification mode is an **opt-in, fail-closed** verification mode. Normal
production remains resilient (fail-soft): a degraded model or thin evidence
still yields a best-effort verdict. Certification mode is the opposite — it
exists to *prove* the runtime chain executed for real, so any of the following
becomes a CONTROLLED FAILURE instead of being silently presented as a genuine,
model-backed result:

  * the detector fell back to its hardcoded baseline (real HaluEval never ran),
  * the BGE reranker fell back (hybrid scores passed through as if they were BGE),
  * the DeBERTa NLI fell back on evidence that was actually present,
  * evidence is mock / empty / malformed.

It is OFF by default (``CERTIFICATION_MODE`` env / ``Settings.certification_mode``)
so this module changes nothing for existing production callers.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

__all__ = [
    "CertificationError",
    "certification_enabled_from_env",
    "enforce_reranker",
    "enforce_nli",
    "enforce_retrieval_evidence",
    "enforce_detector",
]


class CertificationError(RuntimeError):
    """Raised in certification mode when a stage cannot be certified as real.

    Carries the offending ``stage`` and a human-readable ``reason`` so callers
    can surface a precise, non-fabricated failure (never a fake verdict).
    """

    def __init__(self, stage: str, reason: str) -> None:
        self.stage = stage
        self.reason = reason
        super().__init__(f"[certification:{stage}] {reason}")


_TRUE = {"1", "true", "yes", "on"}


def certification_enabled_from_env() -> bool:
    """Read CERTIFICATION_MODE from the environment (used where Settings isn't loaded)."""
    return str(os.environ.get("CERTIFICATION_MODE", "")).strip().lower() in _TRUE


def enforce_reranker(diag: Dict[str, Any], enabled: bool) -> None:
    """Fail if BGE was needed but fell back. ``not_run`` (no passages) is allowed
    here — the retrieval/evidence check owns the empty-evidence case."""
    if not enabled or not diag:
        return
    status = str(diag.get("status", "not_run"))
    if status in ("degraded", "unavailable"):
        raise CertificationError(
            "bge_reranker",
            f"BGE reranker did not run for real (status={status}); "
            "a fail-soft fallback score cannot be certified as a BGE relevance score.",
        )
    if status == "executed" and diag.get("degraded"):
        raise CertificationError(
            "bge_reranker",
            "BGE reranker reported executed but degraded; refusing to certify.",
        )


def enforce_nli(diag: Dict[str, Any], enabled: bool, evidence_present: bool) -> None:
    """Fail if DeBERTa NLI fell back while real evidence was present to classify."""
    if not enabled or not diag or not evidence_present:
        return
    status = str(diag.get("status", "not_run"))
    if status in ("degraded", "unavailable") or not diag.get("inference_executed"):
        raise CertificationError(
            "deberta_nli",
            f"DeBERTa NLI did not run for real on present evidence (status={status}); "
            "degraded NLI cannot be certified.",
        )


def _looks_mock(passage: Any) -> bool:
    """Conservative heuristic for mock / placeholder / malformed passages."""
    url = str(getattr(passage, "url", "") or "").lower()
    source = str(getattr(passage, "source", "") or "").lower()
    title = str(getattr(passage, "title", "") or "")
    snippet = str(getattr(passage, "snippet", "") or "")
    if any(tok in url for tok in ("example.com", "mock", "placeholder", "dummy")):
        return True
    if any(tok in source for tok in ("mock", "placeholder", "dummy", "synthetic")):
        return True
    # Malformed: no textual content at all.
    if not snippet.strip() and not title.strip():
        return True
    return False


def enforce_retrieval_evidence(
    raw_passages: List[Any], enabled: bool, mock_mode: bool
) -> None:
    """Fail if evidence is mock, empty, or malformed (§28 / §45 'verdict without evidence')."""
    if not enabled:
        return
    if mock_mode:
        raise CertificationError(
            "retrieval",
            "MOCK_MODE is enabled; mock evidence cannot be certified. Set MOCK_MODE=false.",
        )
    if not raw_passages:
        raise CertificationError(
            "retrieval",
            "No real evidence was retrieved for this claim; "
            "certification refuses to produce a verdict without evidence.",
        )
    if all(_looks_mock(p) for p in raw_passages):
        raise CertificationError(
            "retrieval",
            "All retrieved passages look mock/placeholder/malformed; cannot certify.",
        )


def enforce_detector(detector_result: Dict[str, Any], enabled: bool) -> None:
    """Fail if the detector fell back to baseline instead of running HaluEval."""
    if not enabled or not detector_result:
        return
    degraded = bool(detector_result.get("detector_degraded"))
    executed = bool(detector_result.get("detector_inference_executed"))
    if degraded or not executed:
        raise CertificationError(
            "detector",
            "Detector fell back to the baseline heuristic; real HaluEval inference did not run. "
            "Fix HALUEVAL_MODEL_PATH / model load before certifying.",
        )
