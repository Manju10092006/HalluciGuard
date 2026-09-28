"""Claim-level evidence selection that reuses the Verifier's production stack.

This module does NOT re-implement claim decomposition, BM25, dense/FAISS, or
cross-encoder reranking. It lazily imports the canonical shared implementations
from ``agents.verifier_agent`` (the same components the ``VerificationPipeline``
uses) so the Detector and the Verifier both load each model at most once through
the shared ``ModelManager``. When the shared stack is unavailable (offline,
missing dependencies, or a model download being disallowed), every operation
degrades to the Detector's own deterministic lexical selection so the
reference-grounded classifier still runs.

Imports are intentionally lazy: the pre-verification triage path in
``DetectorAgent.detect`` must never trigger heavy model imports.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, List, Optional, Sequence

from .text import lexical_evidence

logger = logging.getLogger(__name__)

_VERIFIER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "agents", "verifier_agent")
)

# Tunables. These are the DeBERTa claim-verification default budgets; the
# Verifier's HybridRetriever keeps its own internal boxing, these control how
# many candidates are requested from it before reranking.
DEFAULT_CANDIDATE_K = 10
DEFAULT_POOL_K = 20
DEFAULT_RERANK_TOP = 3

_loaded = False
_decomposer: Any = None
_hybrid_retriever: Any = None
_reranker: Any = None
_passage_cls: Any = None


def _ensure_verifier() -> bool:
    """Insert the verifier package on ``sys.path`` and import shared components.

    Returns True when the shared stack is usable. Mirrors the loader used by
    ``services/llm_detector_verifier_service.py``.
    """
    global _loaded, _decomposer, _hybrid_retriever, _reranker, _passage_cls
    if _loaded:
        return _hybrid_retriever is not None

    if _VERIFIER_DIR not in sys.path:
        sys.path.insert(0, _VERIFIER_DIR)
    try:
        from claims import ClaimDecomposer  # type: ignore  # noqa: E402
        from retrievers import HybridRetriever  # type: ignore  # noqa: E402
        from rerankers import CrossEncoderReranker  # type: ignore  # noqa: E402
        from schemas.models import Passage  # type: ignore  # noqa: E402

        _decomposer = ClaimDecomposer()
        _hybrid_retriever = HybridRetriever()
        _reranker = CrossEncoderReranker()
        _passage_cls = Passage
        _loaded = True
        logger.debug("Claim evidence selector wired to shared Verifier stack.")
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning(
            "Shared Verifier stack unavailable for evidence selection (%s); "
            "falling back to deterministic lexical selection.",
            exc,
        )
        _loaded = True
        _decomposer = None
        _hybrid_retriever = None
        _reranker = None
        return False


def is_available() -> bool:
    return _ensure_verifier()


def decompose_claims(text: str) -> List[str]:
    """Atomic-claim decomposition reusing the Verifier's ClaimDecomposer.

    Returns an empty list when the shared stack is unavailable or the text
    carries no decomposable factual content. The caller falls back to
    sentence-level spans.
    """
    if not text or not text.strip():
        return []
    if not _ensure_verifier() or _decomposer is None:
        return []
    try:
        return _decomposer.decompose(text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ClaimDecomposer failed (%s); falling back to sentence spans.", exc)
        return []


def is_checkable(text: str) -> bool:
    """Non-factual / opinion detection reusing the shared decomposer's filter.

    Returns True for factual content worth evidence verification. Without an
    LLM and without loading spaCy, the shared regex/parser filter decides.
    """
    if not _ensure_verifier() or _decomposer is None:
        return True
    try:
        return _decomposer._is_checkable(text)  # noqa: SLF001 - shared helper
    except Exception:  # pragma: no cover - defensive
        return True


def _to_passage(text: str) -> Any:
    return _passage_cls(
        title="",
        source="evidence",
        url="",
        publication_date="",
        snippet=text,
        source_id="evidence",
    )


def select_evidence(
    claim: str,
    evidence_texts: Sequence[str],
    *,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    pool_k: int = DEFAULT_POOL_K,
    rerank_top: int = DEFAULT_RERANK_TOP,
    dense_model: Optional[str] = None,
) -> List[str]:
    """Select and rank evidence snippets for an atomic claim.

    Uses shared HybridRetriever (BM25 + dense/FAISS rank fusion) then reranks
    the merged pool with the shared cross-encoder. The reranker always receives
    the REAL claim (never a rewritten retrieval query), preserving the fix from
    PR #49. Falls back to deterministic lexical selection when the shared stack
    is unavailable or fails.
    """
    if not claim.strip() or not evidence_texts:
        return []
    texts = [t for t in evidence_texts if t and t.strip()]
    if not texts:
        return []

    if not _ensure_verifier() or _hybrid_retriever is None or _reranker is None:
        return lexical_evidence(claim, texts, limit=max(1, rerank_top))

    try:
        passages = [_to_passage(t) for t in texts]
        merged = _hybrid_retriever.retrieve(
            claim,
            passages,
            k=pool_k,
            dense_model=dense_model,
        )
        ranked = _reranker.rerank(claim, merged[: pool_k + 1], k=rerank_top)
        snippets = [p.snippet for p in ranked if p.snippet and p.snippet.strip()]
        if snippets:
            return snippets
        # Reranker produced no usable snippet (degraded/unavailable): keep the
        # hybrid order as a real fallback rather than silently dropping evidence.
        return [p.snippet for p in merged[:rerank_top] if p.snippet]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Shared evidence selection failed for claim (%s); using lexical fallback.",
            exc,
        )
        return lexical_evidence(claim, texts, limit=max(1, rerank_top))


class ClaimEvidenceEngine:
    """Stateless facade the Detector uses for claim-level evidence verification.

    Kept as a thin object so tests can inject stubs and so the detector has one
    seam to the shared stack. All heavy objects live behind module-level lazy
    singletons; constructing/first use never re-imports or reloads models.
    """

    def __init__(
        self,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        pool_k: int = DEFAULT_POOL_K,
        rerank_top: int = DEFAULT_RERANK_TOP,
        dense_model: Optional[str] = None,
    ) -> None:
        self.candidate_k = candidate_k
        self.pool_k = pool_k
        self.rerank_top = rerank_top
        self.dense_model = dense_model

    def available(self) -> bool:
        return is_available()

    def decompose(self, text: str) -> List[str]:
        return decompose_claims(text)

    def checkable(self, text: str) -> bool:
        return is_checkable(text)

    def select(self, claim: str, evidence_texts: Sequence[str]) -> List[str]:
        return select_evidence(
            claim,
            evidence_texts,
            candidate_k=self.candidate_k,
            pool_k=self.pool_k,
            rerank_top=self.rerank_top,
            dense_model=self.dense_model,
        )