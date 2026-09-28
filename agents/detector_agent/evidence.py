"""Hybrid evidence retrieval for the Detector Agent.

Reuses the Verifier Agent's proven retrieval stack — BM25, FAISS dense
retrieval, and the cross-encoder reranker — instead of re-implementing them:

    claim
      -> BM25 top-k          (lexical)
      -> dense top-k         (FAISS embeddings)
      -> merged pool         (deduplicated)
      -> rerank              (cross-encoder when available)
      -> final evidence top-k

Every returned evidence item carries provenance (``retrieval_method`` /
``retrieval_score``) so the Detector stays debuggable, but the internal
retriever wiring is NOT part of the public Detector contract.

Dense retrieval and reranking degrade softly: when the embedding/cross-encoder
models or ``faiss``/``rank_bm25`` are unavailable, retrieval falls back to
lexical-only rather than failing the whole detection run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from agents.verifier_agent.schemas.models import Passage


@dataclass
class EvidenceItem:
    """One piece of evidence presented to the NLI layer."""

    text: str
    retrieval_score: float = 0.0
    retrieval_method: str = "none"
    rerank_score: Optional[float] = None


@dataclass
class EvidenceResult:
    """Bundle returned by :meth:`EvidenceRetriever.retrieve_evidence`."""

    items: List[EvidenceItem] = field(default_factory=list)
    retrieved: bool = False
    degraded: bool = False
    methods_used: List[str] = field(default_factory=list)
    bm25_top_k: List[str] = field(default_factory=list)
    dense_top_k: List[str] = field(default_factory=list)


EMPTY_RESULT = EvidenceResult()


class EvidenceRetriever:
    """Hybrid BM25 + dense retriever with reranking, reusing Verifier infra.

    Args:
        bm25_k:        BM25 candidate count (task default: 10).
        dense_k:       Dense candidate count (task default: 10).
        pool_size:     Merged pool size after dedup (task default: 20).
        final_k:       Evidence count handed to the NLI layer (task default: 3).
        sparse:        Optional injected BM25 retriever (test seam / reuse).
        dense:         Optional injected dense retriever (test seam / reuse).
        reranker:      Optional injected cross-encoder reranker (reuse).
    """

    def __init__(
        self,
        bm25_k: int = 10,
        dense_k: int = 10,
        pool_size: int = 20,
        final_k: int = 3,
        sparse=None,
        dense=None,
        reranker=None,
    ) -> None:
        self.bm25_k = bm25_k
        self.dense_k = dense_k
        self.pool_size = pool_size
        self.final_k = final_k

        from agents.verifier_agent.retrievers.sparse import BM25Retriever
        from agents.verifier_agent.retrievers.dense import DenseRetriever
        from agents.verifier_agent.rerankers.cross_encoder import CrossEncoderReranker

        self.sparse = sparse or BM25Retriever()
        self.dense = dense or DenseRetriever()
        self.reranker = reranker or CrossEncoderReranker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve_evidence(
        self,
        claim: str,
        documents: Optional[Sequence[str]],
        *,
        final_k: Optional[int] = None,
    ) -> EvidenceResult:
        """Retrieve the best evidence for ``claim`` from ``documents``.

        Args:
            claim: An atomic factual claim.
            documents: Corpus of document snippets (strings). ``None`` or an
                empty sequence means "no evidence available" — callers must
                treat the claim as UNVERIFIED, not as supported or contradicted.
            final_k: Optional override of the final evidence count.

        Returns:
            EvidenceResult. ``retrieved`` is False when no documents were
            supplied or every backend degraded.
        """
        if not documents:
            return EMPTY_RESULT
        docs = [d for d in documents if d and d.strip()]
        if not docs:
            return EMPTY_RESULT

        passages = _build_passages(docs)
        k = max(1, min(final_k or self.final_k, len(passages)))

        try:
            return self._hybrid_retrieve(claim, passages, k)
        except Exception:
            # Deterministic fail-soft: lexical-only order rather than a crash.
            return self._lexical_fallback(claim, passages, k)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _hybrid_retrieve(self, claim: str, passages: List[Passage], k: int) -> EvidenceResult:
        self.sparse.build_index(passages)
        self.dense.build_index(passages)

        sparse_hits = self.sparse.retrieve(claim, self.bm25_k)
        dense_hits = self.dense.retrieve(claim, self.dense_k)

        methods: List[str] = []
        if sparse_hits:
            methods.append("bm25")
        if dense_hits:
            methods.append("dense")
        if not methods:
            return self._lexical_fallback(claim, passages, k)

        sparse_top = [p.snippet for p, _ in sparse_hits]
        dense_top = [p.snippet for p, _ in dense_hits]

        # Merge candidates, deduplicate by normalized snippet (a document can
        # legitimately surface in both backends), and keep the best source.
        merged: dict = {}
        for idx, (passage, score) in enumerate(sparse_hits):
            key = _normalize_snippet(passage.snippet)
            merged.setdefault(
                key,
                {"passage": passage, "sources": set(), "best_score": 0.0},
            )
            merged[key]["sources"].add("bm25")
            merged[key]["best_score"] = max(merged[key]["best_score"], _normalize_score(score))
        for idx, (passage, score) in enumerate(dense_hits):
            key = _normalize_snippet(passage.snippet)
            merged.setdefault(
                key,
                {"passage": passage, "sources": set(), "best_score": 0.0},
            )
            merged[key]["sources"].add("dense")
            merged[key]["best_score"] = max(merged[key]["best_score"], _normalize_score(score))

        entries = sorted(
            merged.values(),
            key=lambda item: item["best_score"],
            reverse=True,
        )[: self.pool_size]

        # Rerank the candidate pool with the cross-encoder when it can run;
        # otherwise keep the hybrid order (scores are heuristic, not calibrated).
        pool = [entry["passage"] for entry in entries]
        reranked = self.reranker.rerank(claim, pool, k)
        rerank_executed = bool(
            self.reranker.last_status == "executed" and self.reranker.last_inference_executed
        )

        chosen = pool[:k] if not rerank_executed else reranked
        method_str = "+".join(methods)
        if rerank_executed:
            method_str = f"{method_str}+reranked"

        # Map each chosen passage back to its originating backend(s) for
        # per-item provenance metadata (debugging, not public contract).
        merged_by_snippet = {entry["passage"].snippet: entry for entry in entries}

        items: List[EvidenceItem] = []
        for passage in chosen:
            entry = merged_by_snippet.get(passage.snippet)
            sources = sorted(entry["sources"]) if entry else methods
            score = float(passage.relevance_score)
            if not rerank_executed:
                score = entry["best_score"] if entry else score
            items.append(
                EvidenceItem(
                    text=passage.snippet,
                    retrieval_score=round(max(0.0, min(1.0, score)), 4),
                    retrieval_method="+".join(sources) + ("+reranked" if rerank_executed else ""),
                    rerank_score=round(max(0.0, min(1.0, float(passage.relevance_score))), 4)
                    if rerank_executed
                    else None,
                )
            )

        return EvidenceResult(
            items=items,
            retrieved=bool(items),
            degraded=not rerank_executed,
            methods_used=methods,
            bm25_top_k=sparse_top,
            dense_top_k=dense_top,
        )

    def _lexical_fallback(self, claim: str, passages: List[Passage], k: int) -> EvidenceResult:
        """Deterministic lexical-only fallback when hybrids fail."""
        scored = sorted(
            passages,
            key=lambda p: _lexical_overlap(claim, p.snippet),
            reverse=True,
        )
        items = [
            EvidenceItem(
                text=p.snippet,
                retrieval_score=round(_lexical_overlap(claim, p.snippet), 4),
                retrieval_method="bm25",
            )
            for p in scored[:k]
        ]
        return EvidenceResult(
            items=items,
            retrieved=bool(items),
            degraded=True,
            methods_used=["bm25"],
            bm25_top_k=[p.snippet for p in scored[:k]],
        )


def _build_passages(documents: Sequence[str]) -> List[Passage]:
    return [
        Passage(
            title="",  # flat corpus: snippet is the full document text
            source="local",
            url="",
            publication_date="",
            snippet=text.strip(),
            source_id=f"doc-{idx}",
        )
        for idx, text in enumerate(documents)
    ]


def _normalize_snippet(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _normalize_score(score: float) -> float:
    """Clamp arbitrary retriever scores to ``[0, 1]`` for ranking."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _lexical_overlap(claim: str, text: str) -> float:
    import re

    def tokens(value: str) -> set:
        stop = {
            "a", "an", "the", "and", "or", "but", "of", "to", "in", "on",
            "for", "is", "are", "was", "were", "it", "its", "by", "at",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", (value or "").lower())
            if token not in stop
        }

    claim_tokens = tokens(claim)
    text_tokens = tokens(text)
    if not claim_tokens or not text_tokens:
        return 0.0
    overlap = len(claim_tokens & text_tokens)
    return overlap / len(claim_tokens)


__all__ = ["EvidenceItem", "EvidenceResult", "EvidenceRetriever"]