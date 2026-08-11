from __future__ import annotations

import re
from typing import Dict, List, Tuple

from schemas.models import Passage
from .sparse import BM25Retriever
from .dense import DenseRetriever


_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "of", "to",
    "in", "on", "for", "with", "by", "from", "as", "at", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those", "it",
    "its", "into", "about", "after", "before", "during", "over", "under",
}


def _tokens(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if token not in _STOP_WORDS
    ]


def _normalize_scores(items: List[Tuple[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    values = [float(score) for _, score in items]
    lo, hi = min(values), max(values)
    if hi - lo <= 1e-9:
        value = 1.0 if hi > 0 else 0.0
        return {key: value for key, _ in items}
    return {
        key: max(0.0, min(1.0, (float(score) - lo) / (hi - lo)))
        for key, score in items
    }


class HybridRetriever:
    """Robust sparse+dense retrieval with lexical-aware score fusion."""

    def __init__(self) -> None:
        self.sparse = BM25Retriever()
        self.dense = DenseRetriever()

    @staticmethod
    def _key(passage: Passage) -> str:
        # source_id is often the provider name (e.g. "pubmed") and is shared
        # by many documents, so it must never be used as the sole identity key.
        if passage.url:
            return passage.url
        return (
            f"{passage.source_id or passage.source}|"
            f"{passage.title}|{passage.snippet}"
        )

    @staticmethod
    def _lexical_score(query: str, passage: Passage) -> float:
        query_tokens = set(_tokens(query))
        if not query_tokens:
            return 0.0

        text_tokens = set(
            _tokens(
                " ".join(
                    [
                        passage.title or "",
                        passage.snippet or "",
                    ]
                )
            )
        )
        if not text_tokens:
            return 0.0

        overlap = len(query_tokens & text_tokens) / len(query_tokens)
        phrase = 1.0 if query.strip().lower() in (passage.snippet or "").lower() else 0.0
        return min(1.0, 0.85 * overlap + 0.15 * phrase)

    def retrieve(
        self,
        query: str,
        passages: List[Passage],
        k: int = 5,
        dense_model: str | None = None,
    ) -> List[Passage]:
        if not passages or k <= 0:
            return []

        # Remove exact duplicates without collapsing distinct documents from
        # the same provider.
        unique: Dict[str, Passage] = {}
        for passage in passages:
            key = self._key(passage)
            if key not in unique:
                unique[key] = passage
        passages = list(unique.values())

        try:
            if dense_model and dense_model != self.dense.model_name:
                self.dense = DenseRetriever(model_name=dense_model)

            self.sparse.build_index(passages)
            self.dense.build_index(passages)

            candidate_k = min(len(passages), max(k * 3, 10))
            sparse_results = self.sparse.retrieve(query, candidate_k)
            dense_results = self.dense.retrieve(query, candidate_k)

            sparse_scores = _normalize_scores(
                [(self._key(p), score) for p, score in sparse_results]
            )
            dense_scores = _normalize_scores(
                [(self._key(p), score) for p, score in dense_results]
            )

            rank_scores: Dict[str, float] = {}
            passage_map: Dict[str, Passage] = {}

            for rank, (passage, _) in enumerate(sparse_results, start=1):
                key = self._key(passage)
                passage_map[key] = passage
                rank_scores[key] = rank_scores.get(key, 0.0) + 1.0 / (60 + rank)

            for rank, (passage, _) in enumerate(dense_results, start=1):
                key = self._key(passage)
                passage_map[key] = passage
                rank_scores[key] = rank_scores.get(key, 0.0) + 1.0 / (60 + rank)

            # Keep the complete adapter candidate pool available when one
            # ranking backend fails or returns fewer candidates.
            for passage in passages:
                passage_map.setdefault(self._key(passage), passage)

            rank_norm = _normalize_scores(list(rank_scores.items()))
            dense_available = bool(dense_scores)

            scored: List[Tuple[Passage, float]] = []
            for key, passage in passage_map.items():
                lexical = self._lexical_score(query, passage)
                sparse = sparse_scores.get(key, 0.0)
                dense = dense_scores.get(key, 0.0)
                rrf = rank_norm.get(key, 0.0)

                if dense_available:
                    fused = (
                        0.45 * dense
                        + 0.30 * sparse
                        + 0.20 * lexical
                        + 0.05 * rrf
                    )
                else:
                    fused = 0.65 * sparse + 0.25 * lexical + 0.10 * rrf

                adapter_signal = max(
                    0.0, min(1.0, float(getattr(passage, "relevance_score", 0.0)))
                )
                if adapter_signal > 0:
                    fused = 0.90 * fused + 0.10 * adapter_signal

                scored.append((passage, max(0.0, min(1.0, fused))))

            scored.sort(key=lambda item: item[1], reverse=True)

            final: List[Passage] = []
            for passage, score in scored:
                duplicate = False
                for selected in final:
                    if self._compute_overlap(passage.snippet, selected.snippet) >= 0.92:
                        duplicate = True
                        break
                if duplicate:
                    continue

                final.append(
                    passage.model_copy(update={"relevance_score": round(score, 6)})
                )
                if len(final) >= k:
                    break

            return final if final else passages[:k]

        except Exception:
            # Retrieval is fail-soft: lexical ranking still gives NLI usable
            # evidence when BM25, FAISS, or an embedding model is unavailable.
            fallback = sorted(
                passages,
                key=lambda p: self._lexical_score(query, p),
                reverse=True,
            )
            return [
                p.model_copy(
                    update={
                        "relevance_score": round(self._lexical_score(query, p), 6)
                    }
                )
                for p in fallback[:k]
            ]
    def _compute_overlap(self, text1: str, text2: str) -> float:
        set1 = set(_tokens(text1))
        set2 = set(_tokens(text2))
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))
