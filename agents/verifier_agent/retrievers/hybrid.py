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


def _rank_signal(rank: int) -> float:
    """Monotonic rank signal used instead of mixing incompatible raw scores."""
    return 1.0 / (60.0 + rank)


def _normalize_rank_fusion(
    rank_scores: Dict[str, float],
    max_rank_sources: int,
) -> Dict[str, float]:
    """Normalize RRF scores to 0..1 without unstable min/max scaling."""
    if not rank_scores:
        return {}
    maximum = max_rank_sources * _rank_signal(1)
    if maximum <= 0:
        return {key: 0.0 for key in rank_scores}
    return {
        key: max(0.0, min(1.0, value / maximum))
        for key, value in rank_scores.items()
    }


class HybridRetriever:
    """Production hybrid retriever using sparse+dense rank fusion plus lexical checks."""

    def __init__(self) -> None:
        self.sparse = BM25Retriever()
        self.dense = DenseRetriever()

    @staticmethod
    def _key(passage: Passage) -> str:
        # A provider/source_id is shared by many documents, so never use it
        # alone as identity. URL is preferred; otherwise use document content.
        if passage.url:
            return passage.url.strip().lower()
        return (
            f"{passage.source_id or passage.source}|"
            f"{passage.title}|{passage.snippet}"
        ).strip().lower()

    @staticmethod
    def _lexical_score(query: str, passage: Passage) -> float:
        """Measure query-term coverage in title + snippet."""
        query_tokens = set(_tokens(query))
        if not query_tokens:
            return 0.0

        text_tokens = set(_tokens(f"{passage.title or ''} {passage.snippet or ''}"))
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
        if not passages or k <= 0 or not query.strip():
            return []

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

            candidate_k = min(len(passages), max(k * 4, 12))
            sparse_results = self.sparse.retrieve(query, candidate_k)
            dense_results = self.dense.retrieve(query, candidate_k)

            # BM25 and cosine similarity live on different scales. Fuse their
            # ranks instead of their raw scores; this remains stable across models.
            rank_scores: Dict[str, float] = {}
            passage_map: Dict[str, Passage] = {}
            backend_count = 0

            for results in (sparse_results, dense_results):
                if results:
                    backend_count += 1
                for rank, (passage, _) in enumerate(results, start=1):
                    key = self._key(passage)
                    passage_map[key] = passage
                    rank_scores[key] = rank_scores.get(key, 0.0) + _rank_signal(rank)

            # Keep all adapter candidates available if one ranking backend fails.
            for passage in passages:
                passage_map.setdefault(self._key(passage), passage)

            rrf_scores = _normalize_rank_fusion(rank_scores, max(backend_count, 1))

            scored: List[Tuple[Passage, float]] = []
            for key, passage in passage_map.items():
                lexical = self._lexical_score(query, passage)
                rrf = rrf_scores.get(key, 0.0)

                # Lexical coverage protects precision; RRF protects semantic recall.
                if backend_count == 2:
                    fused = 0.60 * rrf + 0.25 * lexical
                elif backend_count == 1:
                    fused = 0.55 * rrf + 0.35 * lexical
                else:
                    fused = 0.85 * lexical

                adapter_signal = max(
                    0.0,
                    min(1.0, float(getattr(passage, "relevance_score", 0.0))),
                )
                if adapter_signal > 0:
                    fused = 0.90 * fused + 0.10 * adapter_signal

                # Broad adapter hits with no semantic or lexical signal should
                # not outrank meaningful evidence merely because they exist.
                if lexical == 0.0 and rrf == 0.0:
                    fused *= 0.15

                scored.append((passage, max(0.0, min(1.0, fused))))

            scored.sort(key=lambda item: item[1], reverse=True)

            final: List[Passage] = []
            for passage, score in scored:
                if any(
                    self._compute_overlap(passage.snippet, selected.snippet) >= 0.92
                    for selected in final
                ):
                    continue

                final.append(
                    passage.model_copy(update={"relevance_score": round(score, 6)})
                )
                if len(final) >= k:
                    break

            return final

        except Exception:
            # Deterministic fail-soft fallback when BM25/FAISS/embeddings fail.
            fallback = sorted(
                passages,
                key=lambda p: self._lexical_score(query, p),
                reverse=True,
            )
            return [
                p.model_copy(
                    update={"relevance_score": round(self._lexical_score(query, p), 6)}
                )
                for p in fallback[:k]
            ]

    @staticmethod
    def _compute_overlap(text1: str, text2: str) -> float:
        set1 = set(_tokens(text1))
        set2 = set(_tokens(text2))
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))
