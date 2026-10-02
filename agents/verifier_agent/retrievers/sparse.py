from __future__ import annotations

import re
from typing import List, Tuple

from schemas.models import Passage


_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "of", "to",
    "in", "on", "for", "with", "by", "from", "as", "at", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those", "it",
    "its", "into", "about", "after", "before", "during", "over", "under",
}


def _tokenize(text: str) -> List[str]:
    """Stable BM25 tokenization that removes punctuation and common stop words."""
    return [
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if token not in _STOP_WORDS
    ]


class BM25Retriever:
    """Sparse retriever using BM25 with deterministic tokenization."""

    def __init__(self) -> None:
        self.bm25 = None
        self.passages: List[Passage] = []
        self._diagnostics = {}

    def diagnostics(self) -> dict:
        return {**self._diagnostics, "model": "rank_bm25.BM25Okapi",
                "model_available": self.bm25 is not None}

    def build_index(self, passages: List[Passage]) -> None:
        self.bm25 = None
        self._diagnostics = {"indexing_attempted": True, "indexing_executed": False,
                             "inference_attempted": False, "inference_executed": False,
                             "failure_stage": None, "error_type": None}
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            import logging
            logging.warning("rank_bm25 not installed. BM25Retriever will not function properly.")
            self.bm25 = None
            self.passages = list(passages)
            self._diagnostics.update(failure_stage="initialization", error_type="ImportError")
            return

        self.passages = list(passages)
        tokenized_corpus = [_tokenize(p.snippet) for p in self.passages]
        try:
            self.bm25 = BM25Okapi(tokenized_corpus)
        except Exception as exc:
            self._diagnostics.update(failure_stage="indexing", error_type=type(exc).__name__)
            raise
        self._diagnostics["indexing_executed"] = True

    def retrieve(self, query: str, k: int) -> List[Tuple[Passage, float]]:
        if not self.bm25 or not self.passages or k <= 0:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        self._diagnostics["inference_attempted"] = True
        try:
            scores = self.bm25.get_scores(tokenized_query)
        except Exception as exc:
            self._diagnostics.update(failure_stage="inference", error_type=type(exc).__name__)
            raise
        scored_passages = [
            (self.passages[i], float(score))
            for i, score in enumerate(scores)
        ]
        scored_passages.sort(key=lambda x: x[1], reverse=True)
        self._diagnostics["inference_executed"] = True
        return scored_passages[:k]
