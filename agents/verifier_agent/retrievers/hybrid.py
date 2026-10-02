from __future__ import annotations

import re
import time
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
        self.last_diagnostics: dict = {"route": "not_run", "degraded": False}

    def diagnostics(self) -> dict:
        return dict(self.last_diagnostics)

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
        self, query: str, passages: List[Passage], k: int = 5,
        dense_model: str | None = None,
    ) -> List[Passage]:
        started = time.perf_counter()
        try:
            return self._retrieve(query, passages, k, dense_model)
        finally:
            self.last_diagnostics["total_duration_ms"] = round((time.perf_counter() - started) * 1000, 3)

    def _retrieve(
        self,
        query: str,
        passages: List[Passage],
        k: int = 5,
        dense_model: str | None = None,
    ) -> List[Passage]:
        self.last_diagnostics = {
            "route": "not_run", "degraded": False,
            "sparse_attempted": False, "sparse_executed": False,
            "dense_attempted": False, "dense_executed": False,
            "dense_available": False, "dense_failure_stage": None,
            "sparse_available": False,
            "sparse_availability_checked": False, "dense_availability_checked": False,
            "sparse_contributed": False, "dense_contributed": False,
            "errors": [], "selected_count": 0,
            "input_count": len(passages), "deduplicated_count": 0,
            "sparse_result_count": 0, "dense_result_count": 0,
            "fusion_executed": False, "fusion_backend_count": 0,
            "fallback_reason": None,
            "sparse_requested": True, "dense_requested": True,
            "sparse_status": "not_run", "dense_status": "not_run",
            "fusion_attempted": False, "fusion_status": "not_run",
            "sparse_duration_ms": 0.0, "dense_duration_ms": 0.0,
            "fusion_duration_ms": 0.0,
            "sparse_model": "rank_bm25.BM25Okapi",
            "dense_model": dense_model or self.dense.model_name,
            "selected_contributors": [],
        }
        if not passages or k <= 0 or not query.strip():
            self.last_diagnostics["route"] = "empty_input"
            return []

        unique: Dict[str, Passage] = {}
        for passage in passages:
            key = self._key(passage)
            if key not in unique:
                unique[key] = passage
        passages = list(unique.values())
        self.last_diagnostics["deduplicated_count"] = len(passages)

        stage = "backend_setup"
        try:
            if dense_model and dense_model != self.dense.model_name:
                self.dense = DenseRetriever(model_name=dense_model)

            candidate_k = min(len(passages), max(k * 4, 12))
            sparse_results = []
            dense_results = []
            self.last_diagnostics["sparse_attempted"] = True
            sparse_errored = False
            sparse_started = time.perf_counter()
            try:
                self.sparse.build_index(passages)
                sparse_results = self.sparse.retrieve(query, candidate_k)
                self.last_diagnostics["sparse_executed"] = True
                self.last_diagnostics["sparse_status"] = "succeeded" if sparse_results else "empty"
            except Exception as exc:
                sparse_errored = True
                self.last_diagnostics["sparse_status"] = "timeout" if isinstance(exc, TimeoutError) else "failed"
                self.last_diagnostics["errors"].append({"component": "sparse", "error_type": type(exc).__name__})
            finally:
                self.last_diagnostics["sparse_duration_ms"] = round((time.perf_counter() - sparse_started) * 1000, 3)
            sparse_diag = self.sparse.diagnostics() if hasattr(self.sparse, "diagnostics") else {}
            self.last_diagnostics["sparse_availability_checked"] = bool(sparse_diag)
            self.last_diagnostics["sparse_available"] = bool(sparse_diag.get("model_available", sparse_results))
            if sparse_diag:
                self.last_diagnostics["sparse_details"] = sparse_diag
                self.last_diagnostics["sparse_executed"] = sparse_diag.get("inference_executed") is True
                if sparse_diag.get("error_type"):
                    if not sparse_errored:
                        self.last_diagnostics["sparse_status"] = "unavailable"
                        self.last_diagnostics["errors"].append({"component": "sparse",
                            "stage": sparse_diag.get("failure_stage"), "error_type": sparse_diag["error_type"]})
                    else:
                        for error in self.last_diagnostics["errors"]:
                            if error["component"] == "sparse":
                                error["stage"] = sparse_diag.get("failure_stage")
                elif self.last_diagnostics["sparse_status"] not in {"failed", "timeout"} and not self.last_diagnostics["sparse_executed"]:
                    self.last_diagnostics["sparse_status"] = "skipped"
            self.last_diagnostics["dense_attempted"] = True
            dense_errored = False
            dense_started = time.perf_counter()
            try:
                self.dense.build_index(passages)
                dense_results = self.dense.retrieve(query, candidate_k)
            except Exception as exc:
                dense_errored = True
                self.last_diagnostics["errors"].append({"component": "dense", "error_type": type(exc).__name__})
                self.last_diagnostics["dense_status"] = "timeout" if isinstance(exc, TimeoutError) else "failed"
            finally:
                self.last_diagnostics["dense_duration_ms"] = round((time.perf_counter() - dense_started) * 1000, 3)
            dense_diag = self.dense.diagnostics() if hasattr(self.dense, "diagnostics") else {}
            self.last_diagnostics["dense_availability_checked"] = bool(dense_diag)
            self.last_diagnostics["dense_available"] = bool(dense_diag.get("model_available", dense_results))
            self.last_diagnostics["dense_executed"] = bool(dense_diag.get("inference_executed", dense_results))
            self.last_diagnostics["dense_failure_stage"] = dense_diag.get("failure_stage")
            if not dense_errored:
                self.last_diagnostics["dense_status"] = (
                    "unavailable" if dense_diag.get("error_type") else
                    "succeeded" if dense_results else
                    "empty" if self.last_diagnostics["dense_executed"] else "skipped"
                )
            # Record the dense failure ONCE: the re-raising path above already logged
            # it, so only add the stage-enriched diagnostic entry for the non-raising
            # path (e.g. a sticky init failure where retrieve() returns []), avoiding
            # the double-count (audit #39).
            if dense_diag.get("error_type") and not dense_errored:
                self.last_diagnostics["errors"].append({"component": "dense", "stage": dense_diag.get("failure_stage"), "error_type": dense_diag["error_type"]})
            self.last_diagnostics["sparse_contributed"] = bool(sparse_results)
            self.last_diagnostics["dense_contributed"] = bool(dense_results)
            self.last_diagnostics["sparse_result_count"] = len(sparse_results)
            self.last_diagnostics["dense_result_count"] = len(dense_results)
            self.last_diagnostics["dense_details"] = dense_diag
            route = ("hybrid" if sparse_results and dense_results else
                     "bm25_only" if sparse_results else
                     "dense_only" if dense_results else "lexical_fallback")
            self.last_diagnostics["route"] = route
            if route != "hybrid":
                self.last_diagnostics["fallback_reason"] = (
                    "backend_failure" if self.last_diagnostics["errors"] else "backend_empty_results"
                )
            self.last_diagnostics["degraded"] = (route != "hybrid" or bool(self.last_diagnostics["errors"])
                                                  or not self.last_diagnostics["dense_executed"])

            # BM25 and cosine similarity live on different scales. Fuse their
            # ranks instead of their raw scores; this remains stable across models.
            rank_scores: Dict[str, float] = {}
            passage_map: Dict[str, Passage] = {}
            backend_count = 0
            fusion_started = time.perf_counter()
            stage = "fusion"
            self.last_diagnostics["fusion_attempted"] = bool(sparse_results or dense_results)

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
            self.last_diagnostics["fusion_executed"] = bool(rank_scores)
            self.last_diagnostics["fusion_backend_count"] = backend_count
            self.last_diagnostics["fusion_status"] = "succeeded" if rank_scores else "skipped"
            self.last_diagnostics["fusion_duration_ms"] = round((time.perf_counter() - fusion_started) * 1000, 3)

            stage = "scoring"
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

            stage = "selection"
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

            self.last_diagnostics["selected_count"] = len(final)
            sparse_keys = {self._key(p) for p, _ in sparse_results}
            dense_keys = {self._key(p) for p, _ in dense_results}
            self.last_diagnostics["selected_contributors"] = [
                {"source_id": p.source_id, "source": p.source, "url": p.url,
                 "backends": (["sparse"] if self._key(p) in sparse_keys else [])
                    + (["dense"] if self._key(p) in dense_keys else [])
                    + (["lexical"] if self._lexical_score(query, p) > 0 else [])}
                for p in final
            ]
            return final

        except Exception as exc:
            # Deterministic fail-soft fallback when BM25/FAISS/embeddings fail.
            self.last_diagnostics["route"] = "lexical_fallback"
            self.last_diagnostics["fallback_reason"] = f"{stage}_failure"
            if stage == "fusion":
                self.last_diagnostics["fusion_status"] = "failed"
            if stage == "fusion" and "fusion_started" in locals():
                self.last_diagnostics["fusion_duration_ms"] = round((time.perf_counter() - fusion_started) * 1000, 3)
            self.last_diagnostics["degraded"] = True
            self.last_diagnostics["errors"].append({"component": stage, "error_type": type(exc).__name__})
            fallback = sorted(
                passages,
                key=lambda p: self._lexical_score(query, p),
                reverse=True,
            )
            selected = [
                p.model_copy(
                    update={"relevance_score": round(self._lexical_score(query, p), 6)}
                )
                for p in fallback[:k]
            ]
            self.last_diagnostics["selected_count"] = len(selected)
            self.last_diagnostics["selected_contributors"] = [
                {"source_id": p.source_id, "source": p.source, "url": p.url, "backends": ["lexical"]}
                for p in selected
            ]
            return selected

    @staticmethod
    def _compute_overlap(text1: str, text2: str) -> float:
        set1 = set(_tokens(text1))
        set2 = set(_tokens(text2))
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))
