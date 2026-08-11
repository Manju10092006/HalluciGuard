from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..schemas.models import VectorEntry, VectorSearchResult

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-backed vector store with sentence-transformer embeddings."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        store_path: str = "data/vector_store",
        dimension: int = 384,
        top_k: int = 10,
    ):
        self._model_name = embedding_model
        self._store_path = Path(store_path)
        self._dimension = dimension
        self._top_k = top_k
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._entries: list[VectorEntry] = []
        self._id_map: dict[str, int] = {}

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _ensure_index(self) -> faiss.IndexFlatIP:
        if self._index is None:
            self._index = faiss.IndexFlatIP(self._dimension)
        return self._index

    def initialize(self) -> None:
        self._ensure_model()
        self._ensure_index()
        self._store_path.mkdir(parents=True, exist_ok=True)

        entries_file = self._store_path / "entries.json"
        index_file = self._store_path / "index.faiss"

        if entries_file.exists():
            self._load_entries(entries_file)

        if index_file.exists() and self._entries:
            self._index = faiss.read_index(str(index_file))
            logger.info("Loaded vector store: %d entries", len(self._entries))
        else:
            logger.info("Initialized empty vector store")

    def _load_entries(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._entries = [VectorEntry(**e) for e in data]
            self._id_map = {e.entry_id: i for i, e in enumerate(self._entries)}
        except Exception:
            logger.exception("Failed to load vector entries")

    def _save(self) -> None:
        self._store_path.mkdir(parents=True, exist_ok=True)
        entries_file = self._store_path / "entries.json"
        index_file = self._store_path / "index.faiss"

        serializable = [
            {**e.model_dump(), "embedding": None} for e in self._entries
        ]
        entries_file.write_text(
            json.dumps(serializable, default=str, indent=2), encoding="utf-8"
        )

        if self._index and self._index.ntotal > 0:
            faiss.write_index(self._index, str(index_file))

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    def add(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        entry_id: Optional[str] = None,
    ) -> str:
        eid = entry_id or str(uuid.uuid4())
        index = self._ensure_index()

        embedding = self._embed([text])
        index.add(embedding)

        entry = VectorEntry(
            entry_id=eid,
            text=text,
            metadata=metadata or {},
            embedding=embedding[0].tolist(),
        )
        self._entries.append(entry)
        self._id_map[eid] = len(self._entries) - 1

        if len(self._entries) % 100 == 0:
            self._save()

        return eid

    def add_batch(
        self,
        texts: list[str],
        metadata_list: Optional[list[dict[str, Any]]] = None,
    ) -> list[str]:
        if not texts:
            return []

        index = self._ensure_index()
        embeddings = self._embed(texts)
        index.add(embeddings)

        ids = []
        for i, text in enumerate(texts):
            eid = str(uuid.uuid4())
            meta = metadata_list[i] if metadata_list else {}
            entry = VectorEntry(
                entry_id=eid,
                text=text,
                metadata=meta,
                embedding=embeddings[i].tolist(),
            )
            self._entries.append(entry)
            self._id_map[eid] = len(self._entries) - 1
            ids.append(eid)

        self._save()
        return ids

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResult]:
        index = self._ensure_index()
        if index.ntotal == 0:
            return []

        k = min(top_k or self._top_k, index.ntotal)
        query_embedding = self._embed([query])
        scores, indices = index.search(query_embedding, k)

        results: list[VectorSearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._entries):
                continue
            entry = self._entries[idx]

            if metadata_filter:
                match = all(
                    entry.metadata.get(k) == v for k, v in metadata_filter.items()
                )
                if not match:
                    continue

            results.append(VectorSearchResult(
                entry_id=entry.entry_id,
                text=entry.text,
                score=float(max(0.0, min(1.0, score))),
                metadata=entry.metadata,
            ))

        return results

    def get(self, entry_id: str) -> Optional[VectorEntry]:
        idx = self._id_map.get(entry_id)
        if idx is not None and idx < len(self._entries):
            return self._entries[idx]
        return None

    def delete(self, entry_id: str) -> bool:
        idx = self._id_map.get(entry_id)
        if idx is None:
            return False
        self._entries.pop(idx)
        self._id_map.pop(entry_id, None)
        self._id_map = {
            e.entry_id: i for i, e in enumerate(self._entries)
        }
        self._rebuild_index()
        return True

    def _rebuild_index(self) -> None:
        self._index = faiss.IndexFlatIP(self._dimension)
        if self._entries:
            embeddings = np.array(
                [e.embedding for e in self._entries if e.embedding],
                dtype=np.float32,
            )
            if embeddings.shape[0] > 0:
                self._index.add(embeddings)

    def size(self) -> int:
        return len(self._entries)

    def save(self) -> None:
        self._save()

    def clear(self) -> None:
        self._entries.clear()
        self._id_map.clear()
        self._index = faiss.IndexFlatIP(self._dimension)


_vector_store_instance: Optional[VectorStore] = None


def get_vector_store(
    embedding_model: str = "all-MiniLM-L6-v2",
    store_path: str = "data/vector_store",
    dimension: int = 384,
    top_k: int = 10,
) -> VectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore(
            embedding_model=embedding_model,
            store_path=store_path,
            dimension=dimension,
            top_k=top_k,
        )
    return _vector_store_instance
