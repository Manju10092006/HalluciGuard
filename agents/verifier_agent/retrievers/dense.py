from __future__ import annotations
import logging
from typing import List, Tuple

from schemas.models import Passage
from models.model_manager import get_model_manager

class DenseRetriever:
    """Dense retriever using sentence-transformers and FAISS."""

    def __init__(self, model_name: str = 'BAAI/bge-m3') -> None:
        self.model_name = model_name
        self.model = None
        self.index = None
        self.passages: List[Passage] = []
        self._is_available = True
        self._index_built = False
        self._unavailable_stage = None
        self._unavailable_error_type = None
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        self._last = {
            "component": "dense_retriever", "model": self.model_name,
            "status": "not_run", "degraded": False,
            "load_attempted": False, "index_attempted": False,
            "passage_encoding_attempted": False, "passage_encoding_executed": False,
            "query_encoding_attempted": False, "query_encoding_executed": False,
            "inference_attempted": False, "inference_executed": False,
            "retrieval_attempted": False, "search_attempted": False,
            "search_executed": False, "executed": False, "returned_count": 0,
            "faiss_available": None, "failure_stage": None,
            "error_type": None, "fallback_reason": None,
        }

    def _failure(self, stage: str, error_type: str | None, *, unavailable=False) -> None:
        self._last.update(
            status="unavailable" if unavailable else "failed", degraded=True,
            failure_stage=stage, error_type=error_type,
            fallback_reason=f"{stage}:{error_type or 'unavailable'}",
        )
        if unavailable:
            self._unavailable_stage = stage
            self._unavailable_error_type = error_type

    def diagnostics(self) -> dict:
        """Last build/search proof; returned tuples alone do not prove FAISS ran."""
        return {
            **self._last,
            "loaded": self.model is not None,
            "available": bool(self._is_available and self.model is not None
                              and self._last["faiss_available"]),
            "index_built": self._index_built and self.index is not None,
        }
        
    def _load_model(self) -> None:
        if self.model is not None or not self._is_available:
            if not self._is_available:
                self._failure(self._unavailable_stage or "model_load",
                              self._unavailable_error_type, unavailable=True)
                if self._unavailable_stage == "faiss_import":
                    self._last["faiss_available"] = False
            return
            
        try:
            self._last["load_attempted"] = True
            self.model = get_model_manager().load_embedding_model(self.model_name)
        except ImportError as exc:
            logging.warning("sentence_transformers not installed. DenseRetriever falling back.")
            self._is_available = False
            self._failure("model_load", type(exc).__name__, unavailable=True)
        except Exception as e:
            # The most common cause of the cryptic "'NoneType' object has no attribute
            # 'endswith'" here is that the embedding model is not present in the local
            # HuggingFace cache while ALLOW_MODEL_DOWNLOADS is false: sentence-transformers'
            # snapshot resolution returns None for the model path, then calls .endswith on it.
            # This is a RECALL degradation (dense/semantic retrieval is skipped; sparse BM25
            # + lexical + cross-encoder reranking still run), not a correctness corruption.
            is_none_path = isinstance(e, AttributeError) and "endswith" in str(e)
            if is_none_path:
                logging.warning(
                    "Dense model '%s' could not be resolved locally (likely not cached and "
                    "ALLOW_MODEL_DOWNLOADS is false). Falling back to sparse+rerank retrieval "
                    "(reduced semantic recall). Set ALLOW_MODEL_DOWNLOADS=true or pre-cache the "
                    "model to restore dense retrieval. Underlying error: %s",
                    self.model_name,
                    e,
                )
            else:
                logging.warning(f"Error loading dense model {self.model_name}: {e}. Falling back.")
            self._is_available = False
            self._failure("model_load", type(e).__name__, unavailable=True)

    def build_index(self, passages: List[Passage]) -> None:
        """
        Encode snippets and build FAISS IndexFlatIP.
        
        Args:
            passages: List of Passage objects to index.
        """
        self._reset_diagnostics()
        # Never expose a previous corpus's index as proof for this build.
        self.index = None
        self._index_built = False
        self._last["index_attempted"] = True
        self._load_model()
        self.passages = passages
        
        if not self._is_available or not self.passages:
            return
            
        try:
            import faiss
            self._last["faiss_available"] = True
        except ImportError as exc:
            logging.warning("FAISS not available. Dense retrieval disabled for this request.")
            self._is_available = False
            self._last["faiss_available"] = False
            self._failure("faiss_import", type(exc).__name__, unavailable=True)
            return
        except Exception as exc:
            self._failure("faiss_import", type(exc).__name__)
            raise

        stage = "passage_encoding"
        try:
            texts = [p.snippet for p in passages]
            self._last["passage_encoding_attempted"] = True
            self._last["inference_attempted"] = True
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            self._last["passage_encoding_executed"] = True
            self._last["inference_executed"] = True
            stage = "index_build"
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
            self._index_built = True
            self._last["status"] = "ready"
            logging.info("Built FAISS index with %d passages, dim=%d", len(passages), dim)
            
        except ImportError as exc:
            # Preserve the existing fail-soft ImportError contract, but identify
            # the operation that failed rather than blaming every error on FAISS.
            self._is_available = False
            self._failure(stage, type(exc).__name__, unavailable=True)
        except Exception as exc:
            self.index = None
            self._index_built = False
            self._failure(stage, type(exc).__name__)
            raise

    def retrieve(self, query: str, k: int) -> List[Tuple[Passage, float]]:
        """
        Encode query and search FAISS for top k passages.
        
        Args:
            query: The search query.
            k: Number of passages to retrieve.
            
        Returns:
            List of tuples containing (Passage, score).
        """
        self._last.update(
            retrieval_attempted=True, query_encoding_attempted=False,
            query_encoding_executed=False, search_attempted=False,
            search_executed=False, executed=False, returned_count=0,
        )
        if self._index_built:
            self._last.update(status="not_run", degraded=False, failure_stage=None,
                              error_type=None, fallback_reason=None,
                              inference_attempted=False, inference_executed=False)
        self._load_model()
        if not self._is_available or self.index is None or not self.passages:
            if self.index is None:
                logging.debug("Dense retrieval skipped: no index available")
            return []
            
        stage = "query_encoding"
        try:
            self._last["query_encoding_attempted"] = True
            self._last["inference_attempted"] = True
            query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            self._last["query_encoding_executed"] = True
            self._last["inference_executed"] = True

            stage = "search"
            self._last["search_attempted"] = True
            scores, indices = self.index.search(query_embedding, min(k, len(self.passages)))
            self._last["search_executed"] = True
            self._last["executed"] = True

            stage = "result_decode"
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:
                    results.append((self.passages[idx], float(scores[0][i])))
            self._last.update(status="executed", returned_count=len(results))
            return results
        except Exception as exc:
            self._failure(stage, type(exc).__name__)
            raise
