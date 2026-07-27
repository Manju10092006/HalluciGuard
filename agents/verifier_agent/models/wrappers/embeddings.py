import logging
from typing import List
import numpy as np
from .base import BaseModelWrapper
from models.model_manager import get_model_manager

logger = logging.getLogger(__name__)

class EmbeddingModelWrapper(BaseModelWrapper):
    """
    Wrapper for SentenceTransformer embedding models.
    Supports:
    - BAAI/bge-large-en-v1.5
    - intfloat/e5-large-v2
    - sentence-transformers/all-mpnet-base-v2
    - sentence-transformers/all-MiniLM-L6-v2
    - BAAI/bge-m3 (default)
    """

    def __init__(self, model_id: str = "BAAI/bge-m3"):
        super().__init__(model_id)
        self._dim = 1024 if "large" in model_id or "m3" in model_id else 768
        if "MiniLM" in model_id:
            self._dim = 384

    def _load_internal(self) -> None:
        self._model = get_model_manager().load_embedding_model(self.model_id)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate normalized embeddings for a list of texts."""
        self.ensure_loaded()
        if not self._is_available or self._model is None:
            logger.warning(f"Model {self.model_id} not available, using fallback zero embeddings.")
            return np.zeros((len(texts), self._dim), dtype=np.float32)
        try:
            return self._model.encode(texts, normalize_embeddings=True)
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            return np.zeros((len(texts), self._dim), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a single query."""
        res = self.embed([query])
        return res[0]

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        emb1 = self.embed_query(text1)
        emb2 = self.embed_query(text2)
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
