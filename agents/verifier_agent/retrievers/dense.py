from __future__ import annotations
import logging
from typing import List, Tuple, Optional
import numpy as np

from schemas.models import Passage
from models.model_manager import get_model_manager

class DenseRetriever:
    """Dense retriever using sentence-transformers and FAISS."""

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> None:
        self.model_name = model_name
        self.model = None
        self.index = None
        self.passages: List[Passage] = []
        self._is_available = True
        
    def _load_model(self) -> None:
        if self.model is not None or not self._is_available:
            return
            
        try:
            self.model = get_model_manager().load_embedding_model()
        except ImportError:
            logging.warning("sentence_transformers not installed. DenseRetriever falling back.")
            self._is_available = False

    def build_index(self, passages: List[Passage]) -> None:
        """
        Encode snippets and build FAISS IndexFlatIP.
        
        Args:
            passages: List of Passage objects to index.
        """
        self._load_model()
        self.passages = passages
        
        if not self._is_available or not self.passages:
            return
            
        try:
            import faiss
            
            texts = [p.snippet for p in passages]
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
            
        except ImportError:
            logging.warning("faiss not installed. DenseRetriever falling back.")
            self._is_available = False

    def retrieve(self, query: str, k: int) -> List[Tuple[Passage, float]]:
        """
        Encode query and search FAISS for top k passages.
        
        Args:
            query: The search query.
            k: Number of passages to retrieve.
            
        Returns:
            List of tuples containing (Passage, score).
        """
        self._load_model()
        if not self._is_available or self.index is None or not self.passages:
            return []
            
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        scores, indices = self.index.search(query_embedding, min(k, len(self.passages)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append((self.passages[idx], float(scores[0][i])))
                
        return results
