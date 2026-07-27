import logging
from typing import List, Tuple
from .base import BaseModelWrapper
from models.model_manager import get_model_manager

logger = logging.getLogger(__name__)

class RerankerWrapper(BaseModelWrapper):
    """
    Wrapper for cross-encoder reranking models.
    Supports: cross-encoder/ms-marco-MiniLM-L-6-v2, BAAI/bge-reranker-large (default)
    """

    def __init__(self, model_id: str = "BAAI/bge-reranker-large"):
        super().__init__(model_id)
        self._model = None

    def _load_internal(self) -> None:
        self._model = get_model_manager().load_reranker_model(self.model_id)

    def rerank(self, query: str, passages: List[str], k: int = 5) -> List[Tuple[int, float]]:
        """
        Rerank a list of passages based on their relevance to a query.
        Returns a list of tuples (original_index, score), sorted descending by score.
        """
        self.ensure_loaded()
        if not self._is_available or self._model is None:
            logger.warning("Using fallback random scores for reranking.")
            # Fallback: original order with dummy scores
            scores = [(i, 1.0 / (i + 1)) for i in range(len(passages))]
            return sorted(scores, key=lambda x: x[1], reverse=True)[:k]
            
        try:
            pairs = [[query, passage] for passage in passages]
            scores = self._model.predict(pairs)
            
            scored_passages = [(i, float(score)) for i, score in enumerate(scores)]
            scored_passages.sort(key=lambda x: x[1], reverse=True)
            
            return scored_passages[:k]
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            fallback = [(i, 1.0 / (i + 1)) for i in range(len(passages))]
            return sorted(fallback, key=lambda x: x[1], reverse=True)[:k]

    def score_pair(self, query: str, passage: str) -> float:
        """Score a single query-passage pair."""
        self.ensure_loaded()
        if not self._is_available or self._model is None:
            return 0.5
        try:
            return float(self._model.predict([[query, passage]])[0])
        except Exception as e:
            logger.error(f"Error in scoring pair: {e}")
            return 0.5
