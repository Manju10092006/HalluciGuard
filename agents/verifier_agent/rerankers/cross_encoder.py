from __future__ import annotations
import logging
from typing import List

from schemas.models import Passage
from models.model_manager import get_model_manager

class CrossEncoderReranker:
    """Reranks passages using a cross-encoder model."""

    def __init__(self, model_name: str = 'BAAI/bge-reranker-large') -> None:
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._is_available = True
        self._load_attempts = 0
        
    def _load_model(self) -> None:
        if self.model is not None or not self._is_available:
            return
            
        if self._load_attempts >= 2:
            self._is_available = False
            return
            
        self._load_attempts += 1
            
        try:
            self.model = get_model_manager().load_reranker_model(self.model_name)
            self._load_attempts = 0
        except ImportError:
            logging.warning("transformers not installed. CrossEncoderReranker falling back.")
            self._is_available = False
        except Exception as e:
            logging.warning(f"Error loading cross-encoder model: {e}. Falling back.")
            if self._load_attempts >= 2:
                self._is_available = False

    def rerank(
        self,
        claim: str,
        passages: List[Passage],
        k: int,
        model_name: str | None = None,
    ) -> List[Passage]:
        """
        Rerank passages against a claim.
        
        Args:
            claim: The claim text.
            passages: The list of candidate passages.
            k: The number of passages to return.
            
        Returns:
            Top-k passages sorted by cross-encoder score.
        """
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self.model = None
            self._is_available = True
        self._load_model()
        
        if not self._is_available or not passages:
            logging.warning("Reranking skipped (model unavailable or empty passages). Returning unsorted.")
            return [p.model_copy(update={"relevance_score": 0.0}) for p in passages[:k]]
            
        try:
            pairs = [(claim, p.snippet) for p in passages]
            scores = self.model.predict(pairs, batch_size=16).tolist()
            
            scored_passages = [(passages[i], scores[i]) for i in range(len(passages))]
            scored_passages.sort(key=lambda x: x[1], reverse=True)
            
            return [
                p.model_copy(update={"relevance_score": float(s)})
                for p, s in scored_passages[:k]
            ]
            
        except Exception as e:
            logging.warning("Reranking failed for %d passages: %s. Returning unsorted.", len(passages), e)
            return [p.model_copy(update={"relevance_score": 0.0}) for p in passages[:k]]
