from __future__ import annotations
import logging
from typing import List

from schemas.models import Passage

class CrossEncoderReranker:
    """Reranks passages using a cross-encoder model."""

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2') -> None:
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._is_available = True
        
    def _load_model(self) -> None:
        if self.model is not None or not self._is_available:
            return
            
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        except ImportError:
            logging.warning("transformers not installed. CrossEncoderReranker falling back.")
            self._is_available = False
        except Exception as e:
            logging.warning(f"Error loading cross-encoder model: {e}. Falling back.")
            self._is_available = False

    def rerank(self, claim: str, passages: List[Passage], k: int) -> List[Passage]:
        """
        Rerank passages against a claim.
        
        Args:
            claim: The claim text.
            passages: The list of candidate passages.
            k: The number of passages to return.
            
        Returns:
            Top-k passages sorted by cross-encoder score.
        """
        self._load_model()
        
        if not self._is_available or not passages:
            return passages[:k]
            
        try:
            import torch
            
            pairs = [(claim, p.snippet) for p in passages]
            features = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
            
            with torch.no_grad():
                scores = self.model(**features).logits
                
            if scores.shape[1] > 1:
                # If model returns multiple logits, typically take the positive class (e.g. idx 1)
                scores = scores[:, 1]
            else:
                scores = scores.squeeze(-1)
                
            scores = scores.tolist()
            
            scored_passages = [(passages[i], scores[i]) for i in range(len(passages))]
            scored_passages.sort(key=lambda x: x[1], reverse=True)
            
            # Optionally update relevance_score
            for p, s in scored_passages:
                p.relevance_score = s
                
            return [p for p, s in scored_passages[:k]]
            
        except Exception as e:
            logging.warning(f"Error during reranking: {e}. Returning original passages.")
            return passages[:k]
