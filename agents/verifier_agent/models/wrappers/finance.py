import logging
from typing import List, Dict
import numpy as np
from .base import BaseModelWrapper
from models.model_manager import get_model_manager

logger = logging.getLogger(__name__)

class FinanceModelWrapper(BaseModelWrapper):
    """
    Wrapper for financial domain models.
    Supports: ProsusAI/finbert (default), yiyanghkust/finbert-tone
    """
    
    def __init__(self, model_id: str = "ProsusAI/finbert"):
        super().__init__(model_id)
        self._dim = 768
        self._st_model = None

    def _load_internal(self) -> None:
        try:
            self._st_model = get_model_manager().load_embedding_model(self.model_id)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
            
        try:
            from transformers import pipeline
            self._sentiment_pipeline = pipeline("text-classification", model=self.model_id)
            self._tone_pipeline = pipeline("text-classification", model="yiyanghkust/finbert-tone")
        except Exception as e:
            logger.warning(f"Failed to load finance pipelines: {e}")
            self._sentiment_pipeline = None
            self._tone_pipeline = None

    def embed(self, texts: List[str]) -> np.ndarray:
        """Finance embeddings via SentenceTransformer."""
        self.ensure_loaded()
        if not self._is_available or self._st_model is None:
            logger.warning("Using fallback zero embeddings for finance.")
            return np.zeros((len(texts), self._dim), dtype=np.float32)
        try:
            return self._st_model.encode(texts, normalize_embeddings=True)
        except Exception as e:
            logger.error(f"Error in finance embed: {e}")
            return np.zeros((len(texts), self._dim), dtype=np.float32)

    def classify_sentiment(self, text: str) -> Dict[str, float]:
        """Financial sentiment (positive/negative/neutral)."""
        self.ensure_loaded()
        if not self._is_available or self._sentiment_pipeline is None:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
        try:
            res = self._sentiment_pipeline(text)[0]
            return {res["label"].lower(): res["score"]}
        except Exception as e:
            logger.error(f"Error in finance classify_sentiment: {e}")
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

    def analyze_tone(self, text: str) -> Dict[str, float]:
        """Financial tone analysis."""
        self.ensure_loaded()
        if not self._is_available or self._tone_pipeline is None:
            return {"bullish": 0.33, "bearish": 0.33, "neutral": 0.34}
        try:
            res = self._tone_pipeline(text)[0]
            return {res["label"].lower(): res["score"]}
        except Exception as e:
            logger.error(f"Error in finance analyze_tone: {e}")
            return {"bullish": 0.33, "bearish": 0.33, "neutral": 0.34}
