import logging
from typing import List, Dict
import numpy as np
from .base import BaseModelWrapper
from models.model_manager import get_model_manager

logger = logging.getLogger(__name__)

class LegalModelWrapper(BaseModelWrapper):
    """
    Wrapper for legal domain models.
    Supports: nlpaueb/legal-bert-base-uncased (default)
    """
    
    def __init__(self, model_id: str = "nlpaueb/legal-bert-base-uncased"):
        super().__init__(model_id)
        self._dim = 768
        self._st_model = None

    def _load_internal(self) -> None:
        try:
            self._st_model = get_model_manager().load_embedding_model(self.model_id)
        except Exception as e:
            logger.error(f"Failed to load legal embedding model: {e}")
            raise
            
        try:
            from transformers import pipeline
            self._classification_pipeline = pipeline("text-classification", model=self.model_id)
        except Exception as e:
            logger.warning(f"Transformers pipeline load failed: {e}")
            self._classification_pipeline = None

    def embed(self, texts: List[str]) -> np.ndarray:
        """Legal domain embeddings."""
        self.ensure_loaded()
        if not self._is_available or self._st_model is None:
            logger.warning("Using fallback zero embeddings for legal.")
            return np.zeros((len(texts), self._dim), dtype=np.float32)
        try:
            return self._st_model.encode(texts, normalize_embeddings=True)
        except Exception as e:
            logger.error(f"Error in legal embed: {e}")
            return np.zeros((len(texts), self._dim), dtype=np.float32)

    def classify(self, text: str) -> Dict[str, float]:
        """Legal text classification."""
        self.ensure_loaded()
        if not self._is_available or self._classification_pipeline is None:
            return {"LABEL_0": 0.5, "LABEL_1": 0.5}
        try:
            res = self._classification_pipeline(text)[0]
            return {res["label"]: res["score"]}
        except Exception as e:
            logger.error(f"Error in legal classify: {e}")
            return {"LABEL_0": 0.5, "LABEL_1": 0.5}
