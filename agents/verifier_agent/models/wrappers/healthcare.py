import logging
from typing import List, Dict
import numpy as np
from .base import BaseModelWrapper
from models.model_manager import get_model_manager

logger = logging.getLogger(__name__)

class HealthcareModelWrapper(BaseModelWrapper):
    """
    Wrapper for biomedical and healthcare domain models.
    Supports:
    - dmis-lab/biobert-base-cased-v1.1
    - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract (default)
    - NeuML/pubmedbert-base-embeddings
    - allenai/scibert_scivocab_uncased
    - emilyalsentzer/Bio_ClinicalBERT
    - cambridgeltl/SapBERT-from-PubMedBERT-fulltext
    """
    
    def __init__(self, model_id: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        super().__init__(model_id)
        self._dim = 768
        self._st_model = None

    def _load_internal(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            # We use sentence transformers for embeddings for these models
            self._st_model = get_model_manager().load_embedding_model(self.model_id)
        except ImportError:
            logger.error("sentence_transformers not available")
            raise
            
        try:
            from transformers import pipeline
            self._classification_pipeline = pipeline("text-classification", model=self.model_id)
            self._ner_pipeline = pipeline("ner", model=self.model_id, aggregation_strategy="simple")
        except Exception as e:
            logger.warning(f"Transformers pipeline load failed: {e}")
            self._classification_pipeline = None
            self._ner_pipeline = None

    def embed(self, texts: List[str]) -> np.ndarray:
        """Domain-specific embeddings for healthcare texts."""
        self.ensure_loaded()
        if not self._is_available or self._st_model is None:
            logger.warning("Using fallback zero embeddings for healthcare.")
            return np.zeros((len(texts), self._dim), dtype=np.float32)
        try:
            return self._st_model.encode(texts, normalize_embeddings=True)
        except Exception as e:
            logger.error(f"Error in healthcare embed: {e}")
            return np.zeros((len(texts), self._dim), dtype=np.float32)

    def classify(self, text: str) -> Dict[str, float]:
        """Medical text classification."""
        self.ensure_loaded()
        if not self._is_available or self._classification_pipeline is None:
            return {"LABEL_0": 0.5, "LABEL_1": 0.5}
        try:
            res = self._classification_pipeline(text)[0]
            return {res["label"]: res["score"]}
        except Exception as e:
            logger.error(f"Error in healthcare classify: {e}")
            return {"LABEL_0": 0.5, "LABEL_1": 0.5}

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Biomedical NER entity extraction."""
        self.ensure_loaded()
        if not self._is_available or self._ner_pipeline is None:
            return []
        try:
            res = self._ner_pipeline(text)
            return [{"entity_group": r.get("entity_group", ""), "word": r.get("word", ""), "score": str(r.get("score", 0.0))} for r in res]
        except Exception as e:
            logger.error(f"Error in healthcare extract_entities: {e}")
            return []
