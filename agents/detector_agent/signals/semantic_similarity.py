import math
from typing import Optional
from pydantic import BaseModel, Field
from ..utils import init_dll_paths
init_dll_paths()

import torch
import torch.nn.functional as F
from ..model_manager import ModelManager


class SemanticSimilarityMetrics(BaseModel):
    """Structured output containing semantic similarity metrics between query and response."""

    cosine_similarity: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Cosine similarity between query and response embeddings (-1.0 to 1.0)."
    )
    semantic_distance: float = Field(
        ...,
        ge=0.0,
        description="Semantic distance between query and response embeddings (1.0 - cosine_similarity)."
    )
    normalized_similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized similarity score bounded strictly in [0.0, 1.0]."
    )


class SemanticSimilarityCalculator:
    """Computes embedding-based semantic similarity between user query and LLM response."""

    def __init__(self, model_manager: ModelManager) -> None:
        """Initialize SemanticSimilarityCalculator with a shared ModelManager instance."""
        self.model_manager: ModelManager = model_manager
        self._fallback_tokenizer = None
        self._fallback_model = None

    def _get_fallback_embedding_model(self):
        """Native Hugging Face transformers fallback when sentence_transformers is not installed."""
        if self._fallback_tokenizer is None or self._fallback_model is None:
            from transformers import AutoModel, AutoTokenizer
            model_name = self.model_manager.embedding_model_name
            if not model_name.startswith("sentence-transformers/"):
                model_name = f"sentence-transformers/{model_name}"
            self._fallback_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._fallback_model = AutoModel.from_pretrained(model_name)
            self._fallback_model.eval()
        return self._fallback_tokenizer, self._fallback_model

    def _encode_text(self, text: str) -> torch.Tensor:
        """Helper to encode a single text into a normalized sentence embedding tensor [1, D]."""
        embedding_model = self.model_manager.get_embedding_model()
        if embedding_model is not None:
            emb = embedding_model.encode(text, convert_to_tensor=True)
            if len(emb.shape) == 1:
                emb = emb.unsqueeze(0)
            return emb

        tokenizer, model = self._get_fallback_embedding_model()
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            token_embeddings = outputs[0]
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            return F.normalize(embedding, p=2, dim=1)

    def compute(self, user_query: str, llm_response: str) -> Optional[SemanticSimilarityMetrics]:
        """Compute cosine_similarity, semantic_distance, and normalized_similarity."""
        try:
            emb_query = self._encode_text(user_query)
            emb_response = self._encode_text(llm_response)

            cos_sim_tensor = F.cosine_similarity(emb_query, emb_response)
            cos_sim = cos_sim_tensor.item()

            semantic_dist = 1.0 - cos_sim
            normalized_sim = (cos_sim + 1.0) / 2.0
            normalized_sim = max(0.0, min(1.0, normalized_sim))

            return SemanticSimilarityMetrics(
                cosine_similarity=round(cos_sim, 6),
                semantic_distance=round(semantic_dist, 6),
                normalized_similarity=round(normalized_sim, 6)
            )

        except Exception as e:
            print(f"[SemanticSimilarityCalculator] Warning: Error computing similarity: {e}")
            return None
