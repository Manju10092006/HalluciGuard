from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelManager:
    """Singleton manager for loading and caching Hugging Face model, tokenizer, and embedding model instances.
    
    Ensures heavy ML models are loaded into memory exactly once and shared across DetectorAgent
    signal processors (Token Probability, Entropy, Semantic Similarity, etc.).
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", embedding_model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name: str = model_name
        self.embedding_model_name: str = embedding_model_name
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForCausalLM] = None
        self._embedding_model = None

    def get_model_and_tokenizer(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Loads and returns the cached tokenizer and causal model instance."""
        if self._tokenizer is None or self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.eval()
        return self._tokenizer, self._model

    def get_embedding_model(self):
        """Loads and returns the cached sentence-transformers embedding model instance."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                print(f"[ModelManager] Warning: Failed to load SentenceTransformer '{self.embedding_model_name}': {e}")
                return None
        return self._embedding_model
