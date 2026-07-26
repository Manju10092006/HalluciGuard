from typing import Any, Dict, Optional
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder

from config.settings import get_settings

class ModelManager:
    _instance: Optional['ModelManager'] = None

    def __init__(self) -> None:
        self.device = self.get_device()
        self._models: Dict[str, Any] = {}
        
    @classmethod
    def get_instance(cls) -> 'ModelManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def get_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_nli_model(self) -> Any:
        model_name = get_settings().nli_model
        if model_name not in self._models:
            self._models[model_name] = pipeline(
                "text-classification",
                model=model_name,
                top_k=None,
                device=0 if self.device == "cuda" else -1,
            )
        return self._models[model_name]

    def load_reranker_model(self) -> Any:
        model_name = get_settings().reranker_model
        if model_name not in self._models:
            self._models[model_name] = CrossEncoder(model_name, device=self.device)
        return self._models[model_name]

    def load_embedding_model(self) -> Any:
        model_name = get_settings().embedding_model
        if model_name not in self._models:
            self._models[model_name] = SentenceTransformer(model_name, device=self.device)
        return self._models[model_name]

    def load_zero_shot_model(self) -> Any:
        model_name = get_settings().zero_shot_model
        if model_name not in self._models:
            self._models[model_name] = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
            )
        return self._models[model_name]

    def unload_all(self) -> None:
        self._models.clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def status(self) -> Dict[str, str]:
        all_expected_models = [
            get_settings().nli_model,
            get_settings().reranker_model,
            get_settings().embedding_model,
            get_settings().zero_shot_model,
        ]
        return {
            m: "loaded" if m in self._models else "not_loaded"
            for m in all_expected_models
        }

def get_model_manager() -> ModelManager:
    return ModelManager.get_instance()
