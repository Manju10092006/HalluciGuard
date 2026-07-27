from typing import Any, Dict, Optional
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder

from config.settings import get_settings
from .domain_intelligence import get_domain_intelligence_registry

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

    def load_nli_model(self, model_name: Optional[str] = None) -> Any:
        model_name = model_name or get_settings().nli_model
        settings = get_settings()
        if model_name not in self._models:
            self._models[model_name] = pipeline(
                "text-classification",
                model=model_name,
                top_k=None,
                device=0 if self.device == "cuda" else -1,
                model_kwargs={"local_files_only": not settings.allow_model_downloads},
                tokenizer_kwargs={"local_files_only": not settings.allow_model_downloads},
            )
        return self._models[model_name]

    def load_reranker_model(self, model_name: Optional[str] = None) -> Any:
        model_name = model_name or get_settings().reranker_model
        settings = get_settings()
        if model_name not in self._models:
            self._models[model_name] = CrossEncoder(
                model_name,
                device=self.device,
                automodel_args={"local_files_only": not settings.allow_model_downloads},
                tokenizer_args={"local_files_only": not settings.allow_model_downloads},
            )
        return self._models[model_name]

    def load_embedding_model(self, model_name: Optional[str] = None) -> Any:
        model_name = model_name or get_settings().embedding_model
        settings = get_settings()
        if model_name not in self._models:
            self._models[model_name] = SentenceTransformer(
                model_name,
                device=self.device,
                local_files_only=not settings.allow_model_downloads,
            )
        return self._models[model_name]

    def load_zero_shot_model(self) -> Any:
        model_name = get_settings().zero_shot_model
        settings = get_settings()
        if model_name not in self._models:
            self._models[model_name] = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                model_kwargs={"local_files_only": not settings.allow_model_downloads},
                tokenizer_kwargs={"local_files_only": not settings.allow_model_downloads},
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
        status = {
            m: "loaded" if m in self._models else "not_loaded"
            for m in all_expected_models
        }
        try:
            registry = get_domain_intelligence_registry()
            status["domain_profiles"] = f"{len(registry.list_domains())}_configured"
            status["domain_model_ids"] = str(len(registry.unique_model_ids()))
        except Exception:
            status["domain_profiles"] = "unavailable"
        return status

def get_model_manager() -> ModelManager:
    return ModelManager.get_instance()
