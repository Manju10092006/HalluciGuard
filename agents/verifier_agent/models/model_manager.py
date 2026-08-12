"""
HalluciGuard Model Manager — Production Runtime Model Lifecycle Controller.

Responsibilities:
  - Thread-safe singleton model loading and caching.
  - Lazy loading: models are loaded only on first request.
  - GPU detection with automatic CPU fallback.
  - Memory-aware LRU eviction of infrequently-used models.
  - Domain-specific model loading via the wrapper subsystem.
  - Comprehensive runtime status reporting.

All other modules MUST use this manager to load ML models.
Direct instantiation of HuggingFace models outside this module is prohibited.
"""
from __future__ import annotations

import gc
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline as hf_pipeline

from config.settings import get_settings

logger = logging.getLogger("halluciguard.model_manager")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_MAX_CACHED_MODELS = int(os.environ.get("HALLUCIGUARD_MAX_MODELS", "8"))
_GPU_MEMORY_THRESHOLD_MB = int(os.environ.get("HALLUCIGUARD_GPU_THRESHOLD_MB", "512"))


class ModelManager:
    """
    Centralized lifecycle controller for every production ML model.

    Guarantees:
      - Singleton instance via ``get_instance()``.
      - Thread-safe loading protected by a reentrant lock.
      - Each model is loaded at most once and reused across requests.
      - LRU eviction when the number of cached models exceeds the limit.
      - Transparent CPU fallback when GPU memory is exhausted.
    """

    _instance: Optional["ModelManager"] = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        self.device = self._detect_device()
        self._models: OrderedDict[str, Any] = OrderedDict()
        self._model_load_times: Dict[str, float] = {}
        self._model_access_counts: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._max_cached = _DEFAULT_MAX_CACHED_MODELS
        logger.info(
            "ModelManager initialized — device=%s, max_cached=%d",
            self.device,
            self._max_cached,
        )

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            try:
                free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 2)
                if free_mem > _GPU_MEMORY_THRESHOLD_MB:
                    logger.info("CUDA available — %.0f MB free", free_mem)
                    return "cuda"
                logger.warning(
                    "CUDA available but only %.0f MB free (threshold %d MB). Using CPU.",
                    free_mem,
                    _GPU_MEMORY_THRESHOLD_MB,
                )
            except Exception as e:
                logger.warning("CUDA available but mem_get_info failed: %s. Defaulting to CPU.", e)
                return "cpu"
        return "cpu"

    def get_device(self) -> str:
        """Return the active compute device string."""
        return self.device

    @property
    def _hf_device(self) -> int:
        """Return ``0`` for CUDA, ``-1`` for CPU (HuggingFace pipeline convention)."""
        return 0 if self.device == "cuda" else -1

    # ------------------------------------------------------------------
    # LRU eviction
    # ------------------------------------------------------------------
    def _evict_if_needed(self) -> None:
        """Evict the least-recently-used model if cache is full."""
        while len(self._models) > self._max_cached:
            evicted_name, evicted_model = self._models.popitem(last=False)
            del evicted_model
            self._model_load_times.pop(evicted_name, None)
            self._model_access_counts.pop(evicted_name, None)
            logger.info("Evicted model '%s' from cache (LRU)", evicted_name)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def _touch(self, model_name: str) -> None:
        """Move model to end of OrderedDict (most recently used)."""
        if model_name in self._models:
            self._models.move_to_end(model_name)
            self._model_access_counts[model_name] = (
                self._model_access_counts.get(model_name, 0) + 1
            )

    # ------------------------------------------------------------------
    # Core loaders
    # ------------------------------------------------------------------
    def load_embedding_model(self, model_name: Optional[str] = None) -> Any:
        """Load and cache a SentenceTransformer embedding model."""
        model_name = model_name or get_settings().embedding_model
        with self._lock:
            if model_name in self._models:
                self._touch(model_name)
                return self._models[model_name]
            self._evict_if_needed()
            settings = get_settings()
            t0 = time.monotonic()
            try:
                model = SentenceTransformer(
                    model_name,
                    device=self.device,
                    local_files_only=not settings.allow_model_downloads,
                )
            except Exception as primary_err:
                if self.device == "cuda":
                    logger.warning(
                        "GPU load failed for '%s': %s — retrying on CPU",
                        model_name,
                        primary_err,
                    )
                    model = SentenceTransformer(
                        model_name,
                        device="cpu",
                        local_files_only=not settings.allow_model_downloads,
                    )
                else:
                    raise
            elapsed = time.monotonic() - t0
            self._models[model_name] = model
            self._model_load_times[model_name] = elapsed
            self._model_access_counts[model_name] = 1
            logger.info("Loaded embedding model '%s' in %.2fs", model_name, elapsed)
            return model

    def load_reranker_model(self, model_name: Optional[str] = None) -> Any:
        """Load and cache a CrossEncoder reranker model."""
        model_name = model_name or get_settings().reranker_model
        with self._lock:
            if model_name in self._models:
                self._touch(model_name)
                return self._models[model_name]
            self._evict_if_needed()
            settings = get_settings()
            t0 = time.monotonic()
            ce_kwargs: Dict[str, Any] = {"device": self.device}
            if not os.path.isdir(model_name) and not settings.allow_model_downloads:
                ce_kwargs["model_kwargs"] = {"local_files_only": True}
                ce_kwargs["tokenizer_kwargs"] = {"local_files_only": True}

            try:
                model = CrossEncoder(model_name, **ce_kwargs)
            except Exception as primary_err:
                if self.device == "cuda":
                    logger.warning(
                        "GPU load failed for reranker '%s': %s — retrying on CPU",
                        model_name,
                        primary_err,
                    )
                    ce_kwargs["device"] = "cpu"
                    model = CrossEncoder(model_name, **ce_kwargs)
                else:
                    raise
            elapsed = time.monotonic() - t0
            self._models[model_name] = model
            self._model_load_times[model_name] = elapsed
            self._model_access_counts[model_name] = 1
            logger.info("Loaded reranker model '%s' in %.2fs", model_name, elapsed)
            return model

    def load_nli_model(self, model_name: Optional[str] = None) -> Any:
        """Load and cache an NLI text-classification pipeline."""
        settings = get_settings()
        default_nli = settings.nli_model
        if not model_name:
            model_name = default_nli
        elif not settings.allow_model_downloads and os.path.isdir(default_nli) and not os.path.isdir(model_name):
            logger.info("Using configured local NLI model '%s' instead of remote '%s'", default_nli, model_name)
            model_name = default_nli

        with self._lock:
            if model_name in self._models:
                self._touch(model_name)
                return self._models[model_name]
            self._evict_if_needed()
            t0 = time.monotonic()

            kwargs: Dict[str, Any] = {
                "task": "text-classification",
                "model": model_name,
                "top_k": None,
                "device": self._hf_device,
            }
            if not os.path.isdir(model_name) and not settings.allow_model_downloads:
                kwargs["model_kwargs"] = {"local_files_only": True}
                kwargs["tokenizer_kwargs"] = {"local_files_only": True}

            try:
                model = hf_pipeline(**kwargs)
            except Exception as primary_err:
                if self.device == "cuda":
                    logger.warning(
                        "GPU load failed for NLI '%s': %s — retrying on CPU",
                        model_name,
                        primary_err,
                    )
                    kwargs["device"] = -1
                    model = hf_pipeline(**kwargs)
                else:
                    raise
            elapsed = time.monotonic() - t0
            self._models[model_name] = model
            self._model_load_times[model_name] = elapsed
            self._model_access_counts[model_name] = 1
            logger.info("Loaded NLI model '%s' in %.2fs", model_name, elapsed)
            return model

    def load_zero_shot_model(self) -> Any:
        """Load and cache a zero-shot classification pipeline."""
        model_name = get_settings().zero_shot_model
        with self._lock:
            if model_name in self._models:
                self._touch(model_name)
                return self._models[model_name]
            self._evict_if_needed()
            settings = get_settings()
            t0 = time.monotonic()
            try:
                model = hf_pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=self._hf_device,
                    model_kwargs={"local_files_only": not settings.allow_model_downloads},
                    tokenizer_kwargs={"local_files_only": not settings.allow_model_downloads},
                )
            except Exception as primary_err:
                if self.device == "cuda":
                    logger.warning("GPU load failed for zero-shot '%s': %s — retrying on CPU", model_name, primary_err)
                    model = hf_pipeline(
                        "zero-shot-classification",
                        model=model_name,
                        device=-1,
                        model_kwargs={"local_files_only": not settings.allow_model_downloads},
                        tokenizer_kwargs={"local_files_only": not settings.allow_model_downloads},
                    )
                else:
                    raise
            elapsed = time.monotonic() - t0
            self._models[model_name] = model
            self._model_load_times[model_name] = elapsed
            self._model_access_counts[model_name] = 1
            logger.info("Loaded zero-shot model '%s' in %.2fs", model_name, elapsed)
            return model

    def load_classification_pipeline(
        self,
        model_name: str,
        task: str = "text-classification",
    ) -> Any:
        """Load a generic HuggingFace classification pipeline."""
        cache_key = f"{task}::{model_name}"
        with self._lock:
            if cache_key in self._models:
                self._touch(cache_key)
                return self._models[cache_key]
            self._evict_if_needed()
            settings = get_settings()
            t0 = time.monotonic()
            try:
                model = hf_pipeline(
                    task,
                    model=model_name,
                    top_k=None,
                    device=self._hf_device,
                    model_kwargs={"local_files_only": not settings.allow_model_downloads},
                    tokenizer_kwargs={"local_files_only": not settings.allow_model_downloads},
                )
            except Exception as primary_err:
                if self.device == "cuda":
                    logger.warning("GPU load failed for pipeline '%s' (%s): %s — retrying on CPU", model_name, task, primary_err)
                    model = hf_pipeline(
                        task,
                        model=model_name,
                        top_k=None,
                        device=-1,
                        model_kwargs={"local_files_only": not settings.allow_model_downloads},
                        tokenizer_kwargs={"local_files_only": not settings.allow_model_downloads},
                    )
                else:
                    raise
            elapsed = time.monotonic() - t0
            self._models[cache_key] = model
            self._model_load_times[cache_key] = elapsed
            self._model_access_counts[cache_key] = 1
            logger.info("Loaded pipeline '%s' (%s) in %.2fs", model_name, task, elapsed)
            return model

    def load_ner_pipeline(self, model_name: str) -> Any:
        """Load a Named Entity Recognition pipeline."""
        return self.load_classification_pipeline(
            model_name, task="ner"
        )

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def unload_model(self, model_name: str) -> bool:
        """Unload a specific model from cache. Returns True if found."""
        with self._lock:
            if model_name in self._models:
                del self._models[model_name]
                self._model_load_times.pop(model_name, None)
                self._model_access_counts.pop(model_name, None)
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info("Unloaded model '%s'", model_name)
                return True
            return False

    def unload_all(self) -> None:
        """Unload every cached model and release GPU memory."""
        with self._lock:
            count = len(self._models)
            self._models.clear()
            self._model_load_times.clear()
            self._model_access_counts.clear()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Unloaded all %d cached models", count)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def is_loaded(self, model_name: str) -> bool:
        """Check whether a model is currently cached."""
        return model_name in self._models

    @property
    def loaded_models(self) -> List[str]:
        """Return names of all currently loaded models."""
        return list(self._models.keys())

    @property
    def cached_count(self) -> int:
        """Return the number of currently cached models."""
        return len(self._models)

    def status(self) -> Dict[str, Any]:
        """Return comprehensive runtime status for the /health endpoint."""
        settings = get_settings()
        all_expected = [
            settings.nli_model,
            settings.reranker_model,
            settings.embedding_model,
            settings.zero_shot_model,
        ]
        model_status = {}
        for m in all_expected:
            if m in self._models:
                model_status[m] = {
                    "status": "loaded",
                    "load_time_s": round(self._model_load_times.get(m, 0), 2),
                    "access_count": self._model_access_counts.get(m, 0),
                }
            else:
                model_status[m] = {"status": "not_loaded"}

        try:
            from .domain_intelligence import get_domain_intelligence_registry

            registry = get_domain_intelligence_registry()
            domain_info = {
                "domain_profiles": f"{len(registry.list_domains())}_configured",
                "domain_model_ids": str(len(registry.unique_model_ids())),
            }
        except Exception as e:
            logger.warning("Failed to get domain intelligence info: %s", e)
            domain_info = {"domain_profiles": "unavailable", "domain_model_ids": "unavailable"}

        gpu_info: Dict[str, Any] = {"device": self.device}
        if self.device == "cuda":
            try:
                free, total = torch.cuda.mem_get_info()
                gpu_info["gpu_free_mb"] = round(free / (1024 ** 2), 1)
                gpu_info["gpu_total_mb"] = round(total / (1024 ** 2), 1)
            except Exception as e:
                logger.warning("Failed to get GPU mem info: %s", e)
                gpu_info["gpu_free_mb"] = -1.0
                gpu_info["gpu_total_mb"] = -1.0

        return {
            **model_status,
            "cached_models": self.cached_count,
            "max_cached": self._max_cached,
            "hardware": gpu_info,
            **domain_info,
        }


def get_model_manager() -> ModelManager:
    """Return the global ModelManager singleton."""
    return ModelManager.get_instance()
