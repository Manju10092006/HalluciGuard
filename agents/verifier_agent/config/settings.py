"""
HalluciGuard Configuration — Pydantic Settings with Environment Variable Loading.

All application-wide configuration is managed through this module.
Settings are loaded from ``.env`` file and/or environment variables.
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # -- API Keys (optional — system works without these) --
    alpha_vantage_key: Optional[str] = None
    openfda_key: Optional[str] = None
    nvd_api_key: Optional[str] = None
    semantic_scholar_key: Optional[str] = None
    courtlistener_key: Optional[str] = None
    sec_user_agent: str = "HalluciGuard/2.0 compliance@halluciguard.ai"

    # -- Feature flags --
    mock_mode: bool = False
    allow_model_downloads: bool = False
    enable_domain_classifier: bool = False
    enable_confidence_calibration: bool = True

    # -- Model selections --
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    reranker_model: str = "BAAI/bge-reranker-large"
    embedding_model: str = "BAAI/bge-m3"
    zero_shot_model: str = "facebook/bart-large-mnli"

    # -- Model runtime --
    max_cached_models: int = 8
    gpu_memory_threshold_mb: int = 512
    default_latency_budget: str = "balanced"

    # -- Cache --
    cache_ttl: int = 86400
    verifier_cache_enabled: bool = True

    # -- Server --
    verifier_port: int = 8002
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
