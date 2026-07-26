from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    alpha_vantage_key: Optional[str] = None
    openfda_key: Optional[str] = None
    nvd_api_key: Optional[str] = None
    semantic_scholar_key: Optional[str] = None
    courtlistener_key: Optional[str] = None
    sec_user_agent: str = "HalluciGuard/2.0 compliance@halluciguard.ai"
    
    mock_mode: bool = False
    
    nli_model: str = "microsoft/deberta-v3-base-mnli"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    zero_shot_model: str = "facebook/bart-large-mnli"
    
    cache_ttl: int = 86400
    verifier_port: int = 8002
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache()
def get_settings() -> Settings:
    return Settings()
