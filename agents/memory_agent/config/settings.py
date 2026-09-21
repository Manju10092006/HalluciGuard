from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    memory_agent_port: int = 8005
    mock_mode: bool = False
    memory_agent_api_key: Optional[str] = None

    # Knowledge Graph
    kg_persistence_path: str = "data/knowledge_graph.json"
    kg_max_nodes: int = 100_000
    kg_edge_weight_decay: float = 0.95

    # Verification Cache
    cache_db_path: str = "data/verification_cache.db"
    cache_ttl: int = 86400

    # Vector Store
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_store_path: str = "data/vector_store"
    vector_dimension: int = 384
    vector_top_k: int = 10

    # Pattern Learning
    pattern_db_path: str = "data/patterns.db"
    pattern_min_support: int = 3
    pattern_confidence_threshold: float = 0.6

    # Source Trust
    trust_db_path: str = "data/source_trust.db"
    trust_prior: float = 0.5
    trust_learning_rate: float = 0.1
    trust_decay_rate: float = 0.01

    # Intelligence (v1.1)
    duplicate_similarity_threshold: float = 0.95
    contradiction_similarity_threshold: float = 0.75
    fuzzy_cache_threshold: float = 0.85
    contradiction_top_k: int = 5
    rerank_alpha: float = 0.6
    rerank_beta: float = 0.3
    rerank_gamma: float = 0.1
    recall_min_similarity: float = 0.0

    # Contradiction confirmation (v1.3)
    # Vector similarity only SURFACES candidate contradictions; a second stage
    # confirms them. When nli_contradiction_check is enabled an NLI model is
    # used; otherwise (or when the model is unavailable) explicit structured
    # comparison runs (negation asymmetry, temporal-value mismatch).
    nli_contradiction_check: bool = False
    nli_contradiction_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_contradiction_threshold: float = 0.5

    # Storage journal (v1.3)
    # SQLite-backed operation ledger used to detect partial writes across the
    # multiple subsystems and to drive reconciliation of derived indexes.
    storage_journal_path: str = "data/storage_operations.db"

    # Domain support
    supported_domains: list[str] = [
        "healthcare",
        "finance",
        "cybersecurity",
        "legal",
        "ai_research",
        "science",
        "news",
        "general",
    ]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
