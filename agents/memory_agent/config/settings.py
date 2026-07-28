from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    memory_agent_port: int = 8005
    mock_mode: bool = False

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
