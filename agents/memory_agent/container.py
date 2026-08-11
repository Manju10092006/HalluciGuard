from __future__ import annotations

import logging
from typing import Optional

from .cache.verification_cache import VerificationCache
from .config.settings import Settings, get_settings
from .knowledge_graph.graph import KnowledgeGraph
from .memory.memory_agent import MemoryAgent
from .patterns.pattern_learner import PatternLearner
from .trust.source_trust import SourceTrustManager
from .vector_store.faiss_store import VectorStore

logger = logging.getLogger(__name__)


class Container:
    def __init__(
        self,
        settings: Settings,
        knowledge_graph: KnowledgeGraph,
        cache: VerificationCache,
        pattern_learner: PatternLearner,
        source_trust: SourceTrustManager,
        vector_store: VectorStore,
        memory_agent: MemoryAgent,
    ):
        self.settings = settings
        self.knowledge_graph = knowledge_graph
        self.cache = cache
        self.pattern_learner = pattern_learner
        self.source_trust = source_trust
        self.vector_store = vector_store
        self.memory_agent = memory_agent


async def create_container(settings: Optional[Settings] = None) -> Container:
    settings = settings or get_settings()

    kg = KnowledgeGraph(
        persistence_path=settings.kg_persistence_path,
        max_nodes=settings.kg_max_nodes,
        weight_decay=settings.kg_edge_weight_decay,
    )
    cache = VerificationCache(
        db_path=settings.cache_db_path, ttl=settings.cache_ttl
    )
    patterns = PatternLearner(
        db_path=settings.pattern_db_path,
        min_support=settings.pattern_min_support,
        confidence_threshold=settings.pattern_confidence_threshold,
    )
    trust = SourceTrustManager(
        db_path=settings.trust_db_path,
        prior=settings.trust_prior,
        learning_rate=settings.trust_learning_rate,
        decay_rate=settings.trust_decay_rate,
    )
    vectors = VectorStore(
        embedding_model=settings.embedding_model,
        store_path=settings.vector_store_path,
        dimension=settings.vector_dimension,
        top_k=settings.vector_top_k,
    )

    agent = MemoryAgent(
        settings=settings,
        knowledge_graph=kg,
        cache=cache,
        pattern_learner=patterns,
        source_trust=trust,
        vector_store=vectors,
    )
    await agent.initialize()

    return Container(
        settings=settings,
        knowledge_graph=kg,
        cache=cache,
        pattern_learner=patterns,
        source_trust=trust,
        vector_store=vectors,
        memory_agent=agent,
    )


_container_instance: Optional[Container] = None


async def get_container() -> Container:
    global _container_instance
    if _container_instance is None:
        _container_instance = await create_container()
    return _container_instance
