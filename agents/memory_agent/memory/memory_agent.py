from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

from ..cache.verification_cache import VerificationCache
from ..config.settings import Settings, get_settings
from ..knowledge_graph.graph import KnowledgeGraph
from ..patterns.pattern_learner import PatternLearner
from ..schemas.models import (
    CacheStats,
    EntityType,
    HallucinationPattern,
    MemoryStatsResponse,
    PatternType,
    RecallRequest,
    RecallResponse,
    RelationType,
    SourceTrustRecord,
    StoreFactRequest,
    StoreFactResponse,
    TrustChangeReason,
    TrustUpdate,
    VectorSearchResult,
)
from ..trust.source_trust import SourceTrustManager
from ..vector_store.faiss_store import VectorStore

logger = logging.getLogger(__name__)


class MemoryAgent:
    """Orchestrator that ties all memory subsystems together."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        cache: Optional[VerificationCache] = None,
        pattern_learner: Optional[PatternLearner] = None,
        source_trust: Optional[SourceTrustManager] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self._settings = settings or get_settings()
        self.kg = knowledge_graph or KnowledgeGraph(
            persistence_path=self._settings.kg_persistence_path,
            max_nodes=self._settings.kg_max_nodes,
            weight_decay=self._settings.kg_edge_weight_decay,
        )
        self.cache = cache or VerificationCache(
            db_path=self._settings.cache_db_path,
            ttl=self._settings.cache_ttl,
        )
        self.patterns = pattern_learner or PatternLearner(
            db_path=self._settings.pattern_db_path,
            min_support=self._settings.pattern_min_support,
            confidence_threshold=self._settings.pattern_confidence_threshold,
        )
        self.trust = source_trust or SourceTrustManager(
            db_path=self._settings.trust_db_path,
            prior=self._settings.trust_prior,
            learning_rate=self._settings.trust_learning_rate,
            decay_rate=self._settings.trust_decay_rate,
        )
        self.vectors = vector_store or VectorStore(
            embedding_model=self._settings.embedding_model,
            store_path=self._settings.vector_store_path,
            dimension=self._settings.vector_dimension,
            top_k=self._settings.vector_top_k,
        )

    async def initialize(self) -> None:
        await self.cache.initialize()
        await self.patterns.initialize()
        await self.trust.initialize()
        self.vectors.initialize()
        logger.info("Memory agent initialized")

    async def close(self) -> None:
        self.kg.save()
        self.vectors.save()
        await self.cache.close()
        await self.patterns.close()
        await self.trust.close()
        logger.info("Memory agent shut down")

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    async def store_fact(self, request: StoreFactRequest) -> StoreFactResponse:
        fact_id = str(uuid.uuid4())
        now = datetime.utcnow()

        claim_node = self.kg.add_entity(
            name=request.claim_text[:120],
            entity_type=EntityType.CLAIM,
            properties={
                "domain": request.domain,
                "verdict": request.verdict,
                "confidence": request.confidence,
                "fact_id": fact_id,
            },
            confidence=request.confidence,
        )

        fact_node = self.kg.add_entity(
            name=fact_id,
            entity_type=EntityType.FACT,
            properties={
                "domain": request.domain,
                "verdict": request.verdict,
                "claim_text": request.claim_text,
            },
            confidence=request.confidence,
        )
        self.kg.add_edge(
            source_id=claim_node.entity_id,
            target_id=fact_node.entity_id,
            relation=RelationType.DERIVED_FROM,
        )

        entities_created = 2
        edges_created = 1

        for evidence in request.evidence:
            source_id = evidence.get("source_id", "")
            if source_id:
                source_node = self.kg.add_entity(
                    name=source_id,
                    entity_type=EntityType.SOURCE,
                    properties={"domain": request.domain},
                )
                self.kg.add_edge(
                    source_id=fact_node.entity_id,
                    target_id=source_node.entity_id,
                    relation=RelationType.MENTIONS,
                )
                edges_created += 1

        self.vectors.add(
            text=request.claim_text,
            metadata={
                "domain": request.domain,
                "verdict": request.verdict,
                "confidence": request.confidence,
                "fact_id": fact_id,
                "timestamp": now.isoformat(),
            },
            entry_id=fact_id,
        )

        pattern_updated = False
        if request.verdict in ("likely_hallucinated", "contradicted"):
            patterns = await self.patterns.observe_claim(
                claim_text=request.claim_text,
                domain=request.domain,
                verdict=request.verdict,
            )
            pattern_updated = len(patterns) > 0

        await self.cache.set(
            domain=request.domain,
            claim_text=request.claim_text,
            verdict=request.verdict,
            evidence_summary=f"Evidence from {len(request.evidence)} sources",
            confidence=request.confidence,
            source_count=len(request.source_ids),
        )

        trust_updates: list[TrustUpdate] = []
        for sid in request.source_ids:
            reason = (
                TrustChangeReason.VERIFIED_CORRECT
                if request.verdict in ("verified", "likely_verified")
                else TrustChangeReason.VERIFIED_INCORRECT
            )
            update = await self.trust.update_trust(
                source_id=sid,
                source_name=sid,
                domain=request.domain,
                reason=reason,
                evidence_count=len(request.evidence),
            )
            trust_updates.append(update)

        self.kg.save()

        return StoreFactResponse(
            fact_id=fact_id,
            entities_created=entities_created,
            edges_created=edges_created,
            pattern_updated=pattern_updated,
            trust_updates=trust_updates,
        )

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    async def recall(self, request: RecallRequest) -> RecallResponse:
        cached: Optional[CacheStats] = None
        similar_facts: list[VectorSearchResult] = []
        related_entities = []
        relevant_patterns: list[HallucinationPattern] = []
        source_trust: list[SourceTrustRecord] = []

        if request.include_cache:
            cached_verification = await self.cache.get(
                domain=request.domain or "general",
                claim_text=request.query,
            )
        else:
            cached_verification = None

        similar_facts = self.vectors.search(
            query=request.query,
            top_k=request.top_k,
            metadata_filter={"domain": request.domain} if request.domain else None,
        )

        if request.include_graph_context:
            claim_nodes = self.kg.find_entity_by_name(request.query[:120])
            for node in claim_nodes[:5]:
                neighbors = self.kg.get_neighbors(node.entity_id)
                for neighbor, edge in neighbors:
                    neighbor.access_count += 1
                    related_entities.append(neighbor)

        if request.include_patterns:
            relevant_patterns = await self.patterns.query_patterns(
                domain=request.domain,
                top_k=10,
            )

        if request.domain:
            source_trust = await self.trust.get_domain_sources(
                domain=request.domain,
                min_trust=0.3,
            )

        return RecallResponse(
            cached_verification=cached_verification,
            similar_facts=similar_facts,
            related_entities=related_entities,
            relevant_patterns=relevant_patterns,
            source_trust=source_trust,
        )

    # ------------------------------------------------------------------
    # Cache operations
    # ------------------------------------------------------------------

    async def check_cache(
        self, domain: str, claim_text: str
    ) -> Optional[dict]:
        result = await self.cache.get(domain, claim_text)
        return result.model_dump() if result else None

    async def invalidate_cache(self, domain: str, claim_text: str) -> bool:
        return await self.cache.invalidate(domain, claim_text)

    # ------------------------------------------------------------------
    # Trust operations
    # ------------------------------------------------------------------

    async def update_source_trust(
        self,
        source_id: str,
        source_name: str,
        domain: str,
        reason: TrustChangeReason,
        evidence_count: int = 1,
    ) -> TrustUpdate:
        return await self.trust.update_trust(
            source_id=source_id,
            source_name=source_name,
            domain=domain,
            reason=reason,
            evidence_count=evidence_count,
        )

    async def get_source_trust(self, source_id: str) -> Optional[SourceTrustRecord]:
        return await self.trust.get_trust(source_id)

    # ------------------------------------------------------------------
    # Pattern operations
    # ------------------------------------------------------------------

    async def query_patterns(
        self,
        domain: Optional[str] = None,
        pattern_type: Optional[PatternType] = None,
        min_confidence: float = 0.0,
        top_k: int = 20,
    ) -> list[HallucinationPattern]:
        return await self.patterns.query_patterns(
            domain=domain,
            pattern_type=pattern_type,
            min_confidence=min_confidence,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> MemoryStatsResponse:
        kg_stats = self.kg.get_stats()
        cache_stats = await self.cache.get_stats()
        patterns_total = await self.patterns.get_total_count()
        sources_total = await self.trust.get_total_sources()
        vector_size = self.vectors.size()

        return MemoryStatsResponse(
            knowledge_graph=kg_stats,
            cache=cache_stats,
            patterns_total=patterns_total,
            sources_tracked=sources_total,
            vector_store_size=vector_size,
        )

    async def save_all(self) -> None:
        self.kg.save()
        self.vectors.save()


_memory_agent_instance: Optional[MemoryAgent] = None


def get_memory_agent(settings: Optional[Settings] = None) -> MemoryAgent:
    global _memory_agent_instance
    if _memory_agent_instance is None:
        _memory_agent_instance = MemoryAgent(settings=settings)
    return _memory_agent_instance
