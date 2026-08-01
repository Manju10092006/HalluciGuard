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
    BatchStoreResponse,
    CacheStats,
    CachedVerification,
    ContradictionAlert,
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

    @staticmethod
    def _verdict_conflicts(verdict_a: str, verdict_b: str) -> bool:
        positive = {"verified", "likely_verified"}
        negative = {"likely_hallucinated", "contradicted"}
        return (verdict_a in positive and verdict_b in negative) or (
            verdict_a in negative and verdict_b in positive
        )

    async def store_fact(self, request: StoreFactRequest) -> StoreFactResponse:
        fact_id = str(uuid.uuid4())
        now = datetime.utcnow()
        contradictions: list[ContradictionAlert] = []

        # Duplicate + contradiction detection against existing memory
        similar = self.vectors.search(
            query=request.claim_text,
            top_k=self._settings.contradiction_top_k,
            metadata_filter={"domain": request.domain} if request.domain else None,
        )
        duplicate_of: Optional[str] = None
        for res in similar:
            if (
                res.score >= self._settings.duplicate_similarity_threshold
                and res.metadata.get("verdict") == request.verdict
            ):
                duplicate_of = res.entry_id
            if (
                res.score >= self._settings.contradiction_similarity_threshold
                and self._verdict_conflicts(
                    request.verdict, str(res.metadata.get("verdict", ""))
                )
            ):
                contradictions.append(
                    ContradictionAlert(
                        existing_fact_id=res.entry_id,
                        existing_claim_text=res.text,
                        similarity_score=res.score,
                        existing_verdict=str(res.metadata.get("verdict", "unknown")),
                        reason=(
                            f"New verdict '{request.verdict}' conflicts with stored "
                            f"verdict '{res.metadata.get('verdict')}' for similar claim"
                        ),
                    )
                )

        if duplicate_of:
            logger.info("Duplicate claim detected, reusing fact %s", duplicate_of)
            return StoreFactResponse(
                fact_id=duplicate_of,
                entities_created=0,
                edges_created=0,
                pattern_updated=False,
                trust_updates=[],
                duplicate_of=duplicate_of,
                contradictions=contradictions,
                stored=False,
            )

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
            duplicate_of=None,
            contradictions=contradictions,
            stored=True,
        )

    async def store_facts_batch(
        self, requests: list[StoreFactRequest]
    ) -> BatchStoreResponse:
        results: list[StoreFactResponse] = []
        errors: list[dict[str, str]] = []
        stored = duplicates = failed = 0

        for request in requests:
            try:
                response = await self.store_fact(request)
                results.append(response)
                if response.duplicate_of:
                    duplicates += 1
                elif response.stored:
                    stored += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                errors.append(
                    {"claim_text": request.claim_text[:120], "error": str(e)}
                )

        return BatchStoreResponse(
            total=len(requests),
            stored=stored,
            duplicates=duplicates,
            failed=failed,
            results=results,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    async def recall(self, request: RecallRequest) -> RecallResponse:
        cached_verification: Optional[CachedVerification] = None
        fuzzy_cache_hits: list[CachedVerification] = []
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

        # Fuzzy cache: near-duplicate claims that were verified before
        if request.include_cache and request.fuzzy_cache:
            for fact in similar_facts:
                if fact.score >= self._settings.fuzzy_cache_threshold:
                    hit = await self.cache.get(
                        domain=request.domain or "general",
                        claim_text=fact.text,
                    )
                    if hit and hit.cache_key not in {
                        h.cache_key for h in fuzzy_cache_hits
                    }:
                        fuzzy_cache_hits.append(hit)

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

        reranked = False
        if request.rerank and similar_facts:
            similar_facts = await self._rerank_results(similar_facts, request.domain)
            reranked = True

        if request.min_similarity > 0:
            similar_facts = [
                f for f in similar_facts
                if f.score >= request.min_similarity
            ]

        return RecallResponse(
            cached_verification=cached_verification,
            fuzzy_cache_hits=fuzzy_cache_hits,
            similar_facts=similar_facts,
            related_entities=related_entities,
            relevant_patterns=relevant_patterns,
            source_trust=source_trust,
            reranked=reranked,
        )

    async def _rerank_results(
        self,
        results: list[VectorSearchResult],
        domain: Optional[str],
    ) -> list[VectorSearchResult]:
        """Blend vector similarity with source trust and recency."""
        reranked: list[VectorSearchResult] = []
        for result in results:
            trust_score = 0.5
            try:
                ts = result.metadata.get("timestamp")
                if ts:
                    age_days = (datetime.utcnow() - datetime.fromisoformat(ts)).total_seconds() / 86400
                    recency = max(0.0, 1.0 - age_days / 365.0)
                else:
                    recency = 0.5
            except Exception:
                recency = 0.5

            # Nearest similar sources' trust, if any
            if domain:
                domain_sources = await self.trust.get_domain_sources(
                    domain=domain, min_trust=0.0
                )
                if domain_sources:
                    trust_score = sum(
                        s.trust_score for s in domain_sources
                    ) / len(domain_sources)

            blended = (
                self._settings.rerank_alpha * result.score
                + self._settings.rerank_beta * trust_score
                + self._settings.rerank_gamma * recency
            )
            reranked.append(
                VectorSearchResult(
                    entry_id=result.entry_id,
                    text=result.text,
                    score=result.score,
                    metadata=result.metadata,
                    reranked_score=round(min(1.0, blended), 4),
                )
            )

        reranked.sort(key=lambda r: r.reranked_score or 0.0, reverse=True)
        return reranked

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
