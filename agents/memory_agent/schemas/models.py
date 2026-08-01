from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    EVENT = "event"
    DATE = "date"
    LOCATION = "location"
    SOURCE = "source"
    CLAIM = "claim"
    FACT = "fact"


class RelationType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    MENTIONS = "mentions"
    DERIVED_FROM = "derived_from"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    OCCURRED_ON = "occurred_on"
    AUTHORED_BY = "authored_by"


class PatternType(str, Enum):
    TEMPORAL = "temporal"
    NUMERICAL = "numerical"
    ENTITY = "entity"
    CAUSAL = "causal"
    STATISTICAL = "statistical"
    DEFINITION = "definition"


class TrustChangeReason(str, Enum):
    VERIFIED_CORRECT = "verified_correct"
    VERIFIED_INCORRECT = "verified_incorrect"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    EXPLICIT_FEEDBACK = "explicit_feedback"


# ---------------------------------------------------------------------------
# Knowledge Graph Models
# ---------------------------------------------------------------------------

class EntityNode(BaseModel):
    model_config = ConfigDict(frozen=False)

    entity_id: str
    entity_type: EntityType
    name: str
    canonical_name: str
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class Edge(BaseModel):
    model_config = ConfigDict(frozen=False)

    source_id: str
    target_id: str
    relation: RelationType
    weight: float = Field(ge=0.0, le=1.0, default=1.0)
    properties: dict[str, Any] = Field(default_factory=dict)
    evidence_count: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_verified_at: Optional[datetime] = None


class KnowledgeGraphStats(BaseModel):
    total_nodes: int
    total_edges: int
    entity_type_counts: dict[str, int]
    relation_type_counts: dict[str, int]
    avg_edge_weight: float
    domains_covered: list[str]


class EntityImportance(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_id: str
    name: str
    entity_type: EntityType
    page_rank: float
    degree: int
    betweenness: float


class GraphAnalytics(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_nodes: int
    total_edges: int
    connected_components: int
    most_important: list[EntityImportance] = Field(default_factory=list)
    communities: int
    top_communities: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Verification Cache Models
# ---------------------------------------------------------------------------

class CachedVerification(BaseModel):
    model_config = ConfigDict(frozen=True)

    cache_key: str
    domain: str
    claim_text: str
    verdict: str
    evidence_summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_count: int
    created_at: datetime
    expires_at: datetime


class CacheStats(BaseModel):
    total_entries: int
    total_hits: int
    total_misses: int
    hit_rate: float
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]
    domain_breakdown: dict[str, int]


# ---------------------------------------------------------------------------
# Pattern Learning Models
# ---------------------------------------------------------------------------

class HallucinationPattern(BaseModel):
    model_config = ConfigDict(frozen=False)

    pattern_id: str
    pattern_type: PatternType
    domain: str
    description: str
    examples: list[str] = Field(default_factory=list)
    frequency: int = 0
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    keywords: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen_at: Optional[datetime] = None


class PatternMatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    pattern_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    claim_text: str
    matched_keywords: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Source Trust Models
# ---------------------------------------------------------------------------

class SourceTrustRecord(BaseModel):
    model_config = ConfigDict(frozen=False)

    source_id: str
    source_name: str
    domain: str
    trust_score: float = Field(ge=0.0, le=1.0, default=0.5)
    total_verifications: int = 0
    correct_count: int = 0
    incorrect_count: int = 0
    conflict_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TrustUpdate(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    old_score: float = Field(ge=0.0, le=1.0)
    new_score: float = Field(ge=0.0, le=1.0)
    reason: TrustChangeReason
    delta: float


# ---------------------------------------------------------------------------
# Vector Store Models
# ---------------------------------------------------------------------------

class VectorEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    entry_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None


class VectorSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    entry_id: str
    text: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    reranked_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Memory Agent API Models
# ---------------------------------------------------------------------------

class ContradictionAlert(BaseModel):
    model_config = ConfigDict(frozen=True)

    existing_fact_id: str
    existing_claim_text: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    existing_verdict: str
    reason: str


class StoreFactRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_text: str
    domain: str
    verdict: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoreFactResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    fact_id: str
    entities_created: int
    edges_created: int
    pattern_updated: bool
    trust_updates: list[TrustUpdate] = Field(default_factory=list)
    duplicate_of: Optional[str] = None
    contradictions: list[ContradictionAlert] = Field(default_factory=list)
    stored: bool = True


class BatchStoreResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int
    stored: int
    duplicates: int
    failed: int
    results: list[StoreFactResponse] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)


class RecallRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    domain: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)
    include_graph_context: bool = True
    include_cache: bool = True
    include_patterns: bool = True
    fuzzy_cache: bool = True
    rerank: bool = True
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class RecallResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    cached_verification: Optional[CachedVerification] = None
    fuzzy_cache_hits: list[CachedVerification] = Field(default_factory=list)
    similar_facts: list[VectorSearchResult] = Field(default_factory=list)
    related_entities: list[EntityNode] = Field(default_factory=list)
    relevant_patterns: list[HallucinationPattern] = Field(default_factory=list)
    source_trust: list[SourceTrustRecord] = Field(default_factory=list)
    reranked: bool = False


class MemoryStatsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    knowledge_graph: KnowledgeGraphStats
    cache: CacheStats
    patterns_total: int
    sources_tracked: int
    vector_store_size: int


class TrustUpdateRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    source_name: str
    domain: str
    reason: TrustChangeReason
    evidence_count: int = 1


class PatternQueryRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    domain: Optional[str] = None
    pattern_type: Optional[PatternType] = None
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=1, le=100)


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    version: str
    components: dict[str, str]
    stats: Optional[MemoryStatsResponse] = None
