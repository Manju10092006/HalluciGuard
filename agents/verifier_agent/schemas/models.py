"""
HalluciGuard Verifier Agent — Pydantic V2 Data Schemas.

Defines the complete data contract for inputs, outputs, evidence items,
pipeline stages, adapter metadata, and domain statistics.
"""
from __future__ import annotations
import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class EntailmentLabel(str, Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class VerdictLabel(str, Enum):
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNVERIFIED = "unverified"
    CONFLICTED = "conflicted"


class PipelineStage(str, Enum):
    DOMAIN_VALIDATION = "domain_validation"
    CLAIM_DECOMPOSITION = "claim_decomposition"
    QUERY_EXPANSION = "query_expansion"
    RETRIEVAL = "retrieval"
    AGGREGATION = "aggregation"
    RERANKING = "reranking"
    NLI = "nli"
    SCORING = "scoring"
    FORMATTING = "formatting"


class Passage(BaseModel):
    model_config = ConfigDict(frozen=True)
    title: str
    source: str
    url: str
    publication_date: str
    snippet: str
    source_id: str = ""
    relevance_score: float = 0.0


class SuspiciousClaim(BaseModel):
    claim_id: str
    text: str


class DecomposedClaim(BaseModel):
    original_text: str
    sub_claims: list[str]


class VerifierInputV2(BaseModel):
    query_id: str
    domain: str
    suspicious_claims: list[SuspiciousClaim]


class EvidenceItem(BaseModel):
    title: str
    source: str
    url: str
    publication_date: str
    snippet: str
    entailment_label: EntailmentLabel
    entailment_score: float = Field(ge=0.0, le=1.0)
    credibility_score: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Runtime Model Metadata — included in every enterprise response
# ---------------------------------------------------------------------------
class RuntimeModelInfo(BaseModel):
    """Describes which models were selected and executed during verification."""
    embedding_model: str = ""
    reranker_model: str = ""
    nli_model: str = ""
    cross_encoder: str = ""
    classification_model: str = ""
    retrieval_strategy: str = "hybrid_rrf"
    device: str = "cpu"
    claim_complexity: str = "standard"
    latency_budget: str = "balanced"
    routing_reason: str = ""


class ClaimReport(BaseModel):
    claim_id: str
    claim_text: str
    evidence: list[EvidenceItem]
    support_score: float = Field(ge=0.0, le=1.0)
    contradiction_score: float = Field(ge=0.0, le=1.0)
    trust_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    verdict: VerdictLabel
    explanation: str = ""
    # Enhanced fields for enterprise responses
    supporting_sources: list[str] = []
    contradicting_sources: list[str] = []
    retrieved_documents: int = 0
    reranked_documents: int = 0
    verified_evidence: int = 0


class PipelineStageStatus(BaseModel):
    stage: PipelineStage
    status: str  # "completed", "running", "failed", "skipped"
    duration_ms: int = 0
    details: str = ""


class VerifierOutputV2(BaseModel):
    query_id: str
    domain: str
    domain_validated: bool
    adapter: str = "general"
    sources_attempted: list[str] = []
    sources_succeeded: list[str] = []
    sources_failed: list[str] = []
    retrieved_sources: int
    verified_sources: int
    claim_evidence: list[ClaimReport]
    overall_evidence_confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: int
    pipeline_stages: list[PipelineStageStatus] = []
    # Enterprise-grade response metadata
    runtime_models: Optional[RuntimeModelInfo] = None
    cache_hit: bool = False


class AdapterMetadata(BaseModel):
    name: str
    version: str = "1.0.0"
    supported_domains: list[str] = []
    supported_languages: list[str] = ["en"]
    supports_live_search: bool = True
    cacheable: bool = True
    priority: int = 1
    max_results: int = 10
    is_stub: bool = False


class AdapterHealthStatus(BaseModel):
    adapter_name: str
    is_healthy: bool
    last_check: str = ""
    response_time_ms: int = 0
    error: str = ""


class DomainStatistics(BaseModel):
    domain: str
    sources: list[str]
    status: str  # "fully_implemented", "stub", "mock"
    credibility_scores: dict[str, float] = {}
    is_implemented: bool = True
    adapter_health: list[AdapterHealthStatus] = []
