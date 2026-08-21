from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


class PrimaryRetrievalTrace(BaseModel):
    """Trace of the primary domain adapter retrieval."""
    called: bool = False
    result_count: int = 0
    usable_count: int = 0
    relevant_count: int = 0
    top_relevance: float = 0.0
    source_diversity: int = 0
    sufficient: bool = False
    reason: str = ""
    latency_ms: int = 0
    error: Optional[str] = None


class TavilyRetrievalTrace(BaseModel):
    """Trace of Tavily web retrieval fallback."""
    called: bool = False
    reason: str = ""  # PRIMARY_EVIDENCE_SUFFICIENT / INSUFFICIENT_RELEVANCE / PRIMARY_EMPTY / PRIMARY_ERROR / FORCE_TAVILY
    query: str = ""
    search_depth: str = "advanced"
    result_count: int = 0
    extracted_count: int = 0
    usable_count: int = 0
    latency_ms: int = 0
    error: Optional[str] = None
    domains_requested: List[str] = Field(default_factory=list)


class MergedTrace(BaseModel):
    """Trace of primary + Tavily merge/dedup."""
    candidate_count: int = 0
    deduplicated_count: int = 0


class FinalTrace(BaseModel):
    """Trace of final reranking output."""
    reranked_count: int = 0
    decision_grade_count: int = 0


class EvidencePassageTrace(BaseModel):
    """Per-passage diagnostic trace."""
    title: str = ""
    url: str = ""
    source: str = ""
    retrieval_method: str = ""  # "primary" or "tavily_web"
    relevance_score: float = 0.0
    relevance_decision: str = ""  # "relevant" / "irrelevant"
    nli_label: str = ""
    entailment: float = 0.0
    contradiction: float = 0.0
    neutral: float = 0.0
    nli_degraded: bool = False
    evidence_class: str = ""  # SUPPORTING / CONTRADICTING / NEUTRAL / IRRELEVANT
    source_credibility: float = 0.0
    effective_weight: float = 0.0


class RetrievalTrace(BaseModel):
    """Complete retrieval trace for a single claim verification."""
    requested_domain: str = ""
    primary_adapter: str = ""
    retrieval_mode: str = "hybrid"  # hybrid / primary_only / tavily_only
    primary: PrimaryRetrievalTrace = Field(default_factory=PrimaryRetrievalTrace)
    tavily: TavilyRetrievalTrace = Field(default_factory=TavilyRetrievalTrace)
    merged: MergedTrace = Field(default_factory=MergedTrace)
    final: FinalTrace = Field(default_factory=FinalTrace)
    evidence_details: List[EvidencePassageTrace] = Field(default_factory=list)
    timings: dict = Field(default_factory=dict)
