from __future__ import annotations
from typing import Optional, List, Dict, Any
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
    sources_attempted: List[str] = Field(default_factory=list)
    sources_succeeded: List[str] = Field(default_factory=list)
    sources_failed: List[str] = Field(default_factory=list)
    gate_method: str = ""  # "bge_gate" | "lexical_fallback"
    gate_bge_scores: List[float] = Field(default_factory=list)
    disambiguation_filtered: int = 0


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
    duplicates_removed: int = 0


class FinalTrace(BaseModel):
    """Trace of final reranking output."""
    reranked_count: int = 0
    decision_grade_count: int = 0


class ModelExecutionTrace(BaseModel):
    """§26 model-execution proof block.

    Records whether a model (BGE reranker / DeBERTa NLI) actually loaded and ran a
    real forward pass on the most recent call. This is the anti-masquerade signal:
    a fail-soft fallback sets ``inference_executed=False`` and ``degraded=True`` so
    it can never be presented as genuine model output, and certification mode can
    fail-closed on it (§13/§15/§28).
    """
    component: str = ""            # "bge_reranker" | "deberta_nli"
    model: str = ""
    loaded: bool = False
    inference_executed: bool = False
    degraded: bool = False
    status: str = "not_run"        # executed | degraded | unavailable | not_run
    device: str = "unknown"
    latency_ms: int = 0
    scored_count: int = 0          # reranker: passages scored
    batch_size: int = 0            # nli: pairs classified


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
    source_confidence_hint: float = 0.0
    relation_check_result: Optional[str] = None


class RelationCheckTrace(BaseModel):
    """Trace of structured relation verification.

    NOTE: fields are the flat triple components actually populated by
    ``api/pipeline.py`` and read by ``scripts/verify_claim.py``. The previous
    schema (claim_triple/evidence_triples/comparison_result) did not match the
    producer, so Pydantic silently dropped the data and this trace was always
    empty — a real observability bug (see regression test).
    """
    claim_subject: str = ""
    claim_relation: str = ""
    claim_object: str = ""
    evidence_subject: str = ""
    evidence_relation: str = ""
    evidence_object: str = ""
    check_result: str = "NO_TRIPLE_EXTRACTED"  # MATCH / OBJECT_MISMATCH / RELATION_MISMATCH / NO_TRIPLE_EXTRACTED
    combination_rule_applied: str = ""
    details: Optional[str] = None


class GateRelevanceAuditTrace(BaseModel):
    """Trace comparing gate-time heuristics with final BGE relevance scores."""
    source_confidence_hint: float = 0.0
    gate_time_relevance_signal: float = 0.0
    final_bge_relevance_score: float = 0.0
    signals_agree: bool = True


class N8NTrace(BaseModel):
    """Trace of n8n Retrieval Service V2 execution."""
    called: bool = False
    workflow_version: str = "2.0.0"
    request_id: str = ""
    latency_ms: int = 0
    retrieval_mode: str = "hybrid"
    primary_sources: List[str] = Field(default_factory=list)
    tavily_called: bool = False
    evidence_count: int = 0
    counts: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    primary_trace: Optional[Dict[str, Any]] = None
    tavily_trace: Optional[Dict[str, Any]] = None
    stage_traces: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    legacy_n8n_output: Optional[Dict[str, Any]] = None



class RetrievalTrace(BaseModel):
    """Complete retrieval trace for a single claim verification."""
    requested_domain: str = ""
    primary_adapter: str = ""
    retrieval_mode: str = "hybrid"  # hybrid / primary_only / tavily_only
    generated_queries: List[str] = Field(default_factory=list)
    primary: PrimaryRetrievalTrace = Field(default_factory=PrimaryRetrievalTrace)
    tavily: TavilyRetrievalTrace = Field(default_factory=TavilyRetrievalTrace)
    merged: MergedTrace = Field(default_factory=MergedTrace)
    final: FinalTrace = Field(default_factory=FinalTrace)
    evidence_details: List[EvidencePassageTrace] = Field(default_factory=list)
    relation_check: Optional[RelationCheckTrace] = None
    gate_relevance_audit: Optional[GateRelevanceAuditTrace] = None
    n8n_trace: Optional[N8NTrace] = None
    # §26 model-execution proof (populated from reranker/nli .diagnostics()):
    reranker_execution: Optional[ModelExecutionTrace] = None
    nli_execution: Optional[ModelExecutionTrace] = None
    timings: dict = Field(default_factory=dict)

