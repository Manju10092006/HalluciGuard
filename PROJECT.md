# Project: HalluciGuard Verifier

## Architecture
HalluciGuard Verifier is an evidence-grounded, multi-source claim verification system that determines whether factual claims are VERIFIED, CONTRADICTED, CONFLICTED, or UNVERIFIED strictly from real external evidence using BGE reranking, DeBERTa NLI, domain-specific adapters, and Tavily web fallback.

### System Topography & Control Flow
```
User / API Client
       │
       ▼
[Supervisor FastAPI (orchestration/api.py) / Gradio Blocks (app.py)]
       │
       ▼
[LangGraph Supervisor Workflow (orchestration/graph.py)]
       │  StateGraph(HalluciGuardState)
       ├──► 1. Base LLM / Generator
       ├──► 2. Detector Agent (hallucination detection & claim extraction)
       ├──► 3. Verifier Agent (9-stage deterministic verification pipeline)
       └──► 4. Memory Agent (session state, cache & knowledge graph)
```

### 9-Stage Verifier Pipeline (`agents/verifier_agent/api/pipeline.py`)
1. **Stage 1 — Domain Validation & Taxonomy Routing**: Classify claim domain across 30+ domains into 6 core adapters (Healthcare, Cybersecurity, Finance, AI Research, Legal, General) via `DomainValidator`, `ModelRouter`, and `DomainIntelligenceRegistry`.
2. **Stage 2 — Semantic Cache Check**: SQLite persistent caching (`SqliteCache`) with query normalization to eliminate redundant model and network evaluations.
3. **Stage 3 — Claim Normalization & Decomposition**: Normalize text, extract atomic sub-claims (`ClaimDecomposer`), and resolve coreferences.
4. **Stage 4 — Query Expansion & Relation Extraction**: Extract Subject-Verb-Object (SVO) triples (`SVOExtractor`), named entities (`EntityResolver`), and generate predicate/entity query expansions (`QueryExpander`).
5. **Stage 5 — Quality-Gated Multi-Source Retrieval**: Query primary domain endpoints (PubMed, OpenFDA, WHO, NVD, SEC, arXiv, CourtListener, Wikipedia). Assess primary evidence quality (usable passage count, term coverage $\ge 50\%$, top relevance $\ge 0.30$). Only if primary evidence is empty/insufficient, trigger Tavily web search fallback (`WebEnhancedAdapter`).
6. **Stage 6 — Evidence Aggregation & Hybrid Retrieval**: Aggregate passages, canonicalize URLs, deduplicate by token Jaccard similarity (>85%), and perform Hybrid Retrieval (BM25 + FAISS Dense BGE-M3 embeddings via Monotonic Reciprocal Rank Fusion + Lexical coverage).
7. **Stage 7 — Cross-Encoder Reranking**: Re-score aggregated candidate passages against the claim using `BAAI/bge-reranker-large`.
8. **Stage 8 — NLI Inference & SVO Relation Verification**: Evaluate entailment/contradiction/neutral probabilities using `cross-encoder/nli-deberta-v3-base` (Premise=evidence, Hypothesis=claim). Execute deterministic SVO triple verification (`RelationVerifier`) to override false neutral/entailment signals on entity/relation swaps (`BYPASS_SUPPRESSION_FORCE_CONTRADICTION`).
9. **Stage 9 — Calibration, Conflict Resolution & Verdict Formatting**: 
   - Strict 4-class mutually exclusive evidence categorization (`SUPPORTING`, `CONTRADICTING`, `NEUTRAL`, `IRRELEVANT` where $\sum = \text{Total Passages}$, relevance threshold $= 0.20$).
   - Conflict resolution (`ConflictResolver`) and public verdict enum assignment (`VERIFIED`, `CONTRADICTED`, `CONFLICTED`, `UNVERIFIED`).
   - Honest confidence scoring in $[0, 1]$ based on primary evidence strength, consensus ratio, and source density.
   - Per-canonical-URL score capping to prevent multi-chunk score inflation.
   - Structured citation formatting (`CitationFormatter`) and full telemetry tracing (`RetrievalTrace`).

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F0.1 | Architecture Audit | Full mapping of 9-stage pipeline, LangGraph supervisor, and singleton ModelManager | M0 | Survey |
| F0.2 | Baseline Benchmark Freeze | Freeze baseline metrics across 35 multi-domain claims into `baseline_phase0_frozen.json` | M0 | Survey |
| F0.3 | Pre-flight Readiness Verification | Verify deployment checks, dependency matrix, and deterministic test baseline | M0 | Survey |
| F1.1 | Quality-Gated Web Fallback | Evaluate primary passage count, relevance ($\ge 0.30$), and term coverage ($\ge 50\%$) before triggering Tavily | M1 | ORIGINAL_REQUEST §R2 |
| F1.2 | SVO Predicate Query Expansion | Expand queries using predicate aliases, entity variants, and domain synonyms | M1 | ORIGINAL_REQUEST §R2 |
| F1.3 | URL Canonicalization & Dedup | Strip URL fragments/tracking params, token Jaccard deduplication (>85%), per-URL scoring caps | M1 | ORIGINAL_REQUEST §R2 |
| F1.4 | Request-Local Telemetry Tracing | `RetrievalContext` contextvars tracking attempted/succeeded/failed sources and complete `RetrievalTrace` logging | M1 | ORIGINAL_REQUEST §R2 |
| F2.1 | Mutually Exclusive Evidence Semantics | 4-way partition into `SUPPORTING`, `CONTRADICTING`, `NEUTRAL`, `IRRELEVANT` with sum invariant | M2 | ORIGINAL_REQUEST §R3 |
| F2.2 | BGE Relevance Gating | Threshold at 0.20 to filter out spurious/unrelated evidence before NLI scoring | M2 | ORIGINAL_REQUEST §R3 |
| F2.3 | Contradiction Handling & False Suppression Elimination | Object mismatch & SVO contradiction overrides false neutral; 0% false verified rate on negative claims | M2 | ORIGINAL_REQUEST §R3 |
| F2.4 | Calibrated Honest Confidence | Mathematically sound confidence in $[0, 1]$ independent of trust score, yielding valid confidence for both verified and contradicted claims | M2 | ORIGINAL_REQUEST §R3 |
| F2.5 | Canonical Public Verdict Enum | Strict adherence to public verdict enum `[VERIFIED, CONTRADICTED, CONFLICTED, UNVERIFIED]` | M2 | ORIGINAL_REQUEST §R3 |
| F3.1 | WHO GHO OData API Adapter | Query structured WHO Global Health Observatory data for health/epidemiology claims | M3 | ORIGINAL_REQUEST §R4 |
| F3.2 | PubMed & PMC Adapters | E-utilities search + XML abstract parsing + full-text paragraph extraction with PMID provenance | M3 | ORIGINAL_REQUEST §R4 |
| F3.3 | OpenFDA Drug & Event Adapter | Query OpenFDA label endpoint for active ingredients, indications, and boxed warnings | M3 | ORIGINAL_REQUEST §R4 |
| F3.4 | Authoritative Domain Routing | Dynamic classification and routing across 30+ domains to 6 core adapters with credibility weights | M3 | ORIGINAL_REQUEST §R4 |
| F4.1 | E2E Multi-Domain Benchmark Pass | Execute 35-claim benchmark and external datasets, verifying 0% false verified rate and accuracy improvements over Phase 0 baseline | M4 | ORIGINAL_REQUEST §Acceptance Criteria |
| F4.2 | 100% Deterministic Test Pass | Ensure all unit, invariant, and contract tests in `verifier_agent`, `memory_agent`, and `judge_agent` pass 100% | M4 | ORIGINAL_REQUEST §Acceptance Criteria |
| F4.3 | Coverage Hardening & Telemetry Audit | Adversarial verification of edge cases, entity swaps, myth debunking, and empty evidence handling | M4 | ORIGINAL_REQUEST §Acceptance Criteria |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M0 | Phase 0: Baseline Freeze & Audit | Audit architecture, map data/control flow, freeze baseline metrics in `baseline_phase0_frozen.json` | none | DONE |
| M1 | Phase 1 (V1.1): Quality-Gated Retrieval & Fallback | Quality-gated Tavily fallback, SVO query expansion, URL dedup/canonicalization, request telemetry | M0 | DONE |
| M2 | Phase 2 (V1.2): Evidence Semantics & Calibrated Confidence | 4-class mutually exclusive categorization, BGE relevance gate (0.20), contradiction preservation, honest confidence in $[0, 1]$, canonical verdict enum | M1 | DONE |
| M3 | Phase 3 (V1.3): Domain Adapters & Structured Evidence | WHO GHO OData API, PubMed/PMC XML parsing with PMID, OpenFDA drug labels, domain intelligence routing | M2 | DONE |
| M4 | Phase 4: E2E Benchmark Verification & Hardening | Multi-domain 35-claim benchmark execution, 100% test pass verification across all suites, 0% false verified rate confirmation | M3 | DONE |

---

## Interface Contracts

### 1. Verifier API Contract (`schemas/models.py`)
```python
class VerdictLabel(str, Enum):
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    CONFLICTED = "conflicted"
    UNVERIFIED = "unverified"

class EvidenceClassification(str, Enum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"
    IRRELEVANT = "irrelevant"

class VerifierInputV2(BaseModel):
    claim: str
    domain: Optional[str] = None
    context: Optional[str] = None
    use_cache: bool = True
    min_confidence: float = 0.50

class VerifierOutputV2(BaseModel):
    claim: str
    verdict: VerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    trust_score: float = Field(ge=0.0, le=1.0)
    evidence: List[EvidenceItem]
    classification_counts: Dict[str, int]
    explanation: str
    domain: str
    retrieval_trace: Optional[RetrievalTrace] = None
```

### 2. Evidence Semantics Invariant
For every verification output:
$$\text{classification\_counts["supporting"]} + \text{classification\_counts["contradicting"]} + \text{classification\_counts["neutral"]} + \text{classification\_counts["irrelevant"]} == \text{len(passages)}$$
- If $\max(\text{relevance\_score}, \text{source\_confidence\_hint}) < 0.20 \implies \text{IRRELEVANT}$
- Contradicted claims with primary evidence strength $\ge 0.25$ must have calibrated confidence $\ge 0.50$ (never suppressed to 0).

### 3. Retrieval & Fallback Contract (`adapters/web_enhanced.py`)
- Primary adapter search invoked first.
- Primary evidence evaluated: `usable_count >= 2` AND `relevant_count >= 1` AND `top_relevance >= 0.30` AND `claim_term_coverage >= 0.50`.
- If satisfied: Tavily web search is skipped (`trace.tavily.called = False`, reason = `PRIMARY_EVIDENCE_SUFFICIENT`).
- If insufficient/empty: Tavily search/extract is invoked (`trace.tavily.called = True`, reason = `PRIMARY_EVIDENCE_INSUFFICIENT` / `PRIMARY_EMPTY`).
- Tavily LLM-generated answers (`include_answer`) must NEVER be used as evidence.

---

## Code Layout
- `app.py`: Gradio UI + FastAPI web application entry point.
- `run_server.py`: Standalone Verifier service entry point.
- `orchestration/`: LangGraph supervisor state machine, API endpoints, graph contracts.
- `agents/verifier_agent/`:
  - `adapters/`: Domain adapters (`healthcare.py`, `cybersecurity.py`, `finance.py`, `ai_research.py`, `legal_general.py`, `general.py`, `web_enhanced.py`, `web_retriever.py`, `proxy.py`).
  - `api/`: Verifier pipeline (`pipeline.py`, `main.py`).
  - `scorers/`: `evidence_scorer.py`, `conflict_resolver.py`, `nli_engine.py`, `relation_verifier.py`.
  - `retrievers/`: `hybrid.py`, `bge_reranker.py`.
  - `aggregation/`: `duplicate_remover.py`.
  - `formatters/`: `citation_formatter.py`, `explanation_generator.py`.
  - `schemas/`: `models.py`, `retrieval_trace.py`, `passage.py`.
  - `models/`: `model_manager.py`, `domain_intelligence.py`.
  - `benchmarks/`: `runner.py`, `metrics.py`, `baseline_phase0_frozen.json`.
  - `tests/`: 20 unit/contract/hardening test files.
- `agents/memory_agent/`: Memory agent, caching, knowledge graph, trust scoring, and tests.
- `agents/judge_agent/`: Benchmark verification scenarios and contract tests.
- `scripts/`: `benchmark_eval.py`, `deployment_readiness_check.py`, `verify_claim.py`.
