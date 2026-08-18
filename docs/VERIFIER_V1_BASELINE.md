# HalluciGuard Verifier Agent — Baseline Audit (`VERIFIER_V1_BASELINE.md`)

**Date**: 2026-08-18  
**Repository Scope**: `agents/verifier_agent/`  
**Purpose**: Document the authoritative baseline of the Verifier Agent prior to Phase 2–13 stabilization.

---

## 1. Current Input Contract (`VerifierInputV2`)
Defined in [`agents/verifier_agent/schemas/models.py`](file:///c:/Users/S.Manjunath%20Reddy/OneDrive/Music/Pictures/Videos/HalluciGuard/agents/verifier_agent/schemas/models.py#L60-L64):
```python
class SuspiciousClaim(BaseModel):
    claim_id: str
    text: str

class VerifierInputV2(BaseModel):
    query_id: str
    domain: str  # general, healthcare, ai_research, finance, cybersecurity, legal
    suspicious_claims: list[SuspiciousClaim]
```

---

## 2. Current Output Contract (`VerifierOutputV2`)
Defined in [`agents/verifier_agent/schemas/models.py`](file:///c:/Users/S.Manjunath%20Reddy/OneDrive/Music/Pictures/Videos/HalluciGuard/agents/verifier_agent/schemas/models.py#L119-L132):
```python
class VerifierOutputV2(BaseModel):
    query_id: str
    domain: str
    domain_validated: bool
    retrieved_sources: int
    verified_sources: int
    claim_evidence: list[ClaimReport]
    overall_evidence_confidence: float
    latency_ms: int
    pipeline_stages: list[PipelineStageStatus]
    runtime_models: Optional[RuntimeModelInfo]
    cache_hit: bool
```

---

## 3. Current Verdict Enum (`VerdictLabel`)
Defined in [`agents/verifier_agent/schemas/models.py`](file:///c:/Users/S.Manjunath%20Reddy/OneDrive/Music/Pictures/Videos/HalluciGuard/agents/verifier_agent/schemas/models.py#L20-L24):
- `VERIFIED`
- `LIKELY_HALLUCINATED`
- `INSUFFICIENT_EVIDENCE`
- `MIXED_EVIDENCE`

*(Note: Target specification in Phase 3 requires updating this enum to `VERIFIED`, `CONTRADICTED`, `UNVERIFIED`, `CONFLICTED`).*

---

## 4. Current Domain Routing (`ModelRouter` & `AdapterRegistry`)
- `ModelRouter` ([`models/model_router.py`](file:///c:/Users/S.Manjunath%20Reddy/OneDrive/Music/Pictures/Videos/HalluciGuard/agents/verifier_agent/models/model_router.py)) maps incoming domains to internal model configs and adapter names.
- `AdapterRegistry.get_adapter(domain)` ([`adapters/registry.py`](file:///c:/Users/S.Manjunath%20Reddy/OneDrive/Music/Pictures/Videos/HalluciGuard/agents/verifier_agent/adapters/registry.py#L36-L50)) canonicalizes domain names and falls back silently to `"general"` (Wikipedia) if no registered adapter exists.

---

## 5. Current Adapters
- `GeneralAdapter` (`general`): Wikipedia Action API (`/w/api.php`) + REST API fallback (`/w/rest.php/v1/search/page`).
- `HealthcareAdapter` (`healthcare`): PubMed, PubMed Central, OpenFDA, ClinicalTrials.gov.
- `AiResearchAdapter` (`ai_research`): arXiv, Semantic Scholar, Crossref.
- `FinanceAdapter` (`finance`): SEC EDGAR API + AlphaVantage fallback.
- `CybersecurityAdapter` (`cybersecurity`): NVD CVE API + MITRE ATT&CK.
- `LegalGeneralAdapter` (`legal`): CourtListener API + RECAP archive.
- `DomainProxyAdapter`: Wraps domain profiles to delegates.

---

## 6. Current Retrieval Flow
1. Normalizes claim text with `ClaimNormalizer`.
2. Decomposes claim with rule-based `ClaimDecomposer`.
3. Expands query per domain with `QueryExpander`.
4. Invokes domain adapter `search(query, k=5)`.
5. Merges passages via `Aggregator` and ranks candidate passages with `HybridRetriever`.

---

## 7. Current Reranking Flow
- `Reranker` ([`rerankers/reranker.py`](file:///c:/Users/S.Manjunath%20Reddy/OneDrive/Music/Pictures/Videos/HalluciGuard/agents/verifier_agent/rerankers/reranker.py)) loads cross-encoder `BAAI/bge-reranker-large`.
- Computes relevance scores/logits between claim and retrieved passage snippets, sorting passages in descending order of relevance.

---

## 8. Current NLI Flow
- `NLIEngine` ([`nli/entailment.py`](file:///c:/Users/S.Manjunath%20Reddy/OneDrive/Music/Pictures/Videos/HalluciGuard/agents/verifier_agent/nli/entailment.py)) uses `cross-encoder/nli-deberta-v3-base`.
- Computes Softmax probabilities over `[contradiction, entailment, neutral]`.
- Maps top score to `EntailmentLabel`.

---

## 9. Current Scoring Flow
- `EvidenceScorer` ([`scorers/evidence_scorer.py`](file:///c:/Users/S.Manjunath%20Reddy/OneDrive/Music/Pictures/Videos/HalluciGuard/agents/verifier_agent/scorers/evidence_scorer.py)) combines NLI scores (`entailment`, `contradiction`), source credibility (`SourceReliabilityManager`), recency factor, and reranker relevance weight.
- Gates evidence with `MIN_NLI_SIGNAL = 0.45`.
- Derives `support_score`, `contradiction_score`, `trust_score`, and `verdict`.

---

## 10. Current Cache Behavior
- `SqliteCache` ([`cache/sqlite_cache.py`](file:///c:/Users/S.Manjunath%20Reddy/OneDrive/Music/Pictures/Videos/HalluciGuard/agents/verifier_agent/cache/sqlite_cache.py)) stores payloads in `verification_cache.db`.
- Uses SHA-256 versioned key `verifier-v2.1:{domain}:{query}`.
- Currently, cache invalidation can only be done manually; there is no global environment toggle `VERIFIER_CACHE_ENABLED=false`.

---

## 11. Current Claim Decomposition Behavior
- Rule-based `ClaimDecomposer` ([`claims/claim_decomposer.py`](file:///c:/Users/S.Manjunath%20Reddy/OneDrive/Music/Pictures/Videos/HalluciGuard/agents/verifier_agent/claims/claim_decomposer.py)) splits multi-sentence text into individual claims via regex and clause rules.

---

## 12. Known Inconsistencies & Issues Discovered
1. **Silent Fallback**: `AdapterRegistry.get_adapter()` falls back silently to `general` (Wikipedia) for unhandled domains.
2. **Missing Source Traceability**: `VerifierOutputV2` does not log individual source attempt/success/fail telemetry (`sources_attempted`, `sources_succeeded`, `sources_failed`).
3. **Irrelevant NLI Contradiction Leakage**: Unrelated passages (e.g., *Government cheese* or *Paris FC*) are assigned high NLI contradiction probabilities by DeBERTa and shift the verdict toward `LIKELY_HALLUCINATED` or `MIXED_EVIDENCE`.
4. **Descriptive Myth Entailment**: Snippets quoting myths (e.g., *"The Moon is made of green cheese" is a fanciful belief...*) yield 99.2% NLI Entailment because the NLI model matches literal sub-clauses in quotes.
5. **No Minimum Relevance Gate Before NLI**: Passages with low reranker relevance scores still pass to NLI and pollute evidence scoring.
6. **No Cache Toggle**: No `VERIFIER_CACHE_ENABLED` environment variable exists to bypass sqlite cache cleanly during dev/testing.

---

## 13. What is Actually Verified vs. What is NOT Verified
- **Actually Verified**: Passage retrieval from live APIs (Wikipedia, PubMed, OpenFDA, arXiv, SEC EDGAR), cross-encoder reranking, DeBERTa NLI probability generation, and evidence-weighted trust scoring.
- **NOT Verified**: Semantic distinction between factual assertion vs. myth descriptions, irrelevance-gated NLI contradiction filtering, source policy isolation without fallback.
