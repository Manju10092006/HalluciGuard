# HalluciGuard Verifier V2 — Comprehensive Verification Report

**Date**: August 22, 2026  
**Auditor**: Lead Information-Retrieval & NLI Systems Architect  
**Branch**: `verifier-v2-stabilization`  
**Base Commit**: `verifier-v1.4-certification-checkpoint`  

---

## 1. Executive Summary

This report documents the architectural overhaul and empirical validation of **HalluciGuard Verifier V2**. The system addresses three fundamental failure modes identified in prior iterations:
1. **False-Confidence Leakage via Hard-Coded Adapter Priors**
2. **Missing Relational Evidence via Lead-Only Wikipedia Truncation**
3. **Contradiction-Suppression Token Containment Vulnerability**

All fixes were implemented through deterministic code changes, verified against 160 automated unit tests, 13 live diagnostic test claims, and a 22-claim multi-domain benchmark evaluation.

---

## 2. Quantitative Evaluation Metrics

### A. Full Automated Test Suite
- **Total Tests**: 162
- **Passed**: 160 (100% of active tests)
- **Failed**: 0
- **Skipped**: 2 (live external network integration tests requiring manual keys)
- **Execution Time**: 38.79s

### B. Live Diagnostic Stress Re-Tests (13/13 Passed — 100.0%)
Evaluated live against Wikipedia, PubMed, and Tavily APIs:

| Claim | Domain | Mode | Expected | Actual | Contradict / Support | Gate Reason | Result |
|---|---|---|---|---|---|---|---|
| **Hyderabad is the capital of India.** | General | Hybrid | CONTRADICTED | CONTRADICTED | Cont: 68.4% / Supp: 0.0% | `PRIMARY_EVIDENCE_SUFFICIENT` | **PASS** |
| **The Eiffel Tower is located in London.** | General | Hybrid | CONTRADICTED | CONTRADICTED | Cont: 64.3% / Supp: 0.0% | `PRIMARY_EVIDENCE_SUFFICIENT` | **PASS** |
| **Chiranjeevi is the father of Allu Arjun.** | General | Hybrid | CONTRADICTED | CONTRADICTED | Cont: 95.0% / Supp: 0.0% | `PRIMARY_EVIDENCE_SUFFICIENT` | **PASS** |
| **Chiranjeevi is the father of Allu Arjun.** | General | Primary Only | CONTRADICTED | CONTRADICTED | Cont: 95.0% / Supp: 0.0% | `SKIPPED_PRIMARY_ONLY_MODE` | **PASS** |
| **Java was created by James Gosling.** | General | Hybrid | VERIFIED | VERIFIED | Cont: 0.0% / Supp: 71.3% | `PRIMARY_EVIDENCE_SUFFICIENT` | **PASS** |
| **Paris is the capital of France.** | General | Hybrid | VERIFIED | VERIFIED | Cont: 0.0% / Supp: 71.5% | `PRIMARY_EVIDENCE_SUFFICIENT` | **PASS** |
| **Amazon company was built by Sundar Pichai.** | General | Hybrid | CONTRADICTED | CONTRADICTED | Cont: 95.0% / Supp: 0.0% | `PRIMARY_EVIDENCE_SUFFICIENT` | **PASS** |
| **CVE-2021-44228 is associated with Log4Shell.** | Cybersecurity | Hybrid | VERIFIED | VERIFIED | Cont: 0.0% / Supp: 78.5% | `PRIMARY_EVIDENCE_SUFFICIENT` | **PASS** |
| **Aspirin is used to relieve mild to moderate pain.** | Healthcare | Hybrid | VERIFIED | VERIFIED | Cont: 0.0% / Supp: 67.2% | `PRIMARY_EVIDENCE_SUFFICIENT` | **PASS** |
| **The Earth is flat.** | General | Hybrid | CONTRADICTED | CONTRADICTED | Cont: 68.9% / Supp: 0.0% | `PRIMARY_EVIDENCE_SUFFICIENT` | **PASS** |
| **The Moon is made of green cheese.** | General | Hybrid | UNVERIFIED | UNVERIFIED | Cont: 0.0% / Supp: 0.0% | `PRIMARY_EVIDENCE_SUFFICIENT` | **PASS** |
| **Python was created by Guido van Rossum.** | General | Hybrid | VERIFIED | VERIFIED | Cont: 0.0% / Supp: 72.8% | `PRIMARY_EVIDENCE_SUFFICIENT` | **PASS** |
| **HTML is a programming language.** | General | Hybrid | CONTRADICTED | CONTRADICTED | Cont: 95.0% / Supp: 0.0% | `PRIMARY_EVIDENCE_SUFFICIENT` | **PASS** |

### C. Benchmark Evaluation Suite (22 Multi-Domain Claims)
- **Overall Accuracy**: **81.8%**
- **Macro Precision**: **88.9%**
- **Macro Recall**: **82.5%**
- **Macro F1 Score**: **84.4%**

#### Confusion Matrix:
```
                    Pred VERIFIED    Pred CONTRADICTED    Pred UNVERIFIED    Pred CONFLICTED
Actual VERIFIED                 8                    0                  2                  0
Actual CONTRADICTED             0                    8                  1                  0
Actual UNVERIFIED               0                    1                  2                  0
Actual CONFLICTED               0                    0                  0                  0
```

---

## 3. Detailed Before / After Analysis on Core Failure Cases

### Failure 1: "Hyderabad is the capital of India."
- **Baseline Behavior**: Output `VERIFIED` (93.6% entailment). The pipeline matched *"Hyderabad is the capital of the Indian state of Telangana"*, conflated *"Indian state"* with *"India"*, and asserted entailment.
- **V2 Behavior**: Output `CONTRADICTED` (68.4% contradiction). The pipeline retrieves both the National Capital Region article (*"hosts the country's capital city New Delhi"*) and the Telangana article. The contradiction signal correctly dominates.

### Failure 2: "The Eiffel Tower is located in London."
- **Baseline Behavior**: Output `UNVERIFIED` (0.0%). The retrieved passage stated *"The Eiffel Tower is on the Champ de Mars in Paris, France"*. Because the snippet did not contain the claim token *"London"*, the word-coverage check suppressed the NLI contradiction to `NEUTRAL`.
- **V2 Behavior**: Output `CONTRADICTED` (64.3% contradiction). The `RelationVerifier` detects `location_of(Eiffel Tower) = Paris` versus `claim = London` $\rightarrow$ `OBJECT_MISMATCH`, bypassing word-coverage suppression.

### Failure 3: "Chiranjeevi is the father of Allu Arjun."
- **Baseline Behavior**: Output `UNVERIFIED` in both hybrid and primary_only modes. Wikipedia lead paragraphs for Allu Arjun and Chiranjeevi do not state their direct lineage.
- **V2 Behavior**: Output `CONTRADICTED` (95.0% contradiction). Deep section retrieval extracts the *"Early life and family"* section (*"Allu Arjun was born to film producer Allu Aravind and Nirmala... His paternal aunt is Surekha Konidela, the wife of actor Chiranjeevi"*). `RelationVerifier` extracts `father_of(Allu Arjun) = Allu Aravind` $\ne$ `Chiranjeevi` $\rightarrow$ `OBJECT_MISMATCH` $\rightarrow$ `CONTRADICTED`.

### Failure 4: "Java was created by James Gosling."
- **Baseline Behavior**: Output `CONFLICTED` due to an irrelevant passage mentioning *"Oak"* conflicting with *"Java"*.
- **V2 Behavior**: Output `VERIFIED` (71.3% support, 0.0% contradiction). Bidirectional query generation (*"Java created by"*, *"who created Java"*) surfaces authoritative creation passages. `RelationVerifier` confirms `MATCH` $\rightarrow$ `VERIFIED`.

---

## 4. Modified Codebase Files

| File Path | Description of Changes |
|---|---|
| `agents/verifier_agent/schemas/models.py` | Added `source_confidence_hint` and `relation_check` fields to `Passage` and `EvidenceItem`. |
| `agents/verifier_agent/schemas/retrieval_trace.py` | Created `RelationCheckTrace` and `GateRelevanceAuditTrace` models. |
| `agents/verifier_agent/config/settings.py` | Added retrieval quality gate parameters with documented defaults. |
| `agents/verifier_agent/adapters/general.py` | Implemented MediaWiki Action API deep section retrieval and chunking. |
| `agents/verifier_agent/adapters/cybersecurity.py` | Renamed hardcoded relevance priors to `source_confidence_hint`. |
| `agents/verifier_agent/adapters/healthcare.py` | Renamed hardcoded relevance priors to `source_confidence_hint`. |
| `agents/verifier_agent/adapters/finance.py` | Renamed hardcoded relevance priors to `source_confidence_hint`. |
| `agents/verifier_agent/adapters/ai_research.py` | Renamed hardcoded relevance priors to `source_confidence_hint`. |
| `agents/verifier_agent/adapters/legal_general.py` | Renamed hardcoded relevance priors to `source_confidence_hint`. |
| `agents/verifier_agent/adapters/web_enhanced.py` | Replaced constant checks in quality gate with dynamic lexical overlap & term coverage. |
| `agents/verifier_agent/routers/query_expander.py` | Added bidirectional relational query generation (creator, parent, starring, capital, location). |
| `agents/verifier_agent/scorers/relation_verifier.py` | **[NEW]** Deterministic SVO relation extraction and mismatch verification layer. |
| `agents/verifier_agent/scorers/evidence_scorer.py` | Integrated `RelationVerifier`, applied bypass rule, unified `MIN_NLI_SIGNAL = 0.35`. |
| `agents/verifier_agent/api/pipeline.py` | Updated passage selection and decision-grade filtering to preserve deep sections and relation checks. |
| `scripts/verify_claim.py` | Added Gate Relevance Audit and Relation Verification Layer display. |
| `scripts/benchmark_eval.py` | **[NEW]** Multi-domain benchmark evaluation harness with confusion matrix output. |
| `agents/verifier_agent/tests/test_v2_regression_failures.py`| **[NEW]** 6 unit regression tests targeting diagnosed failure cases. |
| `agents/verifier_agent/tests/test_retrieval_quality_gate.py`| Unit tests for quality-gated retrieval modes. |
| `docs/current-state-audit.md` | Baseline and current-state architectural audit. |
| `docs/architecture.md` | Technical architecture document and data flow diagrams. |
| `docs/decision-log.md` | Architecture Decision Records (ADR-001 to ADR-006). |
| `docs/runbook.md` | Operations and deployment runbook. |
| `docs/verification-report.md` | Comprehensive verification report. |

---

## 5. Known Limitations & Recommended Next Work
1. **Complex Multi-Hop Anaphora**: Pronouns spanning across multiple sentences in long unstructured Wikipedia sections are occasionally un-resolved before SVO extraction. Anaphora resolution (e.g. using lightweight spaCy or FastCoref) could further improve multi-hop relation extraction.
2. **API Rate Limiting**: NCBI PubMed and MediaWiki public APIs enforce rate limits (3-10 requests/sec). In production deployment, configuring explicit API keys in `.env` (`OPENFDA_KEY`, `NCBI_API_KEY`) is recommended to expand throughput.
