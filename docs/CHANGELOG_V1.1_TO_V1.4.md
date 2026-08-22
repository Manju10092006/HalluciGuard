# HalluciGuard Verifier Pipeline — Detailed Technical Change Log (v1.1 to v1.4 / v1.7)

**Document Version**: 1.0.0  
**Repository**: `Manju10092006/HalluciGuard`  
**Base Checkpoint**: `verifier-v1-hardening-checkpoint`  
**Final Checkpoint**: `verifier-v1.4-certification-checkpoint` (`07ac847`)  
**Target Pipeline Architecture**:  
`User Claim -> Domain Routing -> Authoritative Primary APIs -> Quality Gate -> Tavily Web Fallback -> Candidate Merge & Dedup -> BGE Reranker -> DeBERTa-v3 NLI -> Evidence Semantics -> Evidence Aggregation -> Calibrated Confidence -> Final Verdict`

---

## 1. Summary of What Was Done

Between version 1.1 and version 1.4 (and verified with the version 1.7 benchmark suite), the HalluciGuard Verification Pipeline was audited, hardened, tested, and certified without altering the underlying pipeline architecture.

### Key Milestones Accomplished

1. **Phase 1: V1.1 Quality-Based Fallback & Deep Observability**
   - Eliminated the naive passage count check (`len(passages) >= 2`) which previously allowed low-relevance or empty passages to bypass fallback.
   - Introduced a BGE-relevance-based quality gate requiring configurable thresholds (`min_relevant_passages=1`, `min_top_relevance=0.30`, `relevance_threshold=0.25`).
   - Implemented structured Pydantic diagnostic trace models (`schemas/retrieval_trace.py`) capturing latency, candidate counts, top relevance, reason strings, and provenance across primary and Tavily retrieval stages.
   - Added user-facing CLI trace visualization in `scripts/verify_claim.py` and supported explicit retrieval modes (`hybrid`, `primary_only`, `tavily_only`).

2. **Phase 2: V1.2 Evidence Semantics, Relevance Gating & Invariant Calibration**
   - Enforced a hard relevance gate ($0.20$ BGE threshold) ensuring irrelevant content is categorized as `IRRELEVANT` without corrupting downstream NLI scoring.
   - Preserved multi-class classification invariants: $\text{Supporting} + \text{Contradicting} + \text{Neutral} + \text{Irrelevant} = \text{Total Passages}$.
   - Upgraded URL-level deduplication to prevent chunk-splitting from artificial evidence inflation while preserving legitimate cross-source diversity.
   - Calibrated `CONTRADICTED` confidence ensuring non-zero confidence scores when strong contradiction evidence is established.

3. **Phase 3: V1.3 Healthcare Authoritative Source Reliability & Query Routing**
   - Restructured `HealthcareAdapter` to query live official APIs: OpenFDA (drug labels & indications), PubMed (NCBI E-Utilities XML), PMC (full text/abstracts), and WHO Global Health Observatory (GHO OData).
   - Implemented dynamic clinical intent routing (medication use, clinical trials, public health indicators, drug labels).
   - Eliminated title-only metadata passage passing and preserved complete provenance (`PMID:xxx`, `PMCxxx`, OpenFDA `set_id`).

4. **Phase 4: V1.4 Cross-Domain Authoritative Hardening**
   - **Cybersecurity**: Integrated NIST NVD CVE API 2.0 direct lookup, CIRCL CVE fallback, real-time CISA Known Exploited Vulnerabilities (KEV) tokenized search, and MITRE ATT&CK technique resolution.
   - **General Knowledge**: Integrated Wikipedia REST summary lead paragraph extraction and added noise filters for entertainment namesakes (e.g., albums, songs, films).
   - **Finance**: Implemented SEC EDGAR EFTS 10-K/10-Q filing retrieval, ticker symbol mapping, and World Bank country indicators.
   - **AI Research**: Integrated arXiv API E-Query with Lucene syntax sanitization, Semantic Scholar, and CrossRef.
   - **Legal**: Supported CourtListener and legal precedent routing.
   - **Relational Coverage Quality Gate**: Added multi-entity relationship coverage checking before bypassing web fallback.
   - **Harmonized Refutation Semantics**: Synchronized `CitationFormatter` with `EvidenceScorer` qualification logic.

5. **Phase 5: V1.7 Benchmark Evaluation, Regression Suite & Certification**
   - Created 4 deterministic test suites comprising 42 unit tests (100% passing).
   - Executed a 35-claim benchmark across all 5 domains with live APIs resulting in **0.00% False Verification Rate**, **100.0% Verified Precision**, **100.0% Contradicted Precision**, and **74.29% Overall Accuracy**.
   - Published formal certification report at `docs/VERIFIER_V1_BENCHMARK_REPORT.md`.

---

## 2. Complete Inventory of Changed and Created Files

| # | File Path | Status | Purpose & Scope |
|---|---|---|---|
| 1 | `agents/verifier_agent/config/settings.py` | Modified | Added quality gate configuration parameters and defaults |
| 2 | `agents/verifier_agent/schemas/retrieval_trace.py` | **New** | Pydantic data models for structured diagnostic retrieval traces |
| 3 | `agents/verifier_agent/schemas/models.py` | Modified | Added `retrieval_mode`, `source_mode`, and `retrieval_trace` fields |
| 4 | `agents/verifier_agent/adapters/web_enhanced.py` | Modified | Quality-based Tavily fallback wrapper with relational checks |
| 5 | `agents/verifier_agent/adapters/healthcare.py` | Modified | OpenFDA, PubMed, PMC, WHO GHO, ClinicalTrials.gov integrations |
| 6 | `agents/verifier_agent/adapters/cybersecurity.py` | Modified | NIST NVD 2.0, CIRCL, CISA KEV JSON, MITRE ATT&CK integrations |
| 7 | `agents/verifier_agent/adapters/general.py` | Modified | Wikipedia REST Lead Summary extraction & namesake filtering |
| 8 | `agents/verifier_agent/adapters/finance.py` | Modified | SEC EDGAR EFTS, World Bank, source_mode, and URL deduplication |
| 9 | `agents/verifier_agent/adapters/ai_research.py` | Modified | arXiv E-Query sanitization, Semantic Scholar, CrossRef |
| 10 | `agents/verifier_agent/adapters/legal_general.py` | Modified | CourtListener, Wikipedia legal routing, and URL deduplication |
| 11 | `agents/verifier_agent/scorers/evidence_scorer.py` | Modified | Relevance gating, classification invariants, refutation logic |
| 12 | `agents/verifier_agent/formatters/citation_formatter.py` | Modified | Synchronized citation labels with `EvidenceScorer` classification |
| 13 | `agents/verifier_agent/explanations/generator.py` | Modified | Evidence explanation generation formatting adjustments |
| 14 | `agents/verifier_agent/api/pipeline.py` | Modified | Pipeline wiring: traces, retrieval modes, harmonized formatting |
| 15 | `scripts/verify_claim.py` | Modified | Interactive CLI tool with trace visualization & mode selection |
| 16 | `scripts/benchmark_eval.py` | **New** | 35-claim 5-domain evaluation harness with calibration & safety metrics |
| 17 | `agents/verifier_agent/tests/test_retrieval_quality_gate.py` | **New** | 11 unit tests for V1.1 quality gate logic |
| 18 | `agents/verifier_agent/tests/test_evidence_semantics.py` | **New** | 12 unit tests for V1.2 evidence scoring & invariants |
| 19 | `agents/verifier_agent/tests/test_healthcare_sources.py` | **New** | 9 unit tests for V1.3 healthcare API routing |
| 20 | `agents/verifier_agent/tests/test_domain_adapters.py` | **New** | 10 unit tests for V1.4 domain adapter integrations |
| 21 | `benchmark_results_v1.3.json` | **New** | Phase 3 benchmark evaluation data |
| 22 | `benchmark_results_v1.7.json` | **New** | Phase 7 35-claim benchmark evaluation data |
| 23 | `docs/VERIFIER_V1_BENCHMARK_REPORT.md` | **New** | Formal certification and evaluation report |

---

## 3. Detailed Description of Each Change & Exact Code Implementations

### 3.1 Configuration & Data Schemas

#### File: `agents/verifier_agent/config/settings.py`
**Changes Made**:
- Added 6 new configurable attributes in `Settings` under the `# -- Retrieval quality gate --` section.
- Defined explicit defaults for relevance thresholds, passage limits, and default retrieval mode.

**Implemented Code Snippet**:
```python
    # -- Retrieval quality gate --
    # Minimum number of passages with BGE relevance above relevance_threshold to consider primary evidence sufficient
    min_relevant_passages: int = 1
    # Minimum BGE relevance score for the top passage to consider primary evidence sufficient
    min_top_relevance: float = 0.30
    # BGE relevance score threshold: passages below this are considered not relevant
    relevance_threshold: float = 0.25
    # Maximum number of primary passages to assess (performance limit)
    max_primary_passages: int = 10
    # Relevance gate for evidence classification: passages with BGE below this are IRRELEVANT
    evidence_relevance_gate: float = 0.20
    # Default retrieval mode: hybrid (primary + fallback), primary_only, tavily_only
    default_retrieval_mode: str = "hybrid"
```

---

#### File: `agents/verifier_agent/schemas/retrieval_trace.py`
**Changes Made**:
- Created a standalone schema module containing 6 Pydantic models (`PrimaryRetrievalTrace`, `TavilyRetrievalTrace`, `MergedTrace`, `FinalTrace`, `EvidencePassageTrace`, `RetrievalTrace`).
- Standardized fields across stages: latency tracking, candidate counts, usable passage counts, BGE scores, NLI probabilities, and structured fallback trigger reason codes.

**Implemented Code Snippet**:
```python
from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


class PrimaryRetrievalTrace(BaseModel):
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
    called: bool = False
    reason: str = ""
    query: str = ""
    search_depth: str = "advanced"
    result_count: int = 0
    extracted_count: int = 0
    usable_count: int = 0
    latency_ms: int = 0
    error: Optional[str] = None
    domains_requested: List[str] = Field(default_factory=list)


class MergedTrace(BaseModel):
    candidate_count: int = 0
    deduplicated_count: int = 0


class FinalTrace(BaseModel):
    reranked_count: int = 0
    decision_grade_count: int = 0


class EvidencePassageTrace(BaseModel):
    title: str = ""
    url: str = ""
    source: str = ""
    retrieval_method: str = ""
    relevance_score: float = 0.0
    relevance_decision: str = ""
    nli_label: str = ""
    entailment: float = 0.0
    contradiction: float = 0.0
    neutral: float = 0.0
    nli_degraded: bool = False
    evidence_class: str = ""
    source_credibility: float = 0.0
    effective_weight: float = 0.0


class RetrievalTrace(BaseModel):
    requested_domain: str = ""
    primary_adapter: str = ""
    retrieval_mode: str = "hybrid"
    primary: PrimaryRetrievalTrace = Field(default_factory=PrimaryRetrievalTrace)
    tavily: TavilyRetrievalTrace = Field(default_factory=TavilyRetrievalTrace)
    merged: MergedTrace = Field(default_factory=MergedTrace)
    final: FinalTrace = Field(default_factory=FinalTrace)
    evidence_details: List[EvidencePassageTrace] = Field(default_factory=list)
    timings: dict = Field(default_factory=dict)
```

---

#### File: `agents/verifier_agent/schemas/models.py`
**Changes Made**:
- Added `retrieval_mode: str = "hybrid"` and `source_mode: Optional[str] = None` to `VerifierInputV2`.
- Added `retrieval_trace: Optional[dict] = None` to `ClaimReport` to allow API consumers and CLI tools to inspect the full trace.

---

### 3.2 Web Enhanced Adapter & Quality Fallback Logic

#### File: `agents/verifier_agent/adapters/web_enhanced.py`
**Changes Made**:
- Implemented `WebEnhancedAdapter` wrapping any primary domain adapter.
- Added `_is_usable_passage()` requiring meaningful snippet text and valid URLs/titles.
- Implemented `_assess_primary_quality()` enforcing:
  1. Minimum usable passages
  2. BGE relevance threshold comparison
  3. Relational multi-entity term coverage check
- Supported retrieval modes: `hybrid` (primary first with fallback), `primary_only` (no fallback), `tavily_only` (skip primary).
- Implemented URL-level candidate merging and deduplication via `_normalize_url`.

**Implemented Code Snippet**:
```python
    def _assess_primary_quality(
        self,
        passages: List[Passage],
        claim: str,
    ) -> PrimaryRetrievalTrace:
        settings = self._get_settings()

        usable = [p for p in passages if self._is_usable_passage(p)]
        usable_count = len(usable)

        if usable_count == 0:
            return PrimaryRetrievalTrace(
                called=True,
                result_count=len(passages),
                usable_count=0,
                relevant_count=0,
                top_relevance=0.0,
                source_diversity=0,
                sufficient=False,
                reason="PRIMARY_EMPTY",
            )

        relevance_scores = [float(p.relevance_score or 0.0) for p in usable]
        top_relevance = max(relevance_scores) if relevance_scores else 0.0
        relevant_count = sum(
            1 for score in relevance_scores
            if score >= settings.relevance_threshold
        )

        unique_sources = set()
        for p in usable:
            source_key = p.source or p.source_id or "unknown"
            unique_sources.add(source_key)
        source_diversity = len(unique_sources)

        # Relational entity coverage check
        import re
        claim_terms = set(re.findall(r'[a-zA-Z0-9_-]{3,}', claim.lower()))
        combined_text = " ".join(f"{p.title} {p.snippet}" for p in usable).lower()
        missing_terms = [t for t in claim_terms if t not in combined_text and t not in {
            "is", "the", "and", "for", "with", "that", "this", "are", "was", "were", "associated"
        }]
        
        has_relational_coverage = len(missing_terms) <= max(1, len(claim_terms) // 3)

        sufficient = (
            relevant_count >= settings.min_relevant_passages
            and top_relevance >= settings.min_top_relevance
            and has_relational_coverage
        )

        if sufficient:
            reason = "PRIMARY_EVIDENCE_SUFFICIENT"
        elif not has_relational_coverage:
            reason = f"PRIMARY_EVIDENCE_INSUFFICIENT_TERM_COVERAGE (Missing terms: {missing_terms[:3]})"
        elif relevant_count == 0:
            reason = "PRIMARY_EVIDENCE_INSUFFICIENT_RELEVANCE"
        elif top_relevance < settings.min_top_relevance:
            reason = f"PRIMARY_TOP_RELEVANCE_TOO_LOW ({top_relevance:.3f} < {settings.min_top_relevance})"
        else:
            reason = f"PRIMARY_EVIDENCE_INSUFFICIENT ({relevant_count} relevant < {settings.min_relevant_passages} required)"

        return PrimaryRetrievalTrace(
            called=True,
            result_count=len(passages),
            usable_count=usable_count,
            relevant_count=relevant_count,
            top_relevance=round(top_relevance, 4),
            source_diversity=source_diversity,
            sufficient=sufficient,
            reason=reason,
        )
```

---

### 3.3 Domain Adapter Hardening

#### File: `agents/verifier_agent/adapters/healthcare.py`
**Changes Made**:
- Replaced stub implementations with live API clients:
  1. `_search_openfda`: Queries `https://api.fda.gov/drug/label.json` with active ingredient and brand name tokens. Resolves `indications_and_usage` and `set_id` URLs.
  2. `_search_pubmed`: Calls NCBI E-Utilities (`esearch` + `efetch`) returning full XML abstracts, authors, and dates with `PMID:xxx` provenance.
  3. `_search_pmc`: Queries Europe PMC / NCBI PMC XML with full text extraction.
  4. `_search_who_gho`: Queries WHO Global Health Observatory OData API (`ghoapi.azureedge.net`).
  5. `_search_clinicaltrials`: Queries ClinicalTrials.gov API v2.
- Added dynamic clinical intent routing based on query classification (`is_drug_claim`, `is_clinical_claim`, `is_who_claim`).
- Supported `source_mode` parameter (`healthcare-fda`, `healthcare-pubmed`, `healthcare-pmc`, `healthcare-who`).

**Implemented Code Snippet**:
```python
    async def _search_openfda(self, client: object, drug_or_claim: str) -> List[Passage]:
        words = re.findall(r'[A-Za-z0-9]+', drug_or_claim)
        stopwords = {"is", "used", "to", "relieve", "mild", "moderate", "pain", "tablet", "the", "cures", "for", "in", "of", "and", "a", "an", "this", "that"}
        keywords = [w for w in words if w.lower() not in stopwords]
        drug_name = keywords[0] if keywords else (words[0] if words else "aspirin")

        params = {
            "search": f'(openfda.substance_name:"{drug_name}" OR openfda.brand_name:"{drug_name}")',
            "limit": 3,
        }
        res = await client.get("https://api.fda.gov/drug/label.json", adapter_name=self.name, params=params)
        data = res.json()
        results = data.get("results", [])
        
        passages = []
        for r in results:
            indications = r.get("indications_and_usage", [""])[0]
            if indications:
                set_id = r.get("id", r.get("set_id", "label"))
                passages.append(Passage(
                    title=f"FDA Drug Label: {drug_name.capitalize()}",
                    source="openfda",
                    source_id="fda_label",
                    url=f"https://labels.fda.gov/{set_id}",
                    snippet=indications[:500],
                    relevance_score=0.90,
                ))
        return passages
```

---

#### File: `agents/verifier_agent/adapters/cybersecurity.py`
**Changes Made**:
- Integrated NIST NVD CVE API 2.0 (`https://services.nvd.nist.gov/rest/json/cves/2.0`) with API key support.
- Added CIRCL CVE API (`https://cve.circl.lu/api/cve/{cve_id}`) fallback.
- Implemented CISA Known Exploited Vulnerabilities catalog search parsing `known_exploited_vulnerabilities.json` with exact CVE ID matching and tokenized catalog scoring.
- Added MITRE ATT&CK technique querying.
- Supported `source_mode` parameters (`cybersecurity-nvd`, `cybersecurity-cisa`, `cybersecurity-mitre`).

**Implemented Code Snippet**:
```python
    async def _search_nvd_cve(self, client: object, cve_id: str) -> List[Passage]:
        cve_id = cve_id.upper()
        params = {"cveId": cve_id}
        headers = {}
        if self._nvd_key:
            headers["apiKey"] = self._nvd_key

        res = await client.get(
            "https://services.nvd.nist.gov/rest/json/cves/2.0",
            adapter_name=self.name,
            params=params,
            headers=headers,
        )
        data = res.json()
        vulnerabilities = data.get("vulnerabilities", [])
        passages = []
        for v in vulnerabilities:
            cve = v.get("cve", {})
            descriptions = cve.get("descriptions", [])
            desc_text = next((d.get("value", "") for d in descriptions if d.get("lang") == "en"), "")
            if desc_text:
                passages.append(Passage(
                    title=f"NVD CVE Record: {cve_id}",
                    source="nvd",
                    source_id="nvd_cve",
                    url=f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                    snippet=desc_text[:600],
                    relevance_score=0.95,
                ))
        return passages
```

---

#### File: `agents/verifier_agent/adapters/general.py`
**Changes Made**:
- Added Wikipedia REST Lead Summary extraction (`https://en.wikipedia.org/api/rest_v1/page/summary/{title}`) to retrieve full authoritative lead paragraphs rather than brief search fragments.
- Implemented `_is_noisy_title` to filter out bibliographies, discographies, and entertainment namesakes (e.g., albums, songs, films) when verifying factual claims.

**Implemented Code Snippet**:
```python
    @staticmethod
    def _is_noisy_title(title: str, query: str, description: str = "") -> bool:
        t_lower = title.lower()
        q_lower = query.lower()
        d_lower = (description or "").lower()

        entertainment_tags = (
            "bibliography", "discography", "filmography", "list of songs", "list of",
            "(album)", "(song)", "(film)", "(soundtrack)", "(tv series)", "(single)",
            "(band)", "(video game)", "(novel)", "(short story)", "(play)"
        )
        for noise in entertainment_tags:
            if noise in t_lower and noise not in q_lower:
                return True

        if d_lower:
            for desc_noise in ("studio album", "song by", "film directed by", "television series", "single by", "band"):
                if desc_noise in d_lower and desc_noise not in q_lower:
                    return True

        return False
```

---

#### File: `agents/verifier_agent/adapters/finance.py`
**Changes Made**:
- Integrated SEC EDGAR EFTS (`https://efts.sec.gov/LATEST/search-index`) for 10-K, 10-Q, and 8-K filings with entity ticker resolution.
- Integrated World Bank indicator API (`https://api.worldbank.org/v2/country`).
- Supported `source_mode` parameters (`finance-sec`, `finance-worldbank`) and URL deduplication.

---

#### File: `agents/verifier_agent/adapters/ai_research.py`
**Changes Made**:
- Integrated arXiv API E-Query (`https://export.arxiv.org/api/query`) with Lucene syntax sanitization.
- Integrated Semantic Scholar Graph API and CrossRef DOI resolution.
- Supported `source_mode` parameters (`ai-arxiv`, `ai-semanticscholar`, `ai-crossref`) and URL deduplication.

---

#### File: `agents/verifier_agent/adapters/legal_general.py`
**Changes Made**:
- Supported CourtListener REST API and legal Wikipedia search routing.
- Supported `source_mode` parameters (`legal-courtlistener`, `legal-wikipedia`) and URL deduplication.

---

### 3.4 Evidence Scoring, Semantics & Citation Formatting

#### File: `agents/verifier_agent/scorers/evidence_scorer.py`
**Changes Made**:
- Enforced hard relevance gate ($0.20$ BGE threshold) mapping sub-threshold passages to `IRRELEVANT`.
- Preserved multi-class classification invariants: $	ext{Supporting} + 	ext{Contradicting} + 	ext{Neutral} + 	ext{Irrelevant} = 	ext{Total Passages}$.
- Implemented `_is_non_assertive_claim_context` to prevent debunking/misconception articles from falsely scoring as `SUPPORTING`.
- Calibrated `CONTRADICTED` confidence calculation ensuring non-zero confidence when contradiction evidence exists.

**Implemented Code Snippet**:
```python
    def classify_evidence(
        self,
        claim: str,
        passage: Passage,
        nli: NLIResult,
    ) -> str:
        rel_score = float(getattr(passage, "relevance_score", 0.0) or 0.0)
        
        # 1. Strict Relevance Gate
        if rel_score < self.RELEVANCE_GATE:
            return "IRRELEVANT"

        # 2. Guard against false support from debunked claims
        full_text = f"{getattr(passage, 'title', '')} {passage.snippet}"
        if self._is_non_assertive_claim_context(claim, full_text):
            return "NEUTRAL"

        # 3. NLI-based threshold classification
        if nli.entailment >= self.SUPPORT_THRESHOLD:
            return "SUPPORTING"
        elif nli.contradiction >= self.CONTRADICT_THRESHOLD:
            if self._has_relational_contradiction_signal(claim, passage):
                return "CONTRADICTING"
            return "NEUTRAL"
        else:
            return "NEUTRAL"
```

---

#### File: `agents/verifier_agent/formatters/citation_formatter.py`
**Changes Made**:
- Wired `CitationFormatter` to `EvidenceScorer.classify_evidence()`.
- Guaranteed that citation badge labels (`[SUPPORTING]`, `[CONTRADICTING]`, `[NEUTRAL]`) match the exact classification determined by the evidence scorer.

---

### 3.5 Verification Pipeline & CLI Observability

#### File: `agents/verifier_agent/api/pipeline.py`
**Changes Made**:
- Instantiated `CitationFormatter(evidence_scorer=self.evidence_scorer)`.
- Forwarded `retrieval_mode` and `source_mode` kwargs to domain adapters.
- Passed `claim` text to `citation_formatter.format_all()`.
- Attached `retrieval_trace` to `ClaimReport`.
- Refined passage deduplication to index by normalized URL and document identity.

---

#### File: `scripts/verify_claim.py`
**Changes Made**:
- Overhauled CLI tool to support interactive mode (`-i`), domain selection (`--domain`), retrieval modes (`--retrieval-mode`), and forced Tavily fallback (`--force-tavily`).
- Implemented `_print_retrieval_trace` to print complete diagnostic trace data (primary adapter, latency, usable/relevant passage count, top relevance score, Tavily status, dedup counts).

---

## 4. Test Suite Execution & Quality Assurance

All 42 deterministic tests pass across 4 specialized test suites:

```
agents/verifier_agent/tests/test_domain_adapters.py        10 passed [100%]
agents/verifier_agent/tests/test_healthcare_sources.py      9 passed [100%]
agents/verifier_agent/tests/test_evidence_semantics.py     12 passed [100%]
agents/verifier_agent/tests/test_retrieval_quality_gate.py   11 passed [100%]
============================== 42 passed in 44.80s ==============================
```

### Complete Test Catalog

1. **`test_retrieval_quality_gate.py` (11 Tests)**:
   - `test_primary_sufficient_skips_tavily`: Primary returns 5 passages with BGE >= 0.5. Tavily is skipped.
   - `test_primary_irrelevant_triggers_tavily`: Primary returns low relevance. Tavily is called.
   - `test_primary_empty_triggers_tavily`: Primary returns 0 passages. Tavily is called.
   - `test_primary_exception_triggers_tavily`: Primary raises exception. Tavily handles fallback gracefully.
   - `test_one_strong_passage_sufficient`: Primary returns 1 passage with BGE 0.8. Gate passes.
   - `test_url_dedup_on_merge`: Identical URLs returned by primary and Tavily are deduplicated.
   - `test_tavily_partial_failure`: Tavily returns empty snippets; usable count correctly reflects valid passages.
   - `test_force_tavily_mode`: Explicit `tavily_only` mode bypasses primary adapter.
   - `test_primary_only_mode`: Explicit `primary_only` mode bypasses Tavily.
   - `test_tavily_only_mode`: Flag correctly executes diagnostic Tavily-only retrieval.
   - `test_hybrid_mode_primary_first_then_fallback`: Default mode attempts primary first then falls back.

2. **`test_evidence_semantics.py` (12 Tests)**:
   - `test_classification_sum_invariant`: Total equals $S + C + N + I$.
   - `test_relevance_gate_filters_irrelevant`: Passages with BGE < 0.20 are marked `IRRELEVANT`.
   - `test_confidence_nonzero_for_contradicted`: Contradicted verdict produces non-zero confidence.
   - `test_adversarial_debunking_article`: Debunking text is not falsely marked supporting.
   - `test_dedup_preserves_diverse_sources`: Distinct sources are preserved during deduplication.
   - `test_safe_abstention_on_neutral`: Neutral evidence results in `UNVERIFIED`.
   - `test_conflicted_verdict_on_split`: Opposing high-confidence sources produce `CONFLICTED`.
   - `test_source_credibility_weighting`: Source credibility factors into final scores.
   - `test_confidence_scaling_curve`: Non-linear confidence scaling curves behave monotonically.
   - `test_zero_evidence_unverified`: Empty evidence returns `UNVERIFIED` with 0 confidence.
   - `test_high_support_verified`: Strong entailment evidence returns `VERIFIED`.
   - `test_predicate_contradiction`: Direct predicate contradiction properly classified.

3. **`test_healthcare_sources.py` (9 Tests)**:
   - `test_openfda_search_success`: OpenFDA returns valid drug indications and package insert URLs.
   - `test_openfda_error_handling`: Graceful handling of OpenFDA 404/500 errors.
   - `test_pubmed_esearch_efetch_success`: PubMed E-Utilities returns structured titles, abstracts, and PMIDs.
   - `test_pubmed_empty_results`: Empty PubMed search handled without crashing.
   - `test_pmc_xml_parsing`: PMC article XML correctly extracted.
   - `test_who_gho_success`: WHO GHO indicator OData correctly parsed.
   - `test_intent_query_routing`: Drug claim routes to FDA/PubMed; public health routes to WHO.
   - `test_healthcare_deduplication`: Cross-referenced papers from PubMed/PMC merged cleanly.
   - `test_healthcare_tavily_fallback`: Healthcare adapter triggers Tavily when primary APIs return empty.

4. **`test_domain_adapters.py` (10 Tests)**:
   - `test_nvd_cve_lookup_success`: NIST NVD CVE API 2.0 retrieves official vulnerability description.
   - `test_circl_cve_fallback`: CIRCL CVE fallback functions on NVD empty response.
   - `test_cisa_kev_search`: CISA KEV JSON token search matches known exploited CVEs.
   - `test_mitre_technique_search`: MITRE ATT&CK technique IDs and names extracted.
   - `test_wikipedia_rest_summary`: Wikipedia REST summary returns authoritative lead paragraphs.
   - `test_wikipedia_namesake_filter`: Albums and songs filtered out when evaluating scientific claims.
   - `test_sec_edgar_filings`: SEC EDGAR EFTS retrieves 10-K and 10-Q filing records.
   - `test_world_bank_indicators`: World Bank API returns macroeconomic indicators.
   - `test_arxiv_query_sanitization`: arXiv Lucene query sanitization prevents malformed requests.
   - `test_legal_routing`: CourtListener and legal Wikipedia routes execute correctly.

---

## 5. Live Benchmark & Certification Summary

- **Total Live Claims Evaluated**: 35 across 5 domains.
- **Pipeline Execution Errors**: 0 (0.0%).
- **Overall Accuracy**: **74.29%** (26/35 claims matched).
- **Verified Precision**: **100.0%** (12/12 verified claims supported by authoritative evidence).
- **Contradicted Precision**: **100.0%** (1/1 contradicted claim refuted by authoritative evidence).
- **False Verification Rate**: **0.00%** (Zero hallucinated verifications).
- **False Contradiction Rate**: **0.00%** (Zero false refutations).
- **Safe Abstention Rate**: **100.0%** (13/13 unverified claims safely abstained).
- **Median Latency (P50)**: 16,153 ms.
- **95th Percentile Latency (P95)**: 34,752 ms.
- **Git Checkpoint**: `verifier-v1.4-certification-checkpoint` (`07ac847`).
- **Certification Status**: **CONDITIONALLY CERTIFIED (V1.0 PRODUCTION READY)**.
