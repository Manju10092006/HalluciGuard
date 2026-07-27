# 🛡️ HalluciGuard Repository Engineering Audit & Technical Reference

> **Audit Type**: Complete Read-Only Technical Audit  
> **Repository Name**: `HalluciGuard`  
> **Execution Mode**: READ-ONLY  
> **Audit Scope**: Entire Repository (`verifier_agent`, `judge_agent`, `detector_agent`, `corrector_agent`, `memory_agent`)

---

# 1. Complete Folder Structure

```
HalluciGuard/
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── 12121.txt
├── 33.jpeg
├── Archi.jpeg
├── WhatsApp Image 2026-07-21 at 4.23.07 PM.jpeg
├── Stock Market R03417W69RP3OFG8.txt
├── The complete, production-grade Glob.txt
├── Verifier_Agent_Build_Spec_v2.md
├── VERIFIER_AGENT_TECHNICAL_DOCUMENTATION.md
├── VERIFIER_AGENT_TECHNICAL_DOCUMENTATION.html
├── VERIFIER_AGENT_TECHNICAL_DOCUMENTATION.pdf
├── VERIFIER_AGENT_EXECUTION_AND_VALIDATION_REPORT.md
├── VERIFIER_AGENT_EXECUTION_AND_VALIDATION_REPORT.html
├── VERIFIER_AGENT_EXECUTION_AND_VALIDATION_REPORT.pdf
├── VERIFIER_AGENT_STATUS_AND_ROADMAP.md
├── VERIFIER_RETRIEVAL_DEBUG_REPORT.md
├── VERIFIER_RETRIEVAL_DEBUG_REPORT.html
├── VERIFIER_RETRIEVAL_DEBUG_REPORT.pdf
├── JUDGE_AGENT_TECHNICAL_DOCUMENTATION.md
├── JUDGE_AGENT_TECHNICAL_DOCUMENTATION.html
├── JUDGE_AGENT_TECHNICAL_DOCUMENTATION.pdf
├── ctg-oas-v2.yaml
├── halluciguard_judge/
│   ├── app.py
│   ├── confidence_calibrator.py
│   ├── config.py
│   ├── decision_engine.py
│   ├── judge_agent.py
│   ├── nli_engine.py
│   ├── simulator.py
│   └── test_judge.py
└── agents/
    ├── __init__.py
    ├── corrector_agent/
    │   ├── __init__.py
    │   └── README.md
    ├── detector_agent/
    │   ├── __init__.py
    │   └── README.md
    ├── judge_agent/
    │   ├── JUDGE_AGENT_TECHNICAL_DOCUMENTATION.md
    │   ├── JUDGE_AGENT_TECHNICAL_DOCUMENTATION.pdf
    │   ├── README.md
    │   ├── __init__.py
    │   ├── app.py
    │   ├── confidence_calibrator.py
    │   ├── config.py
    │   ├── decision_engine.py
    │   ├── judge_agent.py
    │   ├── nli_engine.py
    │   ├── simulator.py
    │   └── test_judge.py
    ├── memory_agent/
    │   ├── __init__.py
    │   └── README.md
    └── verifier_agent/
        ├── .env.example
        ├── pytest.ini
        ├── requirements.txt
        ├── version.py
        ├── container.py
        ├── verification_cache.db
        ├── VERIFIER_AGENT_EXECUTION_AND_VALIDATION_REPORT.md
        ├── VERIFIER_AGENT_EXECUTION_AND_VALIDATION_REPORT.pdf
        ├── VERIFIER_AGENT_STATUS_AND_ROADMAP.md
        ├── VERIFIER_AGENT_TECHNICAL_DOCUMENTATION.md
        ├── VERIFIER_AGENT_TECHNICAL_DOCUMENTATION.pdf
        ├── VERIFIER_RETRIEVAL_DEBUG_REPORT.pdf
        ├── adapters/
        │   ├── __init__.py
        │   ├── registry.py
        │   ├── domain_proxy.py
        │   ├── healthcare.py
        │   ├── cybersecurity.py
        │   ├── finance.py
        │   ├── ai_research.py
        │   ├── legal_general.py
        │   ├── general.py
        │   ├── stub_adapter.py
        │   ├── mock_adapter.py
        │   └── fixtures/
        │       ├── healthcare.json
        │       ├── cybersecurity.json
        │       └── ai_research.json
        ├── aggregation/
        │   ├── __init__.py
        │   ├── aggregator.py
        │   ├── duplicate_remover.py
        │   └── evidence_merger.py
        ├── api/
        │   ├── __init__.py
        │   ├── main.py
        │   └── pipeline.py
        ├── benchmarks/
        │   ├── __init__.py
        │   ├── fever.py
        │   ├── pubhealth.py
        │   ├── metrics.py
        │   └── runner.py
        ├── cache/
        │   ├── __init__.py
        │   └── sqlite_cache.py
        ├── claims/
        │   ├── __init__.py
        │   ├── claim_decomposer.py
        │   ├── claim_normalizer.py
        │   └── claim_merger.py
        ├── config/
        │   ├── __init__.py
        │   ├── settings.py
        │   ├── credibility.yaml
        │   ├── domain_intelligence.yaml
        │   ├── retry.yaml
        │   └── query_expansion/
        │       ├── __init__.py
        │       ├── healthcare.json
        │       ├── cybersecurity.json
        │       ├── finance.json
        │       ├── legal_general.json
        │       └── ai_research.json
        ├── docs/
        │   ├── README.md
        │   ├── DOMAIN_INTELLIGENCE.md
        │   └── IMPLEMENTATION_REPORT.md
        ├── explanations/
        │   ├── __init__.py
        │   └── generator.py
        ├── formatters/
        │   ├── __init__.py
        │   ├── citation_formatter.py
        │   └── response_formatter.py
        ├── logs/
        │   └── __init__.py
        ├── metrics/
        │   ├── __init__.py
        │   ├── collector.py
        │   └── performance.py
        ├── models/
        │   ├── __init__.py
        │   ├── domain_intelligence.py
        │   ├── model_manager.py
        │   ├── model_router.py
        │   └── research_companion.py
        ├── nli/
        │   ├── __init__.py
        │   └── entailment.py
        ├── reports/
        │   ├── __init__.py
        │   └── evaluation_generator.py
        ├── rerankers/
        │   ├── __init__.py
        │   └── cross_encoder.py
        ├── research/
        │   ├── generate_research_notebooks.py
        │   ├── agriculture/ (1 notebook)
        │   ├── artificial_intelligence/ (1 notebook)
        │   ├── astronomy/ (1 notebook)
        │   ├── biology/ (1 notebook)
        │   ├── business/ (1 notebook)
        │   ├── chemistry/ (1 notebook)
        │   ├── climate_environment/ (1 notebook)
        │   ├── computer_science/ (1 notebook)
        │   ├── cybersecurity/ (1 notebook)
        │   ├── data_science/ (1 notebook)
        │   ├── economics/ (1 notebook)
        │   ├── education/ (1 notebook)
        │   ├── embeddings/ (2 notebooks)
        │   ├── finance/ (1 notebook)
        │   ├── food_nutrition/ (1 notebook)
        │   ├── general/ (1 notebook)
        │   ├── genetics/ (1 notebook)
        │   ├── geography/ (1 notebook)
        │   ├── government_public_policy/ (1 notebook)
        │   ├── healthcare/ (3 notebooks)
        │   ├── history/ (1 notebook)
        │   ├── law/ (1 notebook)
        │   ├── legal/ (1 notebook)
        │   ├── machine_learning/ (1 notebook)
        │   ├── mathematics/ (1 notebook)
        │   ├── medicine/ (1 notebook)
        │   ├── nli/ (1 notebook)
        │   ├── pharmacy/ (1 notebook)
        │   ├── philosophy/ (1 notebook)
        │   ├── physics/ (1 notebook)
        │   ├── programming/ (1 notebook)
        │   ├── psychology/ (1 notebook)
        │   ├── rerankers/ (1 notebook)
        │   ├── sociology/ (1 notebook)
        │   └── space_science/ (1 notebook)
        ├── retrievers/
        │   ├── __init__.py
        │   ├── sparse.py
        │   ├── dense.py
        │   └── hybrid.py
        ├── routers/
        │   ├── __init__.py
        │   ├── domain_validator.py
        │   └── query_expander.py
        ├── schemas/
        │   ├── __init__.py
        │   └── models.py
        ├── scorers/
        │   ├── __init__.py
        │   ├── evidence_scorer.py
        │   ├── source_reliability.py
        │   └── conflict_resolver.py
        ├── scratch/
        │   ├── convert_md_to_pdf.py
        │   ├── run_runtime_validation.py
        │   └── validation_output.txt
        ├── tests/
        │   ├── __init__.py
        │   ├── test_adapters.py
        │   ├── test_claims.py
        │   ├── test_domain_intelligence.py
        │   ├── test_health.py
        │   ├── test_official_integrations.py
        │   ├── test_registry.py
        │   ├── test_research_notebooks.py
        │   └── test_verify_smoke.py
        └── utils/
            ├── __init__.py
            ├── logging.py
            ├── http_client.py
            ├── async_executor.py
            └── health_checker.py
```

---

# 2. Complete Project Inventory

| File Name | Relative Path | File Type | Purpose & Responsibility | Importing Modules | Dependent Files | Currently Used | Partial Impl. | Prod-Ready | Placeholder Logic | Size (LOC) | Last Logical Responsibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `__init__.py` | `agents/__init__.py` | Python | Package root defining the 5 HalluciGuard agents overview | External callers | None | Yes | No | Yes | No | 10 | Package docstring & architecture definition |
| `__init__.py` | `agents/corrector_agent/__init__.py` | Python | Package initialization for Corrector Agent | `agents` | None | Yes | Yes | No | Stub docstring | 14 | Package declaration & status |
| `README.md` | `agents/corrector_agent/README.md` | Markdown | Architecture design & specs for Corrector Agent | Developers | Docs | Yes | No | Yes | No | 33 | Agent role & contract specification |
| `__init__.py` | `agents/detector_agent/__init__.py` | Python | Package initialization for Detector Agent | `agents` | None | Yes | Yes | No | Stub docstring | 14 | Package declaration & status |
| `README.md` | `agents/detector_agent/README.md` | Markdown | Architecture design & specs for Detector Agent | Developers | Docs | Yes | No | Yes | No | 48 | Agent role & contract specification |
| `__init__.py` | `agents/memory_agent/__init__.py` | Python | Package initialization for Memory Agent | `agents` | None | Yes | Yes | No | Stub docstring | 14 | Package declaration & status |
| `README.md` | `agents/memory_agent/README.md` | Markdown | Architecture design & specs for Memory Agent | Developers | Docs | Yes | No | Yes | No | 34 | Agent role & contract specification |
| `app.py` | `agents/judge_agent/app.py` | Python | FastAPI server exposing `/judge`, `/simulate`, and `/health` | CLI / Uvicorn | `judge_agent.py` | Yes | No | Yes | No | 393 | REST API endpoints & route handling |
| `confidence_calibrator.py` | `agents/judge_agent/confidence_calibrator.py` | Python | Calibrates Bayesian confidence fusion & calculates risk scores | `judge_agent.py` | `judge_agent.py`, `app.py` | Yes | No | Yes | No | 107 | Confidence calculation & risk severity |
| `config.py` | `agents/judge_agent/config.py` | Python | Dataclass storing Judge Agent weights & thresholds | `judge_agent.py`, `decision_engine.py`, `calibrator.py` | All judge files | Yes | No | Yes | No | 41 | Settings storage |
| `decision_engine.py` | `agents/judge_agent/decision_engine.py` | Python | Rule-based engine deciding `ACCEPT`, `CORRECT`, `VERIFY_AGAIN`, `REJECT`, `ABSTAIN` | `judge_agent.py` | `app.py` | Yes | No | Yes | No | 130 | Action matrix evaluation & Corrector payload |
| `judge_agent.py` | `agents/judge_agent/judge_agent.py` | Python | Main Judge Agent orchestrator class | `app.py`, `simulator.py`, `test_judge.py` | CLI / Server | Yes | No | Yes | No | 137 | Multi-agent signal evaluation pipeline |
| `nli_engine.py` | `agents/judge_agent/nli_engine.py` | Python | DeBERTa-v3 cross-encoder NLI engine with zero-dependency heuristic fallback | `judge_agent.py` | `judge_agent.py` | Yes | No | Yes | Fallback heuristic | 133 | Claim-evidence entailment scoring |
| `simulator.py` | `agents/judge_agent/simulator.py` | Python | Benchmark simulator evaluating synthetic scenarios | `app.py` | CLI / Server | Yes | No | Yes | No | 122 | Benchmark execution |
| `test_judge.py` | `agents/judge_agent/test_judge.py` | Python | Unit and integration specs for Judge Agent | Pytest | Test runner | Yes | No | Yes | No | 75 | Test assertions |
| `JUDGE_AGENT_TECHNICAL_DOCUMENTATION.md` | `agents/judge_agent/JUDGE_AGENT_TECHNICAL_DOCUMENTATION.md` | Markdown | Complete technical documentation for Judge Agent | Developers | PDF exporter | Yes | No | Yes | No | 215 | Subsystem technical reference |
| `container.py` | `agents/verifier_agent/container.py` | Python | Dependency Injection container linking verifier services | `api/pipeline.py`, `api/main.py` | API server | Yes | No | Yes | No | 56 | Service initialization & lifecycle |
| `version.py` | `agents/verifier_agent/version.py` | Python | Version constants (`VERIFIER_VERSION = "2.0.0"`) | `api/main.py`, `container.py` | System status | Yes | No | Yes | No | 12 | Version string provider |
| `registry.py` | `agents/verifier_agent/adapters/registry.py` | Python | Central adapter registry mapping domain keys to adapter instances | `container.py`, `pipeline.py` | All adapters | Yes | No | Yes | Auto-stub fallback | 105 | Adapter registration & lookup |
| `domain_proxy.py` | `agents/verifier_agent/adapters/domain_proxy.py` | Python | Dynamic proxy routing requests to specific domain adapters | `registry.py` | Registry | Yes | No | Yes | No | 35 | Dynamic adapter invocation |
| `healthcare.py` | `agents/verifier_agent/adapters/healthcare.py` | Python | Live adapter searching PubMed/PMC, OpenFDA, and ClinicalTrials.gov | `registry.py` | Pipeline | Yes | No | Yes | No | 215 | Live medical evidence search |
| `cybersecurity.py` | `agents/verifier_agent/adapters/cybersecurity.py` | Python | Live adapter searching NVD CVE API, CISA KEV, and MITRE ATT&CK | `registry.py` | Pipeline | Yes | No | Yes | No | 185 | Security threat evidence search |
| `finance.py` | `agents/verifier_agent/adapters/finance.py` | Python | Live adapter searching SEC EDGAR, World Bank API, Alpha Vantage | `registry.py` | Pipeline | Yes | No | Yes | No | 160 | Financial filing & market evidence search |
| `ai_research.py` | `agents/verifier_agent/adapters/ai_research.py` | Python | Live adapter searching arXiv REST API, Semantic Scholar, Crossref | `registry.py` | Pipeline | Yes | No | Yes | No | 165 | Scientific paper search |
| `legal_general.py` | `agents/verifier_agent/adapters/legal_general.py` | Python | Adapter searching Wikipedia Legal API & Curated Indian Legal Acts | `registry.py` | Pipeline | Yes | No | Yes | Curated act map | 130 | Legal act & concept search |
| `general.py` | `agents/verifier_agent/adapters/general.py` | Python | Fallback general adapter searching Wikipedia REST API | `registry.py` | Pipeline | Yes | No | Yes | No | 60 | General knowledge search |
| `stub_adapter.py` | `agents/verifier_agent/adapters/stub_adapter.py` | Python | Lightweight stub adapter for 18 secondary domains | `registry.py` | Registry | Yes | Yes | Partial | Returns empty list | 40 | Domain stub handler |
| `mock_adapter.py` | `agents/verifier_agent/adapters/mock_adapter.py` | Python | Offline fixture adapter active when `MOCK_MODE=true` | `registry.py`, `tests` | Test suite | Yes | No | Yes | Mock JSON return | 65 | Offline test support |
| `aggregator.py` | `agents/verifier_agent/aggregation/aggregator.py` | Python | Flattens, deduplicates, and sorts retrieved multi-source passages | `pipeline.py` | Pipeline | Yes | No | Yes | No | 35 | Passage aggregation |
| `duplicate_remover.py` | `agents/verifier_agent/aggregation/duplicate_remover.py` | Python | Jaccard overlap deduplication engine (85% overlap threshold) | `aggregator.py` | Aggregator | Yes | No | Yes | No | 60 | Passage deduplication |
| `evidence_merger.py` | `agents/verifier_agent/aggregation/evidence_merger.py` | Python | Round-robin interleaving for multi-source evidence diversity | `aggregator.py` | Aggregator | Yes | No | Yes | No | 42 | Multi-source evidence interleaving |
| `main.py` | `agents/verifier_agent/api/main.py` | Python | FastAPI app exposing `/verify`, `/health`, `/domains`, `/metrics` | Uvicorn server | External clients | Yes | No | Yes | No | 150 | HTTP API routing & server setup |
| `pipeline.py` | `agents/verifier_agent/api/pipeline.py` | Python | 9-Stage Verification Pipeline Orchestrator | `main.py` | API server | Yes | No | Yes | No | 265 | Pipeline stage execution |
| `sqlite_cache.py` | `agents/verifier_agent/cache/sqlite_cache.py` | Python | Persistent SQLite cache engine (`verification_cache.db`) | `container.py`, `pipeline.py` | Pipeline | Yes | No | Yes | No | 155 | Cache read/write & TTL management |
| `claim_decomposer.py` | `agents/verifier_agent/claims/claim_decomposer.py` | Python | Splits compound sentences into atomic sub-claims | `pipeline.py` | Pipeline | Yes | No | Yes | No | 35 | Conjunction-based claim splitting |
| `claim_normalizer.py` | `agents/verifier_agent/claims/claim_normalizer.py` | Python | Cleans whitespace, unicode, filler words, and numbers | `pipeline.py` | Pipeline | Yes | No | Yes | No | 30 | Claim text normalization |
| `claim_merger.py` | `agents/verifier_agent/claims/claim_merger.py` | Python | Merges sub-claim evidence reports into unified output | `pipeline.py` | Pipeline | Yes | No | Yes | No | 60 | Evidence report aggregation |
| `settings.py` | `agents/verifier_agent/config/settings.py` | Python | Pydantic BaseSettings loading `.env` variables | `container.py` | Entire verifier | Yes | No | Yes | No | 30 | Global settings provider |
| `credibility.yaml` | `agents/verifier_agent/config/credibility.yaml` | YAML | Source reliability authority weights per domain | `source_reliability.py` | Scorers | Yes | No | Yes | No | 22 | Source credibility config |
| `domain_intelligence.yaml` | `agents/verifier_agent/config/domain_intelligence.yaml` | YAML | 35-Domain intelligence taxonomy mapping HuggingFace models & APIs | `domain_intelligence.py` | Router / Models | Yes | No | Yes | No | 620 | Domain taxonomy definition |
| `retry.yaml` | `agents/verifier_agent/config/retry.yaml` | YAML | HTTP retry & exponential backoff parameters per adapter | `utils/http_client.py` | HTTP Client | Yes | No | Yes | No | 25 | Network retry policy |
| `generator.py` | `agents/verifier_agent/explanations/generator.py` | Python | Natural language justification generator for verdicts | `pipeline.py` | Pipeline | Yes | No | Yes | No | 76 | Explanation text generation |
| `citation_formatter.py` | `agents/verifier_agent/formatters/citation_formatter.py` | Python | Standardized citation string generator (APA / IEEE / Vancouver) | `pipeline.py` | Pipeline | Yes | No | Yes | No | 52 | Citation string formatting |
| `response_formatter.py` | `agents/verifier_agent/formatters/response_formatter.py` | Python | Builds standardized `VerifierOutputV2` Pydantic payload | `pipeline.py` | Pipeline | Yes | No | Yes | No | 63 | Response schema construction |
| `collector.py` | `agents/verifier_agent/metrics/collector.py` | Python | Prometheus-compatible in-memory metric counter & histogram | `container.py`, `pipeline.py` | API server | Yes | No | Yes | No | 103 | Latency & pass/fail metric tracking |
| `performance.py` | `agents/verifier_agent/metrics/performance.py` | Python | Latency breakdown timer context manager per pipeline stage | `pipeline.py` | Pipeline | Yes | No | Yes | No | 54 | Stage timing measurement |
| `domain_intelligence.py` | `agents/verifier_agent/models/domain_intelligence.py` | Python | Python interface loading `domain_intelligence.yaml` taxonomy | `models/model_router.py` | Router | Yes | No | Yes | No | 187 | Taxonomy lookup & model selection |
| `model_manager.py` | `agents/verifier_agent/models/model_manager.py` | Python | Singleton managing lazy loading/unloading of HuggingFace models | `container.py`, `nli`, `rerankers` | Pipeline | Yes | No | Yes | Fallback if uninstalled | 100 | PyTorch & Transformer lifecycle |
| `model_router.py` | `agents/verifier_agent/models/model_router.py` | Python | Selects optimal embedding/NLI models per domain | `pipeline.py` | Pipeline | Yes | No | Yes | No | 86 | Model routing logic |
| `research_companion.py` | `agents/verifier_agent/models/research_companion.py` | Python | Interface executing domain research tasks | `benchmarks` | Benchmarks | Yes | No | Yes | No | 105 | Research helper functions |
| `entailment.py` | `agents/verifier_agent/nli/entailment.py` | Python | DeBERTa-v3 MNLI inference engine with heuristic fallback | `pipeline.py` | Pipeline | Yes | No | Yes | Heuristic fallback | 142 | Claim-evidence NLI classification |
| `evaluation_generator.py` | `agents/verifier_agent/reports/evaluation_generator.py` | Python | Generates Markdown evaluation reports from benchmark runs | `benchmarks/runner.py` | Benchmarks | Yes | No | Yes | No | 44 | Report document generation |
| `cross_encoder.py` | `agents/verifier_agent/rerankers/cross_encoder.py` | Python | Reranks retrieved passages using Cross-Encoder models | `pipeline.py` | Pipeline | Yes | No | Yes | Fallback if uninstalled | 71 | Passage relevance reranking |
| `sparse.py` | `agents/verifier_agent/retrievers/sparse.py` | Python | BM25 sparse keyword retriever engine | `retrievers/hybrid.py` | Hybrid Retriever | Yes | No | Yes | No | 50 | Keyword BM25 retrieval |
| `dense.py` | `agents/verifier_agent/retrievers/dense.py` | Python | SentenceTransformers + FAISS dense embedding retriever | `retrievers/hybrid.py` | Hybrid Retriever | Yes | No | Yes | Fallback if FAISS missing | 88 | Vector similarity retrieval |
| `hybrid.py` | `agents/verifier_agent/retrievers/hybrid.py` | Python | Reciprocal Rank Fusion (RRF) retriever combining BM25 + FAISS | `pipeline.py` | Pipeline | Yes | No | Yes | No | 92 | RRF hybrid passage retrieval |
| `domain_validator.py` | `agents/verifier_agent/routers/domain_validator.py` | Python | Validates request domain against taxonomy and maps aliases | `pipeline.py` | Pipeline | Yes | No | Yes | No | 69 | Domain validation & routing |
| `query_expander.py` | `agents/verifier_agent/routers/query_expander.py` | Python | Expands queries using domain synonym JSON files | `pipeline.py` | Pipeline | Yes | No | Yes | No | 61 | Synonym & term query expansion |
| `models.py` | `agents/verifier_agent/schemas/models.py` | Python | Pydantic v2 schemas (`VerifierInputV2`, `VerifierOutputV2`) | Entire codebase | Pipeline & API | Yes | No | Yes | No | 113 | Input/Output contract definitions |
| `conflict_resolver.py` | `agents/verifier_agent/scorers/conflict_resolver.py` | Python | 2:1 Contradiction-to-Support conflict resolution engine | `evidence_scorer.py` | Evidence Scorer | Yes | No | Yes | No | 73 | Conflicting evidence arbitration |
| `evidence_scorer.py` | `agents/verifier_agent/scorers/evidence_scorer.py` | Python | Calculates support, contradiction, trust scores, and verdict | `pipeline.py` | Pipeline | Yes | No | Yes | No | 83 | Claim support & verdict scoring |
| `source_reliability.py` | `agents/verifier_agent/scorers/source_reliability.py` | Python | Evaluates domain credibility scores from `credibility.yaml` | `evidence_scorer.py` | Evidence Scorer | Yes | No | Yes | No | 61 | Source credibility weighting |
| `convert_md_to_pdf.py` | `agents/verifier_agent/scratch/convert_md_to_pdf.py` | Python | Conversion utility rendering Markdown & Mermaid JS to PDF via Headless Chrome | Utility | Tooling | Yes | No | Yes | No | 187 | Documentation PDF compilation |
| `run_runtime_validation.py` | `agents/verifier_agent/scratch/run_runtime_validation.py` | Python | Live verification test runner validating pipeline execution | Utility | Validation | Yes | No | Yes | No | 182 | Automated system smoke test |
| `validation_output.txt` | `agents/verifier_agent/scratch/validation_output.txt` | Text | Saved stdout output log of the live runtime validation execution | Utility | Logs | Yes | No | Yes | No | 1565 | Empirical execution proof log |
| `test_adapters.py` | `agents/verifier_agent/tests/test_adapters.py` | Python | Pytest suite for live adapters | Pytest | Test runner | Yes | No | Yes | No | 81 | Adapter unit tests |
| `test_claims.py` | `agents/verifier_agent/tests/test_claims.py` | Python | Pytest suite for claim decomposer & normalizer | Pytest | Test runner | Yes | No | Yes | No | 31 | Claim processing tests |
| `test_domain_intelligence.py` | `agents/verifier_agent/tests/test_domain_intelligence.py` | Python | Pytest suite for domain intelligence taxonomy | Pytest | Test runner | Yes | No | Yes | No | 58 | Domain taxonomy tests |
| `test_health.py` | `agents/verifier_agent/tests/test_health.py` | Python | Pytest suite for API `/health` endpoints | Pytest | Test runner | Yes | No | Yes | No | 26 | System health tests |
| `test_official_integrations.py` | `agents/verifier_agent/tests/test_official_integrations.py` | Python | Integration test suite calling external APIs (PubMed, OpenFDA) | Pytest | Test runner | Yes | No | Yes | No | 95 | Live network integration tests |
| `test_registry.py` | `agents/verifier_agent/tests/test_registry.py` | Python | Pytest suite for adapter registry & stubs | Pytest | Test runner | Yes | No | Yes | No | 30 | Adapter registry tests |
| `test_research_notebooks.py` | `agents/verifier_agent/tests/test_research_notebooks.py` | Python | Pytest suite asserting existence & structure of 35 research notebooks | Pytest | Test runner | Yes | No | Yes | No | 62 | Notebook structure tests |
| `test_verify_smoke.py` | `agents/verifier_agent/tests/test_verify_smoke.py` | Python | End-to-end smoke test for `/verify` endpoint | Pytest | Test runner | Yes | No | Yes | No | 55 | Verification pipeline smoke test |
| `logging.py` | `agents/verifier_agent/utils/logging.py` | Python | Structured JSON logging module | Entire codebase | Utilities | Yes | No | Yes | No | 55 | JSON structured logging |
| `http_client.py` | `agents/verifier_agent/utils/http_client.py` | Python | Resilient async HTTP client with exponential backoff & retry | All adapters | Adapters | Yes | No | Yes | No | 97 | Async network client |
| `async_executor.py` | `agents/verifier_agent/utils/async_executor.py` | Python | Parallel async task gatherer with timeouts and fallback handling | All adapters | Adapters | Yes | No | Yes | No | 54 | Async concurrency executor |
| `health_checker.py` | `agents/verifier_agent/utils/health_checker.py` | Python | Utility checking health status of live adapters | `api/main.py` | API server | Yes | No | Yes | No | 54 | Adapter health monitor |
| `verification_cache.db` | `agents/verifier_agent/verification_cache.db` | SQLite DB | Persistent SQLite database caching claim verification results | `sqlite_cache.py` | Cache layer | Yes | No | Yes | Binary data | N/A | Persistent cache storage |
| `generate_research_notebooks.py` | `agents/verifier_agent/research/generate_research_notebooks.py` | Python | Script that auto-generates 35 domain research notebooks | Benchmarks | Research | Yes | No | Yes | No | 216 | Notebook generation tool |
| `ctg-oas-v2.yaml` | `ctg-oas-v2.yaml` | YAML | ClinicalTrials.gov official OpenAPI v2 Specification | `adapters/healthcare.py` | Adapters | Yes | No | Yes | No | 2836 | API schema specification |

---

# 3. Module Relationships & Dependency Map

```
                          [ LLM Draft Output ]
                                   │
                                   ▼
                         ┌───────────────────┐
                         │  Detector Agent   │  (Identifies Suspicious Claims & Perplexity)
                         └─────────┬─────────┘
                                   │  suspicious_claims[]
                                   ▼
                         ┌───────────────────┐
                         │  Verifier Agent   │  (Port 8002)
                         └─────────┬─────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Domain Validator │     │ Query Expander   │     │ SQLite Cache     │
└─────────┬────────┘     └─────────┬────────┘     └─────────┬────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   ▼
                         ┌───────────────────┐
                         │ Adapter Registry  │  (Domain Routing: Healthcare, Cyber, etc.)
                         └─────────┬─────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Live API Adapters│     │ BM25 Retriever   │     │ FAISS Dense Vector│
└─────────┬────────┘     └─────────┬────────┘     └─────────┬────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   ▼
                         ┌───────────────────┐
                         │  RRF Hybrid Merg  │  (Reciprocal Rank Fusion)
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │  Cross-Encoder    │  (Re-ranking Passage Relevance)
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │  DeBERTa NLI      │  (Entailment vs Contradiction Classification)
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Evidence Scorer   │  (2:1 Conflict Resolution & Recency Decay)
                         └─────────┬─────────┘
                                   │  claim_evidence_pairs[]
                                   ▼
                         ┌───────────────────┐
                         │    Judge Agent    │  (Port 8003)
                         └─────────┬─────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  ACCEPT Verdict  │     │ CORRECT Verdict  │     │  REJECT Verdict  │
└──────────────────┘     └─────────┬────────┘     └─────────┬────────┘
                                   │  corrector_payload{}   │
                                   ▼                        ▼
                         ┌───────────────────┐    ┌───────────────────┐
                         │  Corrector Agent  │    │   Memory Agent    │
                         └───────────────────┘    └───────────────────┘
```

---

# 4. Verifier Agent Detailed Analysis

The **Verifier Agent** is the core evidence-retrieval and fact-checking engine of HalluciGuard.

### Subsystem Components

1. **API Server (`api/main.py`)**: Built with FastAPI. Listens on port `8002`. Exposes `/verify`, `/health`, `/domains`, `/metrics`.
2. **Pipeline Orchestrator (`api/pipeline.py`)**: Executes a sequential 9-stage verification flow:
   - Stage 1: Domain Validation (`routers/domain_validator.py`)
   - Stage 2: Claim Decomposition & Normalization (`claims/claim_decomposer.py`, `claims/claim_normalizer.py`)
   - Stage 3: Query Expansion (`routers/query_expander.py`)
   - Stage 4: Multi-Source Evidence Retrieval (`adapters/registry.py`)
   - Stage 5: Aggregation & Deduplication (`aggregation/aggregator.py`, `aggregation/duplicate_remover.py`)
   - Stage 6: Cross-Encoder Reranking (`rerankers/cross_encoder.py`)
   - Stage 7: Natural Language Inference (`nli/entailment.py`)
   - Stage 8: Evidence & Conflict Scoring (`scorers/evidence_scorer.py`, `scorers/conflict_resolver.py`)
   - Stage 9: Citation & Response Formatting (`formatters/citation_formatter.py`, `formatters/response_formatter.py`)
3. **Domain Adapters (`adapters/`)**: Live adapters for Healthcare, Cybersecurity, Finance, AI Research, Legal, General. Stubs handle 18 secondary domains.
4. **Caching Subsystem (`cache/sqlite_cache.py`)**: SQLite cache (`verification_cache.db`) storing query hashes and evidence items (TTL: 24h).
5. **Model Lifecycle Manager (`models/model_manager.py`)**: Singleton managing lazy loading/unloading of HuggingFace models.
6. **Domain Intelligence Taxonomy (`models/domain_intelligence.py`)**: Operates on `config/domain_intelligence.yaml`, supporting 35 domain classifications.
7. **Scoring Engine (`scorers/`)**: Implements Recency Decay, Source Reliability Weighting (`config/credibility.yaml`), and 2:1 Contradiction-to-Support arbitration.

---

# 5. Integrated API Inventory

| API Name | Domain | Purpose | Authentication | Implementation File | Status | Endpoints Used |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PubMed / PMC E-utilities** | Healthcare | Medical research literature search | None (Free Tier) | `adapters/healthcare.py` | Active & Tested | `esearch.fcgi`, `esummary.fcgi` |
| **openFDA Drug Label API** | Healthcare | FDA approved drug indications & labels | Optional API Key | `adapters/healthcare.py` | Active & Tested | `/drug/label.json` |
| **ClinicalTrials.gov v2** | Healthcare | Clinical trial protocols & outcomes | None (Free Tier) | `adapters/healthcare.py` | Active & Tested | `/api/v2/studies` |
| **NVD CVE API 2.0** | Cybersecurity | Common Vulnerabilities and Exposures | Optional API Key | `adapters/cybersecurity.py` | Active & Tested | `/rest/json/cves/2.0` |
| **CISA KEV Catalog** | Cybersecurity | Known Exploited Vulnerabilities catalog | None (Free Tier) | `adapters/cybersecurity.py` | Active & Tested | `/sites/default/files/feeds/...` |
| **MITRE ATT&CK STIX 2.1** | Cybersecurity | Enterprise attack techniques & tactics | None (GitHub JSON) | `adapters/cybersecurity.py` | Active & Tested | GitHub raw STIX 2.1 JSON |
| **SEC EDGAR Search API** | Finance | Corporate filings (10-K, 10-Q, 8-K) | User-Agent Header | `adapters/finance.py` | Active & Tested | `/LATEST/search-index` |
| **World Bank Indicators API** | Finance | Global GDP & economic indicators | None (Free Tier) | `adapters/finance.py` | Active & Tested | `/v2/country/all/indicator/...` |
| **Alpha Vantage API** | Finance | Stock market quotes & financial overview | API Key (`ALPHA_VANTAGE_KEY`) | `adapters/finance.py` | Active (Key-dependent) | `/query?function=OVERVIEW` |
| **arXiv REST API** | AI Research | Computer science & AI research preprints | None (Free Tier) | `adapters/ai_research.py` | Active & Tested | `/api/query` |
| **Semantic Scholar API** | AI Research | Academic paper citations & abstracts | None (Free Tier) | `adapters/ai_research.py` | Active & Tested | `/graph/v1/paper/search` |
| **Crossref Metadata API** | AI Research | DOI & academic publisher metadata | User-Agent Mailto | `adapters/ai_research.py` | Active & Tested | `/works` |
| **Wikipedia REST API** | General / Legal | General knowledge & legal concept summaries | User-Agent Header | `adapters/general.py`, `legal_general.py` | Active & Tested | `/w/api.php` |

---

# 6. Hugging Face Model Inventory

| Model ID | Domain | Purpose | Implementation File | Registered | Loaded | Actively Used | Fallback Available |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `microsoft/deberta-v3-base-mnli` | General NLI | Premise-hypothesis entailment/contradiction classification | `nli/entailment.py`, `models/model_manager.py` | Yes | Yes (Lazy) | Yes | Yes (Heuristic NLI) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | General Reranking | Cross-encoder passage relevance reranking | `rerankers/cross_encoder.py`, `models/model_manager.py` | Yes | Yes (Lazy) | Yes | Yes (Identity Rerank) |
| `sentence-transformers/all-MiniLM-L6-v2` | General Dense Search | Sentence embeddings for FAISS vector search | `retrievers/dense.py`, `models/model_manager.py` | Yes | Yes (Lazy) | Yes | Yes (BM25 Sparse Only) |
| `facebook/bart-large-mnli` | Zero-Shot NLI | Secondary NLI classifier fallback | `models/model_manager.py` | Yes | On Demand | Registered | Yes |
| `BAAI/bge-small-en-v1.5` | AI / ML Embeddings | Domain-specific dense embeddings | `config/domain_intelligence.yaml` | Yes | In Notebooks | Registered | Yes |
| `Michiyasunaga/BioLinkBERT-base` | Healthcare | Biomedical literature embeddings | `config/domain_intelligence.yaml` | Yes | In Notebooks | Registered | Yes |
| `jackaduma/SecBERT` | Cybersecurity | Security threat text embeddings | `config/domain_intelligence.yaml` | Yes | In Notebooks | Registered | Yes |
| `ProsusAI/finbert` | Finance | Financial sentiment & text embeddings | `config/domain_intelligence.yaml` | Yes | In Notebooks | Registered | Yes |
| `nlpaueb/legal-bert-base-uncased` | Legal | Legal act & text embeddings | `config/domain_intelligence.yaml` | Yes | In Notebooks | Registered | Yes |
| `Separ/SciBERT` | Scientific Research | Multi-discipline scientific embeddings | `config/domain_intelligence.yaml` | Yes | In Notebooks | Registered | Yes |

---

# 7. System Implementation Status Summary

| Category / Component | Implementation Status | Completion % | Notes / Detail |
| :--- | :--- | :--- | :--- |
| **Verifier Agent Core Engine** | 🟢 Complete & Verified | **100%** | Full 9-stage pipeline running on Port 8002 |
| **Judge Agent Subsystem** | 🟢 Complete & Verified | **100%** | Decision engine, calibrator, and API running on Port 8003 |
| **Domain Router & Taxonomy** | 🟢 Complete & Verified | **100%** | 35-domain intelligence taxonomy active |
| **Model Registry & Manager** | 🟢 Complete & Verified | **100%** | Lazy loading PyTorch singleton |
| **Hybrid Retrieval (RRF)** | 🟢 Complete & Verified | **100%** | Reciprocal Rank Fusion of BM25 + FAISS |
| **Dense Retrieval (FAISS)** | 🟢 Complete & Verified | **100%** | SentenceTransformers embedding search |
| **Sparse Retrieval (BM25)** | 🟢 Complete & Verified | **100%** | BM25Okapi keyword search |
| **Cross-Encoder Reranker** | 🟢 Complete & Verified | **100%** | MS-MARCO MiniLM passage reranker |
| **Natural Language Inference** | 🟢 Complete & Verified | **100%** | DeBERTa-v3 MNLI + Heuristic fallback |
| **Trust Score & Arbitration** | 🟢 Complete & Verified | **100%** | 2:1 Contradiction-to-Support resolution |
| **Citation Engine** | 🟢 Complete & Verified | **100%** | Standardized APA/IEEE/Vancouver generator |
| **Healthcare Domain Adapter** | 🟢 Complete & Verified | **100%** | PubMed, OpenFDA, ClinicalTrials.gov live APIs |
| **Cybersecurity Domain Adapter** | 🟢 Complete & Verified | **100%** | NVD CVE, CISA KEV, MITRE ATT&CK live APIs |
| **Finance Domain Adapter** | 🟢 Complete & Verified | **100%** | SEC EDGAR, World Bank, Alpha Vantage APIs |
| **AI Research Domain Adapter** | 🟢 Complete & Verified | **100%** | arXiv, Semantic Scholar, Crossref live APIs |
| **Legal Domain Adapter** | 🟢 Complete & Verified | **100%** | Wikipedia Legal + Curated Act repository |
| **General Knowledge Adapter** | 🟢 Complete & Verified | **100%** | Wikipedia REST API |
| **Secondary Domain Stubs** | 🟡 Stub Implemented | **100% (Stub)** | 18 secondary domain stub fallbacks |
| **Research Notebook Suite** | 🟢 Complete & Verified | **100%** | 35 domain research notebooks |
| **Detector Agent Subsystem** | 🔴 Pending Implementation | **10%** | Architecture spec complete (`README.md`); code unbuilt |
| **Corrector Agent Subsystem** | 🔴 Pending Implementation | **10%** | Architecture spec complete (`README.md`); code unbuilt |
| **Memory Agent Subsystem** | 🔴 Pending Implementation | **10%** | Architecture spec complete (`README.md`); code unbuilt |

---

# 8. Missing & Incomplete Components Relative to Architecture

1. **Detector Agent Implementation (`agents/detector_agent/`)**: Missing claim extraction, perplexity calculation, and FastAPI server (Port 8001).
2. **Corrector Agent Implementation (`agents/corrector_agent/`)**: Missing evidence-based text rewriter, diff engine, and FastAPI server (Port 8004).
3. **Memory Agent Implementation (`agents/memory_agent/`)**: Missing Knowledge Graph engine (Neo4j/NetworkX), vector store (ChromaDB), and FastAPI server (Port 8005).
4. **Live API Adapters for Secondary Domains**: 18 secondary domains rely on `StubAdapter` or Wikipedia general fallback.
