# HalluciGuard Verifier V2 — Operations Runbook

## 1. Environment Configuration

Create a `.env` file in the project root:

```env
# Optional API Keys (system works offline or with free endpoints where available)
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxx
OPENFDA_KEY=
NVD_API_KEY=
ALPHA_VANTAGE_KEY=

# Model Configurations
NLI_MODEL=cross-encoder/nli-deberta-v3-base
RERANKER_MODEL=BAAI/bge-reranker-large
EMBEDDING_MODEL=BAAI/bge-m3

# Feature Flags & Quality Gate Settings
MOCK_MODE=false
ALLOW_MODEL_DOWNLOADS=false
VERIFIER_CACHE_ENABLED=true
RELEVANCE_THRESHOLD=0.25
MIN_TOP_RELEVANCE=0.30
MIN_RELEVANT_PASSAGES=1
EVIDENCE_RELEVANCE_GATE=0.20
DEFAULT_RETRIEVAL_MODE=hybrid
```

---

## 2. CLI Usage & Verification Commands

### A. Single Claim Verification
```bash
# Default Hybrid Mode (Primary first, Tavily fallback if needed)
python scripts/verify_claim.py "Hyderabad is the capital of India."

# Primary Only Mode (No Tavily web search)
python scripts/verify_claim.py "The Eiffel Tower is located in London." --retrieval-mode primary_only

# Tavily Only Mode (Direct web search fallback)
python scripts/verify_claim.py "The Earth is flat." --retrieval-mode tavily_only

# Force Tavily Flag
python scripts/verify_claim.py "Java was created by James Gosling." --force-tavily

# Specify Domain Intelligence Profile
python scripts/verify_claim.py "Aspirin is used to relieve mild pain." --domain healthcare
python scripts/verify_claim.py "CVE-2021-44228 is associated with Log4Shell." --domain cybersecurity
```

### B. Interactive REPL Mode
```bash
python scripts/verify_claim.py -i
```

---

## 3. Running Automated Tests

```bash
# Run Full Verification Test Suite (160 tests)
pytest agents/verifier_agent/tests/ -v

# Run V2 Regression Suite
pytest agents/verifier_agent/tests/test_v2_regression_failures.py -v

# Run Retrieval Quality Gate Tests
pytest agents/verifier_agent/tests/test_retrieval_quality_gate.py -v
```

---

## 4. Running Benchmark Evaluation

```bash
# Execute Full Multi-Domain Benchmark
python scripts/benchmark_eval.py --output benchmark_results.json

# Limit to First N Claims
python scripts/benchmark_eval.py --limit 10
```

---

## 5. Starting the API Server

```bash
uvicorn agents.verifier_agent.main:app --host 0.0.0.0 --port 8002 --reload
```
API Documentation will be accessible at: `http://localhost:8002/docs`
