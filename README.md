# HalluciGuard 🛡️

### AI-Powered Multi-Agent Trust Layer for LLM Output Verification

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What is HalluciGuard?

**HalluciGuard** is a production-grade, multi-agent system that detects, verifies, judges, and corrects hallucinations in Large Language Model (LLM) outputs. It operates as a **trust layer** — sitting between an LLM's raw output and the end user — ensuring factual accuracy across 6+ specialized domains.

> *"Don't trust. Verify."*

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HalluciGuard Pipeline                        │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌───────┐   ┌──────────┐           │
│  │ Detector │──▶│ Verifier │──▶│ Judge │──▶│Corrector │           │
│  │  Agent   │   │  Agent   │   │ Agent │   │  Agent   │           │
│  └──────────┘   └──────────┘   └───────┘   └──────────┘           │
│       │              │             │             │                  │
│       └──────────────┴─────────────┴─────────────┘                 │
│                          │                                          │
│                    ┌─────▼─────┐                                   │
│                    │  Memory   │                                   │
│                    │  Agent    │                                   │
│                    └───────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## 🤖 The 5 Agents

| Agent | Port | Status | Role |
|-------|------|--------|------|
| 🔍 **Detector Agent** | 8001 | 🟡 Awaiting | Identifies suspicious claims in LLM outputs |
| ✅ **Verifier Agent** | 8002 | 🟢 Implemented | Retrieves evidence, scores claim support/contradiction |
| ⚖️ **Judge Agent** | 8003 | 🟡 Awaiting | Makes final accept/reject/flag decisions |
| ✏️ **Corrector Agent** | 8004 | 🟡 Awaiting | Rewrites hallucinated content with verified facts |
| 🧠 **Memory Agent** | 8005 | 🟡 Awaiting | Persistent knowledge graph and learning history |

## 📁 Project Structure

```
HalluciGuard/
├── README.md                          # This file
├── .gitignore
├── agents/
│   ├── __init__.py                    # Multi-agent package
│   ├── detector_agent/                # 🔍 Claim detection (🟡 awaiting)
│   │   ├── __init__.py
│   │   └── README.md
│   ├── verifier_agent/                # ✅ Evidence verification (🟢 done)
│   │   ├── __init__.py
│   │   ├── version.py
│   │   ├── container.py               # Dependency injection
│   │   ├── requirements.txt
│   │   ├── .env.example
│   │   ├── config/                    # Settings, credibility, retries
│   │   ├── schemas/                   # Pydantic v2 data models
│   │   ├── adapters/                  # 6 domain adapters + mock/stub
│   │   ├── claims/                    # Claim decomposition pipeline
│   │   ├── retrievers/                # BM25 + Dense + Hybrid (RRF)
│   │   ├── aggregation/               # Evidence aggregation + dedup
│   │   ├── rerankers/                 # Cross-encoder reranking
│   │   ├── nli/                       # DeBERTa NLI entailment
│   │   ├── scorers/                   # Evidence scoring + conflict resolution
│   │   ├── explanations/              # Human-readable explanations
│   │   ├── formatters/                # Citation + response formatting
│   │   ├── routers/                   # Domain validation + query expansion
│   │   ├── cache/                     # SQLite async caching
│   │   ├── metrics/                   # Performance tracking
│   │   ├── models/                    # ML model lifecycle manager
│   │   ├── utils/                     # Logging, HTTP client, health checks
│   │   ├── api/                       # FastAPI app + 8-stage pipeline
│   │   ├── tests/                     # Pytest test suite
│   │   ├── benchmarks/                # PubHealth, FEVER benchmarks
│   │   ├── reports/                   # Evaluation report generator
│   │   └── docs/                      # Agent documentation
│   ├── judge_agent/                   # ⚖️ Decision making (🟡 awaiting)
│   │   ├── __init__.py
│   │   └── README.md
│   ├── corrector_agent/               # ✏️ Fact correction (🟡 awaiting)
│   │   ├── __init__.py
│   │   └── README.md
│   └── memory_agent/                  # 🧠 Knowledge persistence (🟡 awaiting)
│       ├── __init__.py
│       └── README.md
└── docs/                              # Project-level documentation
    └── architecture/                  # Architecture diagrams
```

## 🚀 Quick Start (Verifier Agent)

```bash
# 1. Navigate to the verifier agent
cd agents/verifier_agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env

# 4. Run the server
uvicorn api.main:app --port 8002 --reload

# 5. Test it
curl -X POST http://localhost:8002/verify \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "test-001",
    "domain": "healthcare",
    "suspicious_claims": [{
      "claim_id": "c1",
      "text": "Metformin completely cures type 2 diabetes"
    }]
  }'
```

## 🌐 Domains Supported

| Domain | Sources | Status |
|--------|---------|--------|
| 🏥 Healthcare | PubMed, openFDA, ClinicalTrials.gov | ✅ Live |
| 🔒 Cybersecurity | NVD, MITRE ATT&CK, CISA KEV | ✅ Live |
| 💰 Finance | SEC EDGAR, World Bank, Alpha Vantage | ✅ Live |
| ⚖️ Legal | Wikipedia Legal, Curated Indian Acts | ✅ Live |
| 🤖 AI Research | arXiv, Semantic Scholar, Crossref | ✅ Live |
| 🌍 General | Wikipedia, Wikidata | ✅ Live |
| 📚 18+ Stubs | Programming, Science, Education, etc. | 🟡 Stub |

## 🧪 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/verify` | POST | Submit claims for verification |
| `/health` | GET | System health + model status |
| `/domains` | GET | List all supported domains with statistics |
| `/pipeline` | GET | Pipeline visualization + stage timings |
| `/metrics` | GET | Performance metrics and counters |

## 🏗️ Tech Stack

- **Runtime**: Python 3.11+ / FastAPI / Uvicorn
- **ML Models**: DeBERTa-v3 (NLI), MiniLM (embeddings), Cross-Encoder (reranking)
- **Retrieval**: BM25 + FAISS Dense + Reciprocal Rank Fusion
- **Cache**: SQLite (async, TTL-based)
- **Testing**: Pytest + httpx AsyncClient

## 👥 Contributing

Each agent has its own branch:
- `verifier-agent` — ✅ Complete
- `detector-agent` — 🟡 Open for implementation
- `judge-agent` — 🟡 Open for implementation
- `corrector-agent` — 🟡 Open for implementation
- `memory-agent` — 🟡 Open for implementation

**To contribute:**
1. Fork the repo
2. Create your agent branch: `git checkout -b <agent-name>`
3. Read the agent's `README.md` for specifications
4. Implement the agent following the existing patterns
5. Submit a PR

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

**Built with ❤️ by the HalluciGuard Team**
