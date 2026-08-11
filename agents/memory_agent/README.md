# 🧠 Memory Agent — HalluciGuard

## Status: ✅ Implemented (v1.0.0)

The Memory Agent is the **knowledge persistence layer** in the HalluciGuard pipeline. It stores verified facts, learns from hallucination patterns, tracks source reliability, and provides historical context to all other agents.

---

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Detector    │───▶│  Verifier    │───▶│    Judge     │
│  Agent       │    │  Agent       │    │    Agent     │
│  (port 8001) │    │  (port 8002) │    │  (port 8003) │
└──────────────┘    └──────┬───────┘    └──────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │   Memory Agent   │◄── All agents query this
                  │   (port 8005)    │    for historical context
                  └──────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │Knowledge │ │  Vector  │ │  SQLite  │
        │  Graph   │ │  Store   │ │  Cache   │
        │(NetworkX)│ │ (FAISS)  │ │ (aiosqlite)│
        └──────────┘ └──────────┘ └──────────┘
```

---

## What Was Built

### 5 Core Modules

| Module | File | Tech | What It Does |
|--------|------|------|-------------|
| **Knowledge Graph** | `knowledge_graph/graph.py` | NetworkX | Stores verified facts as entity-relationship triples. Supports CRUD, neighbor traversal, weight decay, eviction, JSON persistence. |
| **Verification Cache** | `cache/verification_cache.py` | SQLite + aiosqlite | SHA-256 keyed cache with TTL expiration. Tracks hit rates, supports domain-wide invalidation. |
| **Pattern Learner** | `patterns/pattern_learner.py` | SQLite | Classifies claims into 6 pattern types (temporal, numerical, statistical, causal, definition, entity). Tracks frequency and confidence per domain. |
| **Source Trust** | `trust/source_trust.py` | SQLite | Bayesian trust score evolution. Increases on correct evidence, decreases on incorrect. Full audit history. |
| **Vector Store** | `vector_store/faiss_store.py` | FAISS + sentence-transformers | Semantic search over all stored facts. 384-dim embeddings, cosine similarity, metadata filtering. |

### Infrastructure

| Component | File | Description |
|-----------|------|-------------|
| **Orchestrator** | `memory/memory_agent.py` | Ties all 5 modules together. Single `store_fact()` and `recall()` entry points. |
| **FastAPI App** | `api/main.py` | 14 HTTP endpoints on port 8005. CORS enabled. Lifespan management. |
| **DI Container** | `container.py` | Wires all subsystems. Module-level singletons via `get_xxx()` pattern. |
| **Config** | `config/settings.py` | Pydantic Settings with `.env` file support. 20+ configurable parameters. |
| **Schemas** | `schemas/models.py` | 16 Pydantic V2 models, 4 enums. Full request/response contracts. |
| **Entry Point** | `__main__.py` | `python -m agents.memory_agent` starts uvicorn. |

### Tests (75 passing)

| Test File | Tests | Covers |
|-----------|-------|--------|
| `test_knowledge_graph.py` | 17 | Entity CRUD, edge ops, persistence, stats, decay |
| `test_cache.py` | 8 | Set/get, TTL, upsert, invalidation, key generation |
| `test_trust.py` | 13 | Trust updates, bounds, history, decay, domain queries |
| `test_patterns.py` | 12 | Claim classification, pattern learning, queries |
| `test_memory_agent.py` | 8 | Integration: store, recall, cache, patterns |
| `test_api.py` | 17 | All 14 HTTP endpoints |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/store` | Store a verified fact (creates KG nodes, vectors, patterns, cache, trust updates) |
| `POST` | `/recall` | Semantic recall with graph context, patterns, and trust scores |
| `GET` | `/health` | Health check with component status and stats |
| `GET` | `/stats` | Full system statistics |
| `GET` | `/domains` | List supported domains |
| `POST` | `/save` | Persist all data to disk |
| `GET` | `/cache/check` | Check if a claim is cached |
| `DELETE` | `/cache/invalidate` | Remove a cached entry |
| `POST` | `/trust/update` | Manually update a source's trust score |
| `GET` | `/trust/{source_id}` | Get a source's trust record |
| `GET` | `/trust/domain/{domain}` | List all sources in a domain |
| `POST` | `/patterns/query` | Query hallucination patterns |
| `GET` | `/patterns/domain/{domain}` | Get pattern summary for a domain |
| `GET` | `/knowledge-graph/stats` | Knowledge graph statistics |
| `GET` | `/knowledge-graph/entity/{id}` | Get an entity node |
| `GET` | `/knowledge-graph/entity/{id}/neighbors` | Get entity neighbors |
| `GET` | `/vectors/search` | Semantic vector search |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/KushalMadhan3/HalluciGuard.git
cd HalluciGuard

# 2. Install dependencies
pip install -r agents/memory_agent/requirements.txt

# 3. Configure
copy agents\memory_agent\.env.example agents\memory_agent\.env

# 4. Test
cd agents/memory_agent
python -m pytest tests/ -v
cd ../..

# 5. Run
python -m uvicorn agents.memory_agent.api.main:app --host 127.0.0.1 --port 8005

# 6. Verify
curl http://127.0.0.1:8005/health
```

---

## Example Usage

### Store a hallucinated claim
```bash
curl -X POST http://127.0.0.1:8005/store \
  -H "Content-Type: application/json" \
  -d '{
    "claim_text": "The earth was invented in 1999 by NASA",
    "domain": "science",
    "verdict": "likely_hallucinated",
    "evidence": [{"source_id": "wiki", "title": "Earth formation"}],
    "source_ids": ["wiki"],
    "confidence": 0.05
  }'
```

### Recall knowledge
```bash
curl -X POST http://127.0.0.1:8005/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "earth formation history",
    "domain": "science",
    "top_k": 5
  }'
```

### Check source trust
```bash
curl http://127.0.0.1:8005/trust/wiki
```

---

## Directory Structure

```
agents/memory_agent/
├── __init__.py                 # Package docstring
├── __main__.py                 # uvicorn entry point
├── version.py                  # MEMORY_AGENT_VERSION = "1.0.0"
├── container.py                # DI wiring
├── .env.example                # Environment template
├── requirements.txt            # 12 dependencies
├── pytest.ini                  # Test config
├── README.md                   # This file
│
├── config/
│   └── settings.py             # Pydantic Settings (20+ params)
│
├── schemas/
│   └── models.py               # 16 Pydantic models, 4 enums
│
├── knowledge_graph/
│   └── graph.py                # NetworkX KG with persistence
│
├── cache/
│   └── verification_cache.py   # SQLite cache with TTL
│
├── patterns/
│   └── pattern_learner.py      # Hallucination pattern tracker
│
├── trust/
│   └── source_trust.py         # Bayesian trust evolution
│
├── vector_store/
│   └── faiss_store.py          # FAISS semantic search
│
├── memory/
│   └── memory_agent.py         # Orchestrator
│
├── api/
│   └── main.py                 # FastAPI app (14 endpoints)
│
└── tests/
    ├── test_knowledge_graph.py  # 17 tests
    ├── test_cache.py            # 8 tests
    ├── test_trust.py            # 13 tests
    ├── test_patterns.py         # 12 tests
    ├── test_memory_agent.py     # 8 tests
    └── test_api.py              # 17 tests
```

---

## What's Done vs What's Next

### ✅ Done
- [x] Knowledge Graph with NetworkX (entity CRUD, edges, persistence, eviction)
- [x] Verification Cache with SQLite (TTL, SHA-256 keys, hit tracking)
- [x] Pattern Learning (6 claim types, frequency/confidence tracking)
- [x] Source Trust Evolution (Bayesian updates, audit history, decay)
- [x] Vector Store with FAISS (semantic search, metadata filtering)
- [x] Memory Agent orchestrator (store + recall flows)
- [x] FastAPI HTTP interface (14 endpoints, port 8005)
- [x] DI Container and singleton patterns
- [x] Pydantic V2 schemas for all data contracts
- [x] pydantic-settings configuration with .env
- [x] 75 passing tests (unit + integration + API)
- [x] JSON persistence for knowledge graph

### 🔲 Next Steps (Future Work)
- [ ] **Wire into Verifier Agent** — Have the Verifier Agent call Memory Agent's `/recall` before running its 9-stage pipeline (cache-first verification)
- [ ] **Wire into Judge Agent** — Feed Memory Agent's pattern data into the Judge's decision engine
- [ ] **Neo4j migration** — Replace NetworkX with Neo4j for production-scale graph queries
- [ ] **ChromaDB migration** — Replace FAISS with ChromaDB for managed vector storage
- [ ] **Authentication** — Add API key auth for inter-agent communication
- [ ] **Rate limiting** — Protect endpoints from abuse
- [ ] **Monitoring** — Add Prometheus metrics endpoint
- [ ] **Docker** — Containerize with docker-compose for all 5 agents
- [ ] **CI/CD** — GitHub Actions for automated testing and deployment
- [ ] **Batch operations** — `/store/batch` endpoint for bulk fact ingestion
- [ ] **Graph analytics** — PageRank for entity importance, community detection for topic clustering
- [ ] **Temporal reasoning** — Track how facts change over time (versioned facts)
- [ ] **Cross-domain inference** — Detect when a fact in one domain contradicts another

---

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_AGENT_PORT` | `8005` | Server port |
| `KG_PERSISTENCE_PATH` | `data/knowledge_graph.json` | Graph save location |
| `KG_MAX_NODES` | `100000` | Max entities before eviction |
| `CACHE_DB_PATH` | `data/verification_cache.db` | Cache database |
| `CACHE_TTL` | `86400` | Cache expiry (seconds) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `VECTOR_STORE_PATH` | `data/vector_store` | FAISS index location |
| `PATTERN_DB_PATH` | `data/patterns.db` | Pattern database |
| `TRUST_DB_PATH` | `data/source_trust.db` | Trust database |
| `TRUST_PRIOR` | `0.5` | Default trust score |
| `TRUST_LEARNING_RATE` | `0.1` | How fast trust changes |
| `TRUST_DECAY_RATE` | `0.01` | Global trust decay rate |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Knowledge Graph | NetworkX (MultiDiGraph) |
| Vector Store | FAISS (IndexFlatIP) + sentence-transformers |
| Databases | SQLite via aiosqlite (async) |
| API Framework | FastAPI + uvicorn |
| Validation | Pydantic V2 |
| Configuration | pydantic-settings + python-dotenv |
| Testing | pytest + pytest-asyncio |
| Embeddings | all-MiniLM-L6-v2 (384 dimensions) |

---

## Branch

This code lives on the `memory-agent` branch. Create a PR to merge into `main`.
