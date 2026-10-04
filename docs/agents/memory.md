# Memory Agent

The Memory Agent persists facts only after Judge acceptance. `agents/memory_agent/` combines a verification cache, knowledge graph, FAISS vector store, source-trust state and pattern learner.

## Persistence gate

- Only canonical claim reports with `VERIFIED` verdict are eligible.
- A non-`ACCEPT` Judge decision stores no factual claims.
- If correction occurred, only passed ReVerifier reports are considered.
- Batch storage isolates per-fact failure so one bad write does not erase all results.

Outcomes are `STORED`, `DUPLICATE`, `SKIPPED`, or `FAILED`, with counts and fact IDs where available. Duplicate detection avoids repeated entries. Runtime databases and vector indices live under ignored `data/` paths.

Memory is not automatically current truth. Stored facts can become stale, sources can change, and time-sensitive queries require fresh verification.

## What is actually persisted

`MemoryAgent.store_fact()` stores claim text, domain, verdict, operational confidence, timestamps and fact IDs. Graph claim/fact/source nodes and relationships, vector metadata, a verification-cache row, source-trust counters and eligible pattern records occupy separate stores.

Full evidence snippets, citation URLs and request/execution metadata carried in `StoreFactRequest` are **not** all durably retained by the current `store_fact()` implementation. Use the execution report for the original citations and handoffs; do not treat a generic cached evidence-count summary as the underlying evidence.

Default paths, relative to the process working directory, are `data/knowledge_graph.json`, `data/verification_cache.db`, `data/vector_store/{entries.json,index.faiss}`, `data/patterns.db` and `data/source_trust.db`. The standalone API reads path overrides from `.env`/environment. The validation runner instead uses `<report-directory>/runtime/` with `memory-cache.db`; these are different stores.

## Open and inspect

For strict read-only inspection, from the repository root:

```powershell
python scripts/inspect_memory.py --data-dir data
python scripts/inspect_memory.py --runtime-dir "<report-directory>/runtime" --show-text
```

The inspector reads saved JSON and SQLite in read-only mode. It does not load embeddings, expire cache entries, initialize stores, or repair mismatches. Claim text is opt-in because it can contain private user data. It does not automatically load `.env`; see the [inspection guide](../../agents/memory_agent/README.md) for path selection and limitations.

To open the actual service UI/API:

```powershell
python -m uvicorn agents.memory_agent.api.main:app --host 127.0.0.1 --port 8005
```

Open `http://127.0.0.1:8005/docs`. Configure API-key protection before exposing the service. Service startup/shutdown can initialize/save stores; recall/cache calls can update hit counts or remove expired rows, so this is not strict read-only viewing. The direct `/store` endpoint is not the production Judge gate.

## Known lifecycle limitations in this checkout

The current `update_fact()` does not propagate a change to both the linked claim node and fact top-level confidence; the separate quarantine-repair branch must not be assumed integrated here. `delete_fact()` can leave a linked claim node behind, and rebuilding vector indexes after loading saved entries needs additional validation. Cross-store transactions, complete citation persistence and concurrent production consistency are not established by the successful diagnostic read-back. Do not use the inspector to conceal or silently repair these limitations.
