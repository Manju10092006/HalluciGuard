# Memory Agent

The Memory Agent persists facts only after Judge acceptance. `agents/memory_agent/` combines a verification cache, knowledge graph, FAISS vector store, source-trust state and pattern learner.

## Persistence gate

- Only canonical claim reports with `VERIFIED` verdict are eligible.
- A non-`ACCEPT` Judge decision stores no factual claims.
- If correction occurred, only passed ReVerifier reports are considered.
- Batch storage isolates per-fact failure so one bad write does not erase all results.

Outcomes are `STORED`, `DUPLICATE`, `SKIPPED`, or `FAILED`, with counts and fact IDs where available. Duplicate detection avoids repeated entries. Runtime databases and vector indices live under ignored `data/` paths.

Memory is not automatically current truth. Stored facts can become stale, sources can change, and time-sensitive queries require fresh verification.
