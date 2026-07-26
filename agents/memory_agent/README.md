# 🧠 Memory Agent

## Status: 🟡 Awaiting Implementation

## Role
The Memory Agent is the **knowledge persistence layer** in the HalluciGuard pipeline. It stores verified facts, learns from patterns, and provides historical context.

## Architecture Position
```
All Agents ←→ [MEMORY AGENT] ←→ Knowledge Graph / Vector DB
```

## Responsibilities
1. **Knowledge Graph** — Store verified facts as a graph of entities and relationships
2. **Verification Cache** — Cache past results to speed up repeated claim verification
3. **Pattern Learning** — Track common hallucination patterns per domain
4. **Source Trust Evolution** — Update source reliability scores based on verification outcomes
5. **Cross-Session Memory** — Persist knowledge across agent restarts

## Getting Started
1. Review how the Verifier Agent caches results (`agents/verifier_agent/cache/`)
2. Design the knowledge graph schema
3. Implement your agent in this directory
4. Create a PR to the `memory-agent` branch

## Tech Stack Suggestions
- **Neo4j** or **NetworkX** for knowledge graph
- **ChromaDB** or **FAISS** for vector storage
- **SQLite** for structured cache
- **FastAPI** for HTTP interface (port 8005)

## Contact
Assigned to: [Your Name]
