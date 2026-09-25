# Project structure

The repository preserves its working Python import layout. There is no backend `src/` migration because moving packages at finalization would risk breaking imports; `src/` exists where it is native and useful—in both Next.js frontends.

## Frontend

### `frontend-v2/` — active application

```text
frontend-v2/
├── src/
│   ├── app/            # Next.js App Router: layout, landing and chat routes
│   ├── components/     # reusable UI, chat, navigation and activity views
│   ├── lib/            # API client, auth/session helpers and utilities
│   └── views/          # composed page-level views
├── public/             # static media and visual assets
├── package.json        # Next.js 15, React 19 and UI dependencies
├── next.config.mjs
├── tailwind.config.js
└── vercel.json
```

The root `package.json` delegates deployment builds to `frontend-v2`.

### `frontend/` — retained legacy implementation

This directory remains tracked because it is a complete earlier client and may still be useful for comparison or rollback. The repository does not claim both are active production frontends.

## Backend

```text
orchestration/          LangGraph state machine, canonical contracts and product API
services/               hosted LLM routing, Claim Analyzer, evidence and n8n clients
halluciguard_detector/  trained reference-grounded detector package and standalone API
agents/
  verifier_agent/       retrieval, reranking, NLI, scoring and claim verdicts
  judge_agent/          release/correction/retry/reject/abstain policy
  corrector_agent/      targeted repair, validation, training and tests
  memory_agent/         cache, graph, vector, trust and pattern persistence
halluciguard_judge/     standalone experimental/legacy judge-detector package
artifacts/detector-best production Detector checkpoint and evaluation metadata
```

## APIs

| Surface | Entry point | Purpose |
|---|---|---|
| Product orchestration | `orchestration/api.py` | Authentication, history, health and end-to-end verification |
| Detector development API | `halluciguard_detector/api.py` | Health and grounded `/v1/detect` |
| Verifier development API | `agents/verifier_agent/api/main.py` | Verify, domains, pipeline and metrics |
| Corrector development API | `agents/corrector_agent/app/main.py` | `/correct` |
| Memory development API | `agents/memory_agent/api/main.py` | Store, recall, graph, vector, trust, patterns and metrics |
| Hugging Face entry | `app.py` | Gradio/Space-compatible application |

## Operations and validation

```text
n8n/       versioned retrieval workflow exports
scripts/   diagnostics, demonstrations, repair and certification helpers
tests/     cross-package regression tests
docs/      current architecture, operations, research and historical reports
.github/   CI workflows for orchestration and agent suites
```

Generated caches, runtime databases, logs, local environments, model caches, Next.js builds and editor state are ignored and remain local.
