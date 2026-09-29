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

The superseded `frontend/` copy is retained locally for rollback/reference but
is intentionally excluded from GitHub. This keeps the remote repository and
deployment documentation focused on the single active client.

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

## Two detector packages

There are two detectors in this repository. They are **not** interchangeable, and
only one is part of the product path.

| | `halluciguard_detector/` | `halluciguard_judge/` |
|---|---|---|
| Status | **Canonical.** Production. | Standalone, experimental / legacy. |
| Checkpoint | `artifacts/detector-best` (DeBERTa-v3-xsmall, 3-class) | own `checkpoints/` path, binary `hallucination_probability` |
| Class semantics | `SUPPORTED` / `CONTRADICTED` / `NOT_ENOUGH_INFO` | a single hallucination probability |
| Imported by | `orchestration/detector_bridge.py`, `orchestration/runtime_validation.py`, `services/llm_detector_service.py`, `services/llm_detector_verifier_service.py` | nothing outside its own package |
| Shared components | uses the shared Verifier claim decomposer, hybrid retriever and cross-encoder reranker | has its own copies |

Consequences to keep in mind:

- `halluciguard_judge` is retained for reference and experimentation. It is
  **not** on the production path, so it is not covered by the product
  integration tests and its output schema does not match
  `orchestration/schemas.py`.
- It is **not** to be deleted: it still carries a claim extractor and an LLM
  judge path that have no equivalent in the canonical package.
- Its degraded-mode handling was found to be broken and fixed here (see the
  fail-closed note in `halluciguard_judge/detector.py`): the first request
  reported a healthy model while later ones correctly reported degraded, because
  the code returned the classifier's *load success* rather than whether a
  fine-tuned checkpoint was actually in use.
- Anything new belongs in `halluciguard_detector`. Changes to
  `halluciguard_judge` should be treated as maintenance of a frozen subsystem.

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
