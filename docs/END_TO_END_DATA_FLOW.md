# HalluciGuard End-to-End Data Flow

## Repository audit and source of truth

- `orchestration/api.py` exposes the backend FastAPI contract. `POST /verify` is canonical and `POST /api/v1/verify` is a compatibility alias; both call the same `run_verification` implementation.
- `services/base_llm_service.py` is the single Base LLM abstraction. It calls OpenRouter's OpenAI-compatible chat completions endpoint and returns typed generation metadata without exposing `OPENROUTER_API_KEY`.
- `orchestration/graph.py` is the canonical LangGraph. Active nodes are Generator, Supervisor, Detector, Verifier, and Memory. Judge and Corrector code remains in `agents/` but is not active.
- `orchestration/state.py` defines the shared `HalluciGuardState` used across graph nodes.
- `orchestration/interbus.py` defines the in-process structured inter-agent message bus stored in graph state.
- `agents/detector_agent/detector.py` remains the Detector source of truth. The graph calls `DetectorAgent.detect(user_query, draft_response)`.
- `agents/verifier_agent/api/pipeline.py` remains the Verifier source of truth. The graph passes the draft response as the claim text to `VerificationPipeline.verify(...)`.
- `agents/memory_agent/memory/memory_agent.py` remains the Memory Agent source of truth. The graph only builds `StoreFactRequest` objects for verifier reports whose verdict is `verified`.

The frontend repository could not be cloned in this environment because GitHub access returned `CONNECT tunnel failed, response 403`; frontend inspection and code changes are therefore blocked here.

## Product path

```text
User / Browser
  -> Next.js Frontend
  -> VerificationService
  -> HalluciGuardAdapter
  -> FastAPI POST /verify
  -> BaseLLMService
  -> OpenRouter /chat/completions
  -> Qwen3 draft response
  -> HalluciGuardState
  -> LangGraph Supervisor
  -> Detector
  -> Supervisor
  -> Verifier or Memory
  -> Supervisor
  -> Memory
  -> Structured API response
  -> Frontend debug/chat UI
```

## Transition-by-transition contract

| Step | Exact file | Class/function | Request object and important fields | Response/output | Failure behavior |
| --- | --- | --- | --- | --- | --- |
| Frontend UI to frontend service | Frontend repo unavailable here | `VerificationService.verify(...)` | `user_query`, `generation_mode`, optional `domain`/`request_id` | Adapter call | UI must not directly call `fetch`/`axios`; blocked from implementation here. |
| Frontend adapter to backend | Frontend repo unavailable here | `HalluciGuardAdapter` | `POST /verify` JSON body | Backend JSON contract | API errors should render visibly, not as green verified states. |
| API to graph | `orchestration/api.py` | `_run()` -> `run_verification(...)` | `VerificationRequest`: `user_query`, optional `llm_response`, `generation_mode`, history, domain, request ID | `HalluciGuardState` result | Unexpected runner crashes become HTTP 500; graph-level failures stay machine-readable in response. |
| Graph to Base LLM | `orchestration/graph.py` | `_generate_node()` | `HalluciGuardState.user_query`, `conversation_history`, `generation_mode` | `generation`, `draft_response`, `llm_response` | Failure sets `verification_status=generation_failed`, `terminal_status=failed`; Detector is not called with empty content. |
| Base LLM to OpenRouter | `services/base_llm_service.py` | `BaseLLMService.generate(...)` | OpenAI-compatible body: `model`, `temperature`, `messages`, optional configured `max_tokens` | `GenerationResult` with `usage` if provided | Missing key, timeout, network, HTTP, model unavailable, malformed/empty responses return `status=failed`; no fake answer. |
| Generator to bus | `orchestration/graph.py`, `orchestration/interbus.py` | `publish_message(...)` | `DRAFT_RESPONSE` payload with draft/model | `inter_agent_bus[]` event | Failures remain in `errors`/trace. |
| Supervisor routing | `orchestration/graph.py` | `_supervisor_node()`, `_supervisor_route()` | Current node, route, retry/failure state | Next node name | Supervisor only routes; it does not decide factual truth. |
| Detector | `orchestration/graph.py`, `agents/detector_agent/detector.py` | `_detector_node()`, `DetectorAgent.detect(...)` | Original `user_query` + actual `draft_response` | `detector`, risk, probability, confidence, next action | Detector failure is visible; no fabricated detector result in graph. |
| Detector to bus | `orchestration/graph.py` | `publish_message(...)` | `SUSPICIOUS_CLAIMS` for verifier or `DETECTOR_ACCEPT` for memory | Structured bus event | Routing remains auditable. |
| Verifier | `orchestration/graph.py`, `agents/verifier_agent/api/pipeline.py` | `_verifier_node()`, `VerificationPipeline.verify(...)` | `VerifierInputV2` with domain and draft response claim | `verifier`, `claims`, `evidence`, `nli_results` | Bounded retry via graph retry node; no fake verified output. |
| Verifier to bus | `orchestration/graph.py` | `publish_message(...)` | `VERIFICATION_RESULT` payload | Structured bus event | Failures remain in `errors`/trace. |
| Memory | `orchestration/graph.py`, `agents/memory_agent/memory/memory_agent.py` | `_memory_node()` | `StoreFactRequest` for `verdict == verified` only | `memory` count/stored facts | Memory failure becomes `partial_success` only after verification exists; unverified/degraded/failed claims are not stored as truth. |
| Memory to bus | `orchestration/graph.py` | `publish_message(...)` | `MEMORY_WRITE_RESULT` payload | Structured bus event | Failed memory writes emit failed bus status. |
| API response to frontend | `orchestration/api.py` | `_response()` | Final graph state | `execution_id`, `request_id`, `generation`, `draft_response`, agents, detector, verifier, memory, bus, trace, errors, retries, terminal status | Judge/Corrector are explicitly `not_executed`; secrets are never returned. |

## OpenRouter configuration

Set these values in a local backend environment or local `.env` file. Do not commit `.env`.

```bash
OPENROUTER_API_KEY=<server-side secret>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=qwen/qwen3-4b
OPENROUTER_TEMPERATURE=0.7
OPENROUTER_STRESS_TEMPERATURE=0.9
OPENROUTER_TIMEOUT_SECONDS=30
OPENROUTER_MAX_RETRIES=3
# Optional only when needed:
# OPENROUTER_MAX_TOKENS=1024
# OPENROUTER_HTTP_REFERER=https://your-app.example
# OPENROUTER_X_TITLE=HalluciGuard
```

The backend reports safe provenance (`provider=openrouter`, configured model slug, status, latency, usage when returned). It must never expose API keys, authorization headers, or local filesystem paths.

## Active and disabled agents

Active graph path:

```text
START -> GENERATE -> SUPERVISOR -> DETECTOR -> SUPERVISOR -> (VERIFIER -> SUPERVISOR | MEMORY) -> MEMORY -> END
```

Judge and Corrector are intentionally disabled and represented as:

```json
{
  "judge": {"enabled": false, "status": "not_executed"},
  "corrector": {"enabled": false, "status": "not_executed"}
}
```

## Testing categories

- Unit tests: BaseLLM retry/error parsing and graph route tests.
- Contract tests: FastAPI request/response shape and frontend TypeScript contracts.
- Model tests: OpenRouter smoke script and local DeBERTa NLI validation.
- Backend integration tests: API to LangGraph to Detector/Verifier/Memory.
- Real frontend E2E tests: browser to `VerificationService` to `HalluciGuardAdapter` to backend to OpenRouter and back to UI.

In this environment, real frontend and provider/model E2E validation are blocked by the GitHub/network tunnel returning 403 and lack of configured local `OPENROUTER_API_KEY` in the repository runtime.
