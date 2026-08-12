# Detector → Verifier Production Integration

## Purpose

The Detector and Verifier remain independent services with their existing contracts. The Detector is the first-stage gate; the Verifier is invoked only when the Detector returns `HIGH` risk / `Verify`.

## Runtime flow

```text
User Query + LLM Response
          |
          v
   Detector /detect
          |
          v
 HaluEval DistilBERT
          |
          +-------------------+
          |                   |
     LOW / MEDIUM           HIGH
          |                   |
          v                   v
       ACCEPT          POST /verify
                              |
                              v
                    9-stage Verifier
                              |
                              v
                Evidence + NLI + verdict
```

## Existing endpoint preserved

`POST /detect` still performs Detector-only inference. It does not contact the Verifier. This preserves the Detector API contract.

## New orchestration endpoint

`POST /analyze`

Request:

```json
{
  "query_id": "req-001",
  "domain": "general",
  "user_query": "What is the capital of France?",
  "llm_response": "The capital of France is Tokyo."
}
```

Behavior:

- Detector runs first, always.
- `LOW` or `MEDIUM`: response ends at the Detector; `verifier_invoked=false`.
- `HIGH`: exactly one request is sent to the Verifier's `/verify` endpoint.
- The original user query is used for Detector inference, while the complete LLM response is submitted as the suspicious claim to the Verifier.
- A Verifier failure on a HIGH-risk response returns HTTP 503 instead of silently accepting the response.

## Verifier service configuration

The Detector reads:

```text
VERIFIER_AGENT_URL=http://127.0.0.1:8001
VERIFIER_AGENT_TIMEOUT_SECONDS=60
```

The Verifier must expose `POST /verify` using `VerifierInputV2` and return `VerifierOutputV2`.

## Final status mapping

| Verifier verdict | Integrated final status |
|---|---|
| `verified` | `VERIFIED` |
| `likely_hallucinated` | `LIKELY_HALLUCINATED` |
| `insufficient_evidence` | `INSUFFICIENT_EVIDENCE` |
| `mixed_evidence` | `INSUFFICIENT_EVIDENCE` |

The integration never converts missing evidence into a positive verification result.

## Running locally

Terminal 1 — Verifier:

```bash
cd agents/verifier_agent
uvicorn api.main:app --host 127.0.0.1 --port 8001
```

Terminal 2 — Detector:

```bash
uvicorn agents.detector_agent.app:app --host 127.0.0.1 --port 8000
```

Then call:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query_id":"demo-001",
    "domain":"general",
    "user_query":"What is the capital of France?",
    "llm_response":"The capital of France is Tokyo."
  }'
```

## Integration guarantees

1. Detector remains the mandatory first stage.
2. LOW/MEDIUM never invoke Verifier.
3. HIGH invokes Verifier once.
4. `/detect` remains Detector-only.
5. A failed HIGH-risk handoff is fail-closed with HTTP 503.
6. Detector and Verifier remain independently deployable.
7. No trained model weights are transferred over the integration boundary.
