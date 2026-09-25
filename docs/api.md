# API reference

This is a route inventory, not a replacement for generated OpenAPI. Run the relevant FastAPI app and open `/docs` for current request/response schemas.

## Product API — `orchestration.api:app`

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Service metadata |
| `GET` | `/health` | Runtime and dependency health |
| `POST` | `/verify` | Canonical end-to-end verification |
| `POST` | `/api/v1/verify` | Compatibility alias for verification |
| `POST` | `/auth/register` | Local account registration when enabled |
| `POST` | `/auth/login` | Credential login |
| `POST` | `/auth/google`, `/api/v1/auth/google` | Google identity login |
| `GET` | `/auth/me` | Current authenticated user |
| `POST` | `/auth/logout` | Session logout |
| `GET/POST/DELETE` | `/api/history` | User verification history |

Minimal query-only request:

```json
{
  "user_query": "Who founded Microsoft?",
  "generation_mode": "normal"
}
```

The response includes generation metadata, claim analysis, Detector, Verifier, Judge, optional correction/reverification, Memory, terminal status, trace and structured errors.

## Development APIs

### Detector

- `GET /health`
- `POST /v1/detect` with `user_query`, `draft_answer` and non-empty `evidence`.

### Verifier

- `POST /verify`
- `GET /health`, `/domains`, `/pipeline`, `/metrics`.

### Corrector

- `POST /correct` with the canonical correction request.

### Memory

Includes `/store`, `/store/batch`, `/recall`, fact update/delete, cache controls, trust, patterns, knowledge-graph, vector search, health, stats, domains, save and metrics routes.

## Authentication and CORS

Production deployments must configure `JWT_SECRET`, permitted `CORS_ORIGINS`, and any Google client ID. SQLite auth/history paths belong under ignored runtime `data/`. Never expose backend provider credentials to the browser.
