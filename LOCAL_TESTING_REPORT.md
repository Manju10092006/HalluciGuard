# Local testing report — 2026-10-19

Isolated branch: `codex/local-validation-20261019`, base `80769fa`. Original checkout, dirty changes and deployment worktrees preserved. No package installation or deployment.

## Regression

Environment: existing Python 3.13, torch 2.13.0+cu126, transformers 5.2.0, Node 22.14.0, Next 15.5.22. Installed versions differ from historical requirements; actual execution was tested without speculative dependency changes. Existing frontend node_modules were reused through a local junction.

Set HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 and OMP_NUM_THREADS=4, then run:

`python -m pytest -q halluciguard_detector/tests orchestration/tests agents/verifier_agent/tests agents/memory_agent/tests agents/judge_agent/tests agents/corrector_agent/tests services/tests tests --tb=short -o faulthandler_timeout=60`

Result: **1709 passed, 14 skipped**, one existing Starlette/httpx deprecation warning, 280.04 seconds (`acceptance-regression.log`). Two final small concurrency/API aggregation regressions were added after this collection; their separate result will be recorded below.

`node --test frontend-v2/verification-contract.test.mjs`: **8 passed** (`frontend-contract-final.log`). These validate trust-state mapping with fixtures, not browser rendering.

Earlier offline baseline: 1391 passed, 14 skipped in 204.94 seconds. The first unrestricted run stalled during initialization and was stopped. Intermediate regressions exposed permissive fixtures; evidence and safer expectations were added without deleting coverage. Two overlapping later runs were stopped when free RAM fell below 300 MB; incomplete runs are not counted as passes.

## Model and data

`python -m halluciguard_detector.benchmark --json runtime-logs/detector-probes-final.json`: actual checkpoint, 8/8 code invariants, 3/8 labeled probes correct. Saved prediction metrics and dev-only threshold reproduction are in runtime-logs/recomputed-metrics.json and runtime-logs/dev-threshold-reproduction.json.

Ignored original processed datasets were audited read-only: runtime-logs/dataset-audit.json records IDs, counts, hashes, duplicates and split overlaps. Fresh inference uses scripts/local_detector_evaluation.py with the original processed directory, seed 20261019, excluding train/dev exact-pair overlap and within-test duplicates. Results are under runtime-logs/fresh-model-evaluation. No threshold fitting on test data.

## Actual backend

Start scripts/local_validation_server.py with --env-file pointing to the original .env and --port 8020. It loads actual provider configuration and isolates SQLite/KG/vector files under runtime-logs. The missing JWT secret is replaced only by a process-local random key for isolated local testing; persistent authentication across restarts is not established by that key.

Real HTTP tests on port 8019 before final hardening (http-tests.log, runtime-logs/http):

- Query-only France-capital generation: 200, accepted/verified_and_accepted, grounded checkpoint inference, one Memory write; 53.2 seconds.
- Fictional revenue supplied draft: 200, human_review/human_review_required, zero Memory writes; 23.6 seconds. Grounded Detector honestly skipped without evidence.
- False Microsoft-founder supplied draft: 200, correction independently reverified, accepted, three verified Memory writes; 39.7 seconds. Corrector provider openrouter observed.
- Health 200; empty query 422; nonexistent login 401; anonymous/invalid-token history 401. Two real test accounts had separate server histories, auth/me validated tokens, and local frontend CORS preflight passed. Test DBs only.

All seven actual agents executed in the observer-recorded real LangGraph under runtime-logs/local-correction. Wrappers observe original methods without faking providers, evidence or verdicts. Final source is rechecked over HTTP after deterministic verification.

Configured n8n retrieval disconnected without a response (RemoteProtocolError). Existing Python adapters actually supplied fallback evidence. n8n itself is not certified operational. Provider calls were bounded to a small local scenario set.

## Frontend

Actual page served at http://localhost:3020/chat, public backend URL http://127.0.0.1:8020, isolated dist directory .next-local-validation. Start Next from frontend-v2 using its existing node_modules. GET /chat returned 200. An initial launch from the repository root failed Tailwind discovery; correct working directory resolved it.

Production build compiled, checked types and generated routes (frontend-build.log). Standalone copying warned EPERM because Windows could not symlink the borrowed dependency directory. Deployment packaging was not attempted.

`node --experimental-default-type=module scripts/local_client_integration.mjs` passed the actual frontend API-module-to-backend journey: supported answer with real evidence, persist/reload, cancellation without a stored result, unreachable backend rejection without fabrication, and user-local history isolation. Evidence: runtime-logs/frontend-client-integration.json. Storage is a Node shim; this is explicitly not a browser rendering test.

Browser journey remains blocked: in-app browser transport closed, and native Chrome activation/input failed. Automatic approval review rejected headless Chrome launch and an attempted frontend restart with a generic policy rejection. No browser submission, pending/error rendering, source-drawer interaction or visual reload behavior is claimed.

Final-source targeted safety tests: **65 passed** in 42.36 seconds (last-safety-regression.log), including the concurrency and API aggregation regressions added after the broad collection. Full fresh neural inference completed on all **18,777 test pairs**, exactly reproducing all saved logits. See DETECTOR_VALIDATION_REPORT.md for raw and overlap-filtered results. A later Memory promotion/provenance fix has its own complete Memory regression run.

Final Memory and correction-related regression: **200 passed** in 86.74 seconds (final-fix-regression.log), covering the later durable-store, promotion/provenance, presentation-aware topicality and existing wrong-topic checks. The full Memory/safety suite before the last topicality fix passed 160 tests. Final frontend build compiled in 8.5 seconds, passed type checking and generated routes; the Windows standalone symlink warning remains packaging-only.

The final HTTP matrix accepted the generated supported answer (75.7 seconds) and withheld fictional revenue with zero writes (40.8 seconds). The original correction-success assertion failed: a provider returned only an unverifiable denial, which was rejected and not stored. A further positive candidate independently verified both founders but was rejected by the presentation-word topicality bug; its failed trace is preserved. The subsequent bug fix and final live rerun are recorded separately rather than erasing those failures.

Final-source live acceptance after the topicality fix: **PASS** (correction-after-fix.log, runtime-logs/http-final/positive-correction-after-fix.json). HTTP 200 in 129.0 seconds; final answer "Microsoft was founded by Bill Gates and Paul Allen."; terminal accepted / verified_and_accepted; independent reverification passed; all seven trust agents appear in the actual trace; two verified facts stored in a new isolated runtime directory and reopened with attributed evidence. The local frontend responded HTTP 200. Earlier negative-only and false-topicality failures remain preserved.

Local delivery: frontend http://localhost:3020/chat and actual backend http://127.0.0.1:8020, using runtime-logs/api-runtime-delivery. The launcher uses the original provider configuration and process-only local auth key. Existing production Memory/user files remain unchanged and are not retroactively certified or migrated.
