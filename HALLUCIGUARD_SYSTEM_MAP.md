# Local system map — 2026-10-19

Branch `codex/local-validation-20261019`, base `80769fa`. The original dirty checkout and deployment worktrees were not edited.

Browser `/chat` → `frontend-v2/src/app/chat/page.tsx` → `components/shell/HalluciGuardShell.jsx` → `src/lib/api.js` → POST `/api/v1/verify` in `orchestration/api.py` → `graph.run_verification`.

Actual compiled LangGraph (`orchestration/graph.py:build_verification_graph`):
`generate → detector → claim_analyzer → verifier → grounded_detector → judge`.
Judge routes ACCEPT to Memory; VERIFY_AGAIN back to Verifier with a bounded retry budget; CORRECT to Corrector then independent ReVerifier then Judge; REJECT/human escalation also pass through Memory, with persistence skipped.

The seven trust implementations are `halluciguard_detector.agent.DetectorAgent`, `services.claim_analyzer.ClaimAnalyzer`, `agents/verifier_agent/api/pipeline.py:VerificationPipeline`, `agents/judge_agent/judge_agent.py:JudgeAgent`, `services/character_regenerator.py:CharacterRegenerator` plus the configured Corrector, `graph._reverifier_node`, and `agents/memory_agent/memory/memory_agent.py:MemoryAgent`.

`orchestration/state.py:HalluciGuardState` carries draft/final text, detected claims, canonical results, evidence, correction/reverification counters, errors, traces and the inter-agent bus. `orchestration/schemas.py` defines the boundary contracts. `detector_bridge.py` preserves neural class probabilities, triage risk, contradiction mass and evidence provenance.

API response fields include `terminal_status`, `verification_status`, `final_response`, `verifier.claim_evidence`, correction/reverification results, Memory results and actual trace events. `frontend-v2/src/lib/verification-contract.mjs` permits Supported only for accepted, verified, nonempty fully supported claim reports with evidence. Unreleased drafts/corrections are withheld. Network failures propagate.

Conversations/projects/feedback are explicitly browser-local storage, scoped by authenticated user ID; no synthetic history is seeded. Server accounts and authenticated API history use SQLite and JWT in `orchestration/auth.py`. Frontend authentication now calls the backend and validates restored sessions via `/auth/me`.

`.env` was inspected by variable presence only and loaded explicitly by the local launcher. Provider credentials stay server-side; only the public local API URL is passed to Next. `scripts/local_validation_server.py` isolates databases/Memory, disables result caches/downloads and enables verification-first policy. Missing local JWT_SECRET is replaced only by a process-local random key for isolated testing, never a provider credential.
