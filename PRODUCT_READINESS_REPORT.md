# Product readiness — 2026-10-19

## Decision

Substantial local correctness and integration work is complete. This is **not a full browser end-to-end or production-readiness certification**. The trained Detector remains weak on refutations; factual authority stays with independent evidence verification and the Judge. No deployment was performed.

## Verified

- Isolated branch `codex/local-validation-20261019` preserves the original dirty checkout, model weights, user stores and separate deployment worktrees.
- Actual upgraded Detector loads the original safetensors checkpoint. Full fresh CUDA inference on 18,777 historical test pairs exactly reproduces every saved logit. Label mapping, temperature, separate task thresholds, finite outputs and input contracts have regression coverage. Dev-only threshold-grid reproduction recovers 0.46 and 0.625.
- Dataset counts/labels/IDs/source groups audited. Source IDs do not overlap; exact claim/evidence pairs do. Filtered unique test evaluation uses 17,954 pairs, macro-F1 0.508101. Original complete test macro-F1 is 0.507246; these are different evaluation populations, not an improvement claim.
- Actual real-provider LangGraph correction run executes Detector, Claim Analyzer, Verifier, Judge, Corrector, independent ReVerifier and Memory. Initial live HTTP tests exercise generated acceptance, unsupported withholding and corrected acceptance with evidence and safe Memory decisions.
- Judge rejects unsupported peripheral remainder and bare decisive verdicts. Initial verification checks full claim coverage. Hosted extraction has a deterministic source backstop. Corrected text requires independent supported coverage and evidence, not a success flag.
- Memory rejects incomplete/unsupported writes and verified flags without attributed evidence. Negative records cannot be promoted without new evidence; provenance/confidence are retained and vector metadata persists. Single-loop graph writes are serialized to avoid stale snapshot overwrites.
- Broad deterministic regression: 1709 passed, 14 skipped. Final targeted safety regressions: 65 passed. After subsequent Memory update hardening, the complete Memory/safety suite passed 160 tests; the final correction/Memory regression passed 200 tests. These are overlapping suites and must not be added together as unique tests.
- Frontend contract tests: 8 passed. Actual frontend API module reaches the real backend and verifies evidence/status, persistence/reload, cancellation rejection, unavailable backend behavior and user-local history isolation. This test uses Node storage, not a browser.
- Real backend authentication/history isolation, invalid-token behavior and local frontend CORS were checked over HTTP. The actual frontend page serves locally, compiles and type-checks. No configured credential value appeared in changed files or generated static JavaScript during the scan.

## Failed or limited

- Live provider correction is not guaranteed to succeed: a negative-only repair was independently rejected with zero writes. A separate fully supported repair exposed a topicality bug involving presentation instructions; it was fixed and regression-tested. Failed traces are preserved.
- The shipped model matched only 3/8 labeled curated probes. A fixed 24-case held-out sample through upgraded decomposition/selection scored macro-F1 0.282132 and missed all eight contradiction cases. The sample is small and balanced, with historical sentence labels, so it is not a population accuracy estimate. These failures were reported rather than hidden by threshold tuning or fabricated guard probabilities.
- Historical training used sentence/concatenated lexical evidence; upgraded runtime uses atomic claims/single selected evidence. Original raw span annotations and exact historical run parameters are unavailable. Processed splits and saved predictions do exist and were audited. No scientifically justified retraining was attempted without establishing a new aligned training/evaluation design.
- n8n retrieval disconnects without a response under the existing configuration. Existing Python fallback retrieval ran successfully, but n8n itself remains unverified as an operational service.
- Windows standalone build copying warns EPERM for the reused node_modules junction. Local compile, type checking and routes succeed; deployable packaging is not certified and is outside this task.
- The original configuration has no stable JWT secret. The isolated local launcher generates a process-only random key, enabling genuine auth tests; sessions do not survive a restart with that key.
- Memory reads are tested through the Memory API/class lifecycle, but the graph does not read Memory into generation or use it as factual proof. Cross-process/multi-worker persistence is not certified; validated local serving is single-worker.

## Exact remaining blocked / untested cases

- Browser submission-to-rendering journey, pending/withheld/failed visual states, source drawer, visual reload and Google OAuth in the browser: blocked by in-app browser transport failure and native activation/input failure. Automatic approval review also rejected headless Chrome launch with a generic policy rejection. No successful browser journey is claimed.
- In-flight browser cancellation and cancellation propagation into already-running provider/inference work are not certified. The actual API-module test proves pre-aborted requests and unavailable-network failures do not fabricate or store a client result.
- Live timeout/failure induction for every external provider, forced Gemini failover and every external evidence domain were not performed. Deterministic failure/timeout regressions passed; bounded successful provider calls do not certify all outages.
- Four obsolete Detector diagnostics tests skip because the removed legacy agents.detector_agent package is absent; six Corrector curriculum checks skip without contract_v4 training data; two opt-in live n8n and two opt-in live web tests skip by default. Actual new Detector inference and local graph/API runs are separately recorded.
- Full annotation-to-training regeneration and a fresh full-population evaluation of the upgraded atomic-claim runtime remain unperformed. Full historical pairwise neural inference is complete; the 24-case runtime evaluation is reported separately.

See LOCAL_TESTING_REPORT.md for commands and artifacts, DETECTOR_VALIDATION_REPORT.md for scientific limits, and BUG_FIX_AND_REGRESSION_REPORT.md for fixes. Final HTTP rerun evidence is recorded below.

Final-source live acceptance after the topicality fix: **PASS** (correction-after-fix.log, runtime-logs/http-final/positive-correction-after-fix.json). HTTP 200 in 129.0 seconds; final answer "Microsoft was founded by Bill Gates and Paul Allen."; terminal accepted / verified_and_accepted; independent reverification passed; all seven trust agents appear in the actual trace; two verified facts stored in a new isolated runtime directory and reopened with attributed evidence. The local frontend responded HTTP 200. Earlier negative-only and false-topicality failures remain preserved.

Local delivery: frontend http://localhost:3020/chat and actual backend http://127.0.0.1:8020, using runtime-logs/api-runtime-delivery. The launcher uses the original provider configuration and process-only local auth key. Existing production Memory/user files remain unchanged and are not retroactively certified or migrated.
