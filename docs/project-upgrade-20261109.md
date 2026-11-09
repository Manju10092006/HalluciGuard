# Project upgrade audit — November 9, 2026

This work covers the application and deployment. The existing uncommitted research documents, notebooks, reports and verifier research changes were preserved.

## Repository and live-state findings

- GitHub `main` was `82cdc6fb734f5fa31cd345a4060f60178bea2118` when inspected. The local starting commit was `16ab275`, three commits ahead of that remote main. Work is isolated on `codex/project-upgrade-20261109`; no existing branch or user edits were reset.
- The Vercel landing page returned HTTP 200, but `/health` returned 404. The documented Render backend returned 404. Neither result demonstrated a working application connection.
- The existing public Cloud Run service is `halluciguard-detector`, in `asia-south1`, project `project-836457aa-3a05-4a11-8f8`. Its shallow health returned healthy, but deep health was degraded and a real generation request ended in human review with a Base LLM failure. Its only configured runtime environment name was `FRONTEND_ORIGIN`.
- A supplied, factually correct draft also ended in human review with zero retrieved evidence. That deployment was not ready for factual verification.
- Secret Manager already contained `HALLUCIGUARD_OPENROUTER_API_KEY` and `JWT_SECRET`. The existing provider credential returned HTTP 200 from OpenRouter's key-validation endpoint. Secret values were not printed or committed.

## Repairs

| Area | Verified defect | Change |
| --- | --- | --- |
| Active chat | Keyword-based verdicts, invented evidence, a sample signed-in identity and a timed fake success | Real `/verify` requests, actual evidence and trace rendering, anonymous initial state, visible errors |
| Authentication | Passwords ignored; decoded Google tokens trusted client-side; fake JWT persisted | Backend login/register/Google-token exchange; stored JWT validated with `/auth/me` |
| Backend connection | Missing Next.js API route; inconsistent public environment variables and localhost fallbacks | One same-origin `/api/backend` transport with server-only `BACKEND_URL`, endpoint allowlist, uncached responses and bounded timeouts; legacy public backend URLs are ignored |
| Release display | Partial or unavailable results could be presented as supported; draft fallback could leak unreleased text | Supported requires Judge acceptance and every claim verified; no answer fallback to draft; corrections require passed re-verification |
| Detector | Entity-name heuristic replaced calibrated probabilities with fixed 0.05/0.90/0.05 values | Preserve learned probabilities; make entity mismatch an explicit verification requirement rather than proof of contradiction |
| Detector runtime | Missing calibration silently defaulted while adapter claimed calibration; assumed class order; unbounded answer batches | Validate calibration/class order, offline checkpoint loading, eight-sentence batches, finite-output checks, preserve claim tokens |
| Sentence spans | Titles inside sentences split incorrectly; closing quotation marks were dropped | Preserve exact offsets, titles, quotes and lowercase sentence starts |
| Readiness | File existence treated a Git LFS pointer as a ready checkpoint; OpenRouter was required even for Groq-only routing | Validate local artifact metadata and LFS hydration; validate the configured provider router; label configuration checks accurately |
| Memory | Verifier `EMBEDDING_MODEL` could override Memory's 384-dimensional model | Independent `MEMORY_EMBEDDING_MODEL` setting |
| Accounts/history | SQLite on an ephemeral Cloud Run filesystem could lose passwords/history after restart | Optional Firestore repository using the Cloud Run service identity, transactional email uniqueness and per-user history; SQLite retained for local use |
| Dependencies | npm reported six vulnerabilities, including a critical Next.js advisory | Updated lockfile to patched Next.js 15.5.29 and dependencies; compatible PostCSS override; npm audit reports zero |
| CI | Detector workflow referenced deleted `agents/detector_agent`; detector/services tests absent from default pytest paths | One current backend regression/compilation workflow covering all configured agents, replacing redundant older per-agent workflows, plus frontend production/test/audit checks |
| Packaging | Docker sent research, local data and generated files with a broad COPY | Source allowlists for Docker/Cloud Build; no credentials/research/state in the upload |

## Detector quality boundary

No checkpoint was trained or replaced. Actual CPU smoke inference returned risk 0.0359 for the supported Microsoft founder pair, 0.0818 for a false founder pair, and 0.0279 for the supported Paris pair. These three observations are diagnostic examples, not a benchmark. The false-founder example demonstrates that the current learned model can miss an entity substitution. The revised review flag routes entity differences to verification without inventing a high model probability. Always-verify remains enabled and detector bypass disabled.

The next model-quality upgrade should use a source-disjoint, deduplicated evaluation set with relation/entity swaps, negation, dates, multilingual claims and insufficient evidence. Report contradiction precision/recall separately from unknown evidence, recalibrate only on a separate development split, and promote a checkpoint only after held-out evaluation. Existing historical metrics and data overlaps do not establish improved production accuracy.

## Deployment approach

Use Next.js on Vercel, a server-only API proxy, and the Python model runtime on Google Cloud Run. The existing immutable staging model image is reused by `deployment/Dockerfile.upgrade`, with current application sources and patched API packages overlaid. The build checks the exact Detector SHA-256, cached model weights, actual offline Detector inference and HTTP health. This avoids multi-GB downloads during a cold request.

Cloud runtime settings: request-based billing, minimum zero instances, maximum one instance, concurrency one, two CPUs, 8 GiB memory and a 300-second request deadline. Secrets are referenced from Secret Manager at runtime. No provider credential or service-account private key belongs in Vercel or the image.

Firestore stores accounts and authenticated history. Browser conversations remain explicitly browser-local and scoped by account. The FAISS knowledge index and auxiliary Memory caches remain instance-local and rebuildable; they are not certified durable cross-instance storage. Domain-specific verifier models beyond the baked general path have not been certified. External Slack/Linear/Notion workflow integrations are not connected and the UI no longer reports invented runs.

## Validation record

- Initial backend suites: 1,024 passed, 6 skipped.
- Separate Verifier suite: 466 passed, 8 skipped.
- Frontend: five failure/acceptance/evidence/transport contract tests passed; production build passed with Next.js 15.5.29; npm audit reported zero vulnerabilities.
- Targeted cloud-auth/readiness/Detector/span regressions: 14 passed.
- Complete run against patched FastAPI 0.162.2 and Starlette 1.10.0: **1,510 passed, 14 skipped**. The Memory environment-isolation regression found during the preceding run was fixed and the complete suite rerun successfully.
- Real local generation: accepted Microsoft founder answer; actual Groq generation, evidence retrieval, grounded Detector inference, Judge and Memory executed without errors.
- Real local correction diagnostic: the supplied false founder draft was corrected to Bill Gates and Paul Allen, independently re-verified and accepted; Corrector and ReVerifier executed. These are execution demonstrations, not accuracy benchmarks.
- Fresh Linux CI exposed an Arrow/NumPy compatibility issue and n8n tests that depended on local environment flags. Arrow is constrained to a compatible release and mocked-broker tests now select their boundary explicitly. Both fresh Linux backend CI and frontend CI passed on commit `535c9a8`.
- The cloud image self-test also caught an older inherited `datasets` package using Arrow's removed `PyExtensionType`. The overlay now installs the tested `datasets` 5.0.1 alongside Arrow 23.0.1, and retains the Transformers-compatible Hugging Face Hub major version.
- The local preview server launch was blocked by automatic approval review. Browser review is being performed against the Vercel preview instead.

Run `python scripts/check_live_release.py <backend-origin> --auth --correction` to check real generation, evidence, Detector execution, password validation, history and the correction path. For the frontend proxy use `<vercel-origin>/api/backend`. Passing builds or shallow health alone is not a release gate.

References: [Vercel function limits](https://vercel.com/docs/functions/limitations), [Firestore database setup](https://docs.cloud.google.com/firestore/native/docs/manage-databases), [Firestore server clients](https://docs.cloud.google.com/firestore/native/docs/create-database-server-client-library).
