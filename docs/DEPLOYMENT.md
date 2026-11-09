# HalluciGuard deployment

The frontend is `frontend-v2` on Vercel. The Python model backend runs on Google Cloud Run. See [the November 9 audit](project-upgrade-20261109.md) for observed defects, repairs and validation limits.

## Vercel

Use the existing Vercel project with root directory `frontend-v2`, Next.js framework, Node.js 22, install `npm ci` and build `npm run build`. Select `frontend-v2` explicitly: the repository-root package does not declare Next.js, so Vercel version detection fails when the project root is left empty.

Set server-side `BACKEND_URL` to the verified Cloud Run origin with no path or trailing slash. The browser uses `/api/backend`; the legacy `NEXT_PUBLIC_API_BASE_URL` variable is ignored. Set `NEXT_PUBLIC_GOOGLE_CLIENT_ID` to the public Google web client ID that matches backend `GOOGLE_CLIENT_ID`. Provider keys and JWT signing secrets stay on Cloud Run.

The proxy handles health, verification, authentication and authenticated history. It has no success mock and does not cache account data or verification results. Verification allows up to 285 seconds upstream within a 300-second Vercel function deadline; browser timeout defaults to 290 seconds. Confirm the Vercel account supports that duration before release.

## Google Cloud

Production project: `project-836457aa-3a05-4a11-8f8`; region: `asia-south1`; existing public service: `halluciguard-detector`.

`deployment/cloudbuild.yaml` builds a current-code image over the immutable, previously baked CPU model runtime. Set `_RELEASE_SHA` to the exact source commit. The source upload uses `.gcloudignore`; the model weight is reused from the base image and checked against its expected SHA-256 by the container smoke check. The root Dockerfile remains the fresh-environment build route and needs hydrated Git LFS artifacts (`git lfs pull`).

Configure these runtime variables:

```env
HALLUCIGUARD_ENV=production
HALLUCIGUARD_LLM_PROVIDER_ORDER=openrouter
OPENROUTER_MODEL=openrouter/free
HG_CORRECTOR_PROVIDER=openrouter
HG_CORRECTOR_OPENROUTER_MODEL=openrouter/free
AUTH_STORE=firestore
GOOGLE_CLOUD_PROJECT=project-836457aa-3a05-4a11-8f8
FIRESTORE_DATABASE=(default)
MEMORY_EMBEDDING_MODEL=all-MiniLM-L6-v2
ALWAYS_VERIFY=true
ALLOW_DETECTOR_FAST_PATH=false
N8N_RETRIEVAL_ENABLED=false
ALLOW_MODEL_DOWNLOADS=false
```

Bind `OPENROUTER_API_KEY` to the existing Secret Manager secret `HALLUCIGUARD_OPENROUTER_API_KEY` and `JWT_SECRET` to the existing `JWT_SECRET`. Configure public `GOOGLE_CLIENT_ID` to match Vercel. The Cloud Run service identity needs existing access to those secrets and the app's Firestore database. Use application default credentials; no downloaded service-account key is needed.

The default Firestore database is used for accounts and authenticated verification history. Local runs default to SQLite. Passwords are salted hashes, never plaintext. Histories are scoped to the server-authenticated user. A result too large for a Firestore document fails history persistence explicitly; it is not fabricated or truncated silently.

Keep minimum instances zero, maximum one, concurrency one, 2 CPUs, 8 GiB memory and 300-second timeout for the initial release. Deploy a no-traffic tagged revision, check its real endpoints, then route traffic to that exact revision. Preserve the previous revision for rollback. FAISS and auxiliary Memory caches are instance-local; durable distributed semantic Memory is a separate follow-up.

## Release checks

```text
python -m pytest -q
npm --prefix frontend-v2 ci
npm --prefix frontend-v2 test
npm --prefix frontend-v2 run build
npm --prefix frontend-v2 audit --omit=dev
python scripts/check_live_release.py https://BACKEND_ORIGIN --auth --correction
python scripts/check_live_release.py https://FRONTEND_ORIGIN/api/backend --auth
```

The live checker creates a clearly named synthetic smoke-test account when `--auth` is supplied. No credentials are printed. Deep health checks configuration; the subsequent real verification must return accepted claims, evidence and executed Detector inference. Correction additionally requires independent re-verification to pass.
