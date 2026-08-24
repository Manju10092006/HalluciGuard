# HalluciGuard — Frontend

The web interface for **HalluciGuard**, an evidence-grounded verification instrument for
language-model output. It separates *the answer* from *whether the answer is true*, then
shows the full decision path: retrieval, BGE reranking, DeBERTa NLI, evidence scoring, and a
four-state verdict — with every source, score, and model run on the record.

This app is a thin, honest client over a **frozen** FastAPI + LangGraph backend. It never
computes verdicts itself, never fabricates progress or evidence, and never invents an
authentication state. When the backend skips a stage (e.g. the detector fast-path), the UI
says so plainly.

- **Stack:** Next.js 15 (App Router) · React 19 · TypeScript · Tailwind CSS 4
- **Auth:** Google Identity Services (client-only ID-token flow; optional — the product is
  fully usable anonymously)
- **Rendering:** static/SSR pages wrapping thin client workspaces; no server secrets

---

## Quick start (local development)

Requires Node.js 18.18+ (Node 20 LTS recommended) and npm.

```bash
# 1. Install dependencies
npm install

# 2. Configure environment
cp .env.example .env.local
#   → edit .env.local (see "Environment variables" below)

# 3. Run the dev server
npm run dev
#   → http://localhost:3000
```

By default the frontend calls a backend at `http://localhost:8000`. To develop the UI
**without** a running backend, set `NEXT_PUBLIC_ENABLE_MOCK=true` in `.env.local` — this
serves a realistic recorded fixture (including a truly-skipped-verifier fast-path) so the
interface can be built and reviewed offline. The mock is hard-disabled in production builds
and can never satisfy a deployed code path.

### Scripts

| Command | What it does |
| --- | --- |
| `npm run dev` | Start the dev server on `:3000` |
| `npm run build` | Production build (type-checked) |
| `npm run start` | Serve the production build locally |
| `npm run lint` | Run ESLint |

---

## Environment variables

Every variable is prefixed `NEXT_PUBLIC_` and is therefore **shipped to the browser**.
That is intentional — none of these are secrets. See [Security model](#security-model) for
what must *never* appear here.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `NEXT_PUBLIC_API_BASE_URL` | Yes | `http://localhost:8000` | Base URL of the FastAPI backend. Exposes `POST /verify` and `GET /health`. **No trailing slash.** |
| `NEXT_PUBLIC_API_TIMEOUT_MS` | No | `120000` | Client-side timeout for a verification call. Verification runs retrieval + rerank + NLI, so keep it generous. |
| `NEXT_PUBLIC_GOOGLE_CLIENT_ID` | No | *(empty)* | OAuth 2.0 **Client ID** for a Google "Web application" credential. Blank → sign-in is hidden and the app runs fully anonymously. |
| `NEXT_PUBLIC_ENABLE_MOCK` | No | `false` | Dev-only. `true` uses the local mock adapter instead of the backend. Ignored in production builds. |

Set these three ways depending on environment:

- **Local:** `.env.local` (gitignored — never committed)
- **Vercel:** Project → Settings → Environment Variables (per environment)
- **Reference:** `.env.example` (committed; contains no real values)

---

## Google OAuth setup

Sign-in uses the **Google Identity Services** browser flow, which needs only a public
**Client ID** — there is no client secret, no redirect callback route, and no token
exchange on any server. If you don't configure a Client ID, the app simply hides the
sign-in control; everything else works.

To enable it:

1. Go to **Google Cloud Console → APIs & Services → Credentials**.
2. Create (or open) an **OAuth 2.0 Client ID** of type **Web application**.
3. Under **Authorized JavaScript origins**, add every origin you serve the app from — one
   entry per origin, scheme included, no path, no trailing slash:

   ```
   http://localhost:3000
   https://<your-project>.vercel.app
   https://<your-project>-<hash>-<scope>.vercel.app   # Vercel preview URLs
   https://<your-production-domain>                    # if using a custom domain
   ```

   > Vercel generates a new preview URL per deployment. If you want sign-in to work on
   > previews, either add the preview origins you use, or map a stable preview alias in
   > Vercel and authorize that. Sign-in gracefully no-ops on any origin you haven't
   > authorized — the rest of the app is unaffected.

4. You do **not** need to fill in "Authorized redirect URIs" — this flow doesn't use them.
5. Copy the **Client ID** into `NEXT_PUBLIC_GOOGLE_CLIENT_ID`.

The client secret shown in the Console is **not used by this frontend** and must never be
added to it.

---

## Backend CORS

The browser calls the backend directly, so the **backend** must allow the frontend's
origin. The frontend sends **no cookies** (`credentials: "omit"`), so the backend should
run with `allow_credentials=False`, which also permits a wildcard origin if you want it.

Reference configuration for the FastAPI backend (this lives in the backend repo — the
frontend does not set CORS):

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://<your-project>.vercel.app",
        # add your production domain here
    ],
    allow_credentials=False,     # frontend sends no cookies
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)
```

To allow all Vercel preview deployments, either enumerate the origins you use or use an
`allow_origin_regex` such as `r"https://.*\.vercel\.app"` (with `allow_credentials=False`).

If a verification request fails with a CORS error in the browser console, it is almost
always a missing origin here — not a frontend bug.

---

## Deploying to Vercel

1. **Push** this `frontend-v2` directory to a Git repository.
2. In Vercel, **New Project → Import** that repo. If the repo root is the monorepo (not
   `frontend-v2`), set **Root Directory** to `frontend-v2`.
3. Vercel auto-detects Next.js. `vercel.json` already pins the framework and the
   build/dev/install commands, so no build overrides are needed.
4. Add **Environment Variables** (Settings → Environment Variables):
   - `NEXT_PUBLIC_API_BASE_URL` → your deployed backend URL (must be HTTPS in production,
     or browsers will block the mixed-content request from an HTTPS page).
   - `NEXT_PUBLIC_GOOGLE_CLIENT_ID` → your Client ID (optional).
   - `NEXT_PUBLIC_API_TIMEOUT_MS` → optional override.
   - Leave `NEXT_PUBLIC_ENABLE_MOCK` unset (or `false`) — mock is disabled in production
     regardless.
5. **Deploy.**
6. Post-deploy checklist:
   - Add the deployment origin(s) to Google **Authorized JavaScript origins**.
   - Add the deployment origin(s) to the backend **CORS** allow-list.
   - Confirm the backend is reachable over HTTPS from the deployed page.

Because all configuration is env-driven, the same build promotes cleanly from preview to
production — no code changes, no hardcoded URLs.

---

## Security model

The guiding rule: **this is a public client. Nothing secret may live in it.**

**Public by design (safe to expose, all `NEXT_PUBLIC_*`):**
the backend base URL, the API timeout, the Google OAuth *Client ID*, and the mock flag.

**Must stay server-side — never in this frontend, never in a `NEXT_PUBLIC_*` var, never in
the client bundle:**

- Google OAuth **client secret**
- OpenRouter API key
- Tavily API key
- n8n webhook secret
- NVD API key
- any other backend token or credential

These belong to the backend and are used only there. The frontend authenticates users with
the client-only Google flow (public Client ID → ID token decoded in the browser for display
name/email/avatar) and calls the backend with no privileged credentials.

Verification history is stored **locally in the browser** (`localStorage`), scoped per
signed-in user (or an anonymous bucket). It is never uploaded anywhere.

Response headers (`X-Content-Type-Options`, `Referrer-Policy`, `X-Frame-Options: DENY`) are
set in `next.config.mjs`.

---

## Project structure

```
frontend-v2/
├─ src/
│  ├─ app/                 # App Router routes
│  │  ├─ page.tsx          # Landing (anonymous, thesis-first)
│  │  ├─ verify/           # Verification workspace
│  │  ├─ history/          # Local verification history
│  │  ├─ error.tsx         # Route error boundary
│  │  └─ not-found.tsx     # 404
│  ├─ components/
│  │  ├─ shell/            # Nav, mobile nav, account menu, app shell
│  │  ├─ landing/          # Landing sections + signature interaction
│  │  ├─ verify/           # Query composer, workspace
│  │  ├─ results/          # Verdict, claims, evidence, source explorer, traces
│  │  ├─ history/          # History list + detail
│  │  └─ ui/               # Design-system primitives (Button, Panel, Modal, …)
│  └─ lib/
│     ├─ api/              # types · client (HTTP) · map (raw → view models) · mock
│     ├─ auth/             # Google Identity Services (client-only)
│     ├─ history/          # localStorage store
│     ├─ hooks/            # useVerification
│     └─ config.ts         # env-sourced runtime config
├─ .env.example            # documented, value-free env template
├─ next.config.mjs         # security headers, build settings
└─ vercel.json             # framework + commands
```

### How the frontend stays honest

- **Answer ≠ verification.** The generated answer and the verification report are always
  presented as separate things.
- **Four verdicts.** `VERIFIED · CONTRADICTED · UNVERIFIED · CONFLICTED`, each with a label
  and icon — never color alone.
- **Missing stays missing.** Absent scores render as `—`, never as `0`. Stages the backend
  didn't report are shown as "not reported"; a truly-skipped verifier is shown as skipped.
- **Python owns the judgment.** Retrieval/orchestration is attributed to n8n; BGE reranking,
  DeBERTa NLI, evidence scoring, and the verdict are attributed to the Python engine. n8n is
  never shown as the judge.
- **No chain-of-thought.** The UI shows models, scores, and evidence — not hidden reasoning.
```
