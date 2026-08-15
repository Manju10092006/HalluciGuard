# HalluciGuard Production Deployment Guide
**Next.js Frontend on Vercel + FastAPI Backend on Render + OpenRouter Base LLM**

---

## 1. Architecture Overview

```
                          +-------------------------+
                          |   User Browser / Client |
                          +------------+------------+
                                       |
                                       v
                    +------------------------------------+
                    |  Vercel Global Edge (Next.js 15)   |
                    |  - VerificationService             |
                    |  - HalluciGuardAdapter             |
                    |  - Live UI Event Bus & Dashboard   |
                    +------------------+-----------------+
                                       | HTTPS POST /verify
                                       v
                    +------------------------------------+
                    |  Render Web Service (FastAPI)      |
                    |  - LangGraph Production Supervisor |
                    |  - Inter-Agent Event Bus           |
                    +----+-------------+---------------+
                         |             |
        +----------------+             +----------------+
        |                                               |
        v                                               v
+-------------------------------+               +-------------------------------+
| OpenRouter Base LLM API       |               | LangGraph Active Pipeline     |
| Model: qwen/qwen-2.5-7b-inst. |               | 1. Base LLM (Generation)      |
| Temp: 0.7 (normal) / 0.9 (st.)|               | 2. Detector Agent (HaluEval)  |
+-------------------------------+               | 3. Verifier Agent (NLI 9-stg) |
                                                | 4. Memory Agent (FAISS + KG)  |
                                                +-------------------------------+
                                                * Judge: DISABLED (ENABLE_JUDGE=false)
                                                * Corrector: DISABLED (ENABLE_CORRECTOR=false)
```

---

## 2. Security & Secrets Boundary

| Secret / Config | Destination | Exposed to Client? | Rule |
| :--- | :--- | :--- | :--- |
| `OPENROUTER_API_KEY` | Render Environment Variable | ❌ **NEVER** | Server-side only; never in git, frontend, or logs. |
| `NEXT_PUBLIC_HALLUCIGUARD_API_URL` | Vercel Environment Variable | ✅ Yes (Public URL) | Points to your deployed Render URL (`https://<service>.onrender.com`). |
| `CORS_ORIGINS` | Render Environment Variable | ❌ Server Config | Explicit list of allowed frontend origins (`https://<app>.vercel.app`). |

---

## 3. Backend Deployment on Render

### Step 1: Create Web Service via Blueprint or Manual Setup
- **Repository**: `https://github.com/Manju10092006/HalluciGuard`
- **Branch**: `main`
- **Runtime**: `Python 3.11.9`
- **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
- **Start Command**: `uvicorn orchestration.api:app --host 0.0.0.0 --port $PORT`

### Step 2: Configure Environment Variables in Render Dashboard
Under **Environment Variables** in Render, set:
```env
PYTHON_VERSION=3.11.9
HALLUCIGUARD_ENV=production
OPENROUTER_API_KEY=<YOUR_OPENROUTER_API_KEY>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=qwen/qwen-2.5-7b-instruct
OPENROUTER_TEMPERATURE=0.7
OPENROUTER_STRESS_TEMPERATURE=0.9
OPENROUTER_TIMEOUT_SECONDS=30
OPENROUTER_MAX_RETRIES=3
ENABLE_JUDGE=false
ENABLE_CORRECTOR=false
ALLOW_MODEL_DOWNLOADS=true
NLI_MODEL=cross-encoder/nli-deberta-v3-base
CORS_ORIGINS=https://<your-vercel-app>.vercel.app,http://localhost:3000
```

---

## 4. Frontend Deployment on Vercel

### Step 1: Import Project to Vercel
- **Repository**: `https://github.com/Manju10092006/HalluciGuard-Frontend`
- **Framework Preset**: `Next.js`
- **Root Directory**: `frontend` (or project root if monorepo)

### Step 2: Configure Environment Variables in Vercel Dashboard
Under **Settings → Environment Variables**, add:
```env
NEXT_PUBLIC_HALLUCIGUARD_API_URL=https://<your-render-service>.onrender.com
```

---

## 5. Verification & Health Monitoring

### Health Endpoint (`GET /health`)
Query the deployed backend health:
```bash
curl -X GET https://<your-render-service>.onrender.com/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "backend_status": "healthy",
  "environment": "production",
  "engine": "langgraph_production_supervisor",
  "active_agents": ["base_llm", "detector", "verifier", "memory"],
  "disabled_agents": ["judge", "corrector"],
  "base_llm": {
    "provider": "openrouter",
    "model": "qwen/qwen-2.5-7b-instruct",
    "key_configured": true
  }
}
```

### Verification API Endpoint (`POST /verify`)
```bash
curl -X POST https://<your-render-service>.onrender.com/verify \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "What is the capital of France?",
    "generation_mode": "normal"
  }'
```

---

## 6. Local Development Run Instructions

### 1. Run Backend Locally:
```bash
cd HalluciGuard
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn orchestration.api:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Run Frontend Locally:
```bash
cd HalluciGuard-Frontend/frontend
npm install
export NEXT_PUBLIC_HALLUCIGUARD_API_URL=http://localhost:8000
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 7. Troubleshooting & Rollback

- **CORS Errors**: Ensure `CORS_ORIGINS` on Render includes your exact Vercel deployment domain.
- **OpenRouter Errors**: Check `OPENROUTER_API_KEY` validity and account credits on OpenRouter.
- **Degraded Status**: If `GET /health` reports `degraded`, check the `checks` array in the response to isolate the failing component.
- **Rollback**: In Vercel, use **Deployments → Promote to Production** to instantly rollback to a previous working deployment. In Render, select **Deploys → Rollback to this deploy**.
