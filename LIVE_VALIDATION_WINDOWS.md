# HalluciGuard — Windows Live Validation Guide

**Vertical slice:** `Base LLM → Detector → n8n Retrieval V2 → BGE Reranker → DeBERTa NLI → EvidenceScorer → Final Verdict`

This guide is the **live-validation** half of the integration task. All static
verification (code audit, `py_compile`, structural checks, unit-test authoring)
was completed in the sandbox. The runtime chain with **real** OpenRouter, n8n V2,
BGE, and DeBERTa can only be proven on this Windows machine, because those
require network + torch + the actual models. Per the spec's Completion Criteria,
the task is **not** "complete" until at least one real claim has demonstrated the
full runtime chain with all runtime inputs visible in the diagnostic output.

This pass also added **real-execution-only diagnostics** and an opt-in
**certification mode**. Every stage now emits proof that it actually ran a model
(detector `Inference Executed`, and per-claim `MODEL EXECUTION PROOF` for BGE and
NLI). Certification mode (Step 4) turns any fail-soft fallback or mock/empty
evidence into a **controlled failure** instead of a best-effort verdict — that is
the strongest available proof the chain is genuinely live.

Run the steps below **in order** and record the outputs.

---

## 0. Prerequisites (one-time)

Open **PowerShell** in the repository root (the folder that contains `pytest.ini`,
`agents/`, `services/`, `scripts/`).

```powershell
cd C:\path\to\HalluciGuard

# (Recommended) activate your existing venv / conda env that already has
# torch, transformers, sentence-transformers, httpx, pydantic, pytest, pytest-asyncio.
# e.g.  conda activate halluciguard   OR   .\.venv\Scripts\Activate.ps1

python --version          # expect the 3.13 env the project was developed on
python -c "import torch, transformers, sentence_transformers, httpx, pydantic; print('deps OK')"
```

### Configure `.env` (never commit real secrets)

Create/verify a `.env` file in the repository root. Both runner scripts call
`load_dotenv()` on this file, and the verifier `Settings` also reads it.

```ini
# --- Base LLM (OpenRouter / Qwen) ---
OPENROUTER_API_KEY=<your-openrouter-key>
OPENROUTER_MODEL=qwen/qwen3-4b

# --- n8n Retrieval Service V2 (FROZEN endpoint — do not change the URL) ---
N8N_RETRIEVAL_ENABLED=true
N8N_RETRIEVAL_WEBHOOK_URL=https://manjusogala.app.n8n.cloud/webhook/halluciguard-verify-v2
N8N_AUTH_MODE=header
N8N_HEADER_NAME=X-API-Key
N8N_WEBHOOK_SECRET=<your-n8n-webhook-secret>
N8N_TIMEOUT_SECONDS=60

# --- Detector (HaluEval DistilBERT) ---
# Use the HF repo default, OR point to your local fine-tuned weights directory:
# HALUEVAL_MODEL_PATH=artifacts/halueval-detector
HALUEVAL_MODEL_PATH=Manjunath2000006/halluciguard-detector

# --- Models / runtime ---
# First run: allow downloads so BGE (bge-reranker-large) + DeBERTa (nli-deberta-v3-base)
# and the detector can be fetched/cached. MUST NOT be mock mode.
ALLOW_MODEL_DOWNLOADS=true
MOCK_MODE=false
LOG_LEVEL=INFO

# --- Certification mode (OFF by default; see Step 4) ---
# Leave these unset/false for normal (resilient) production runs. Step 4 sets
# them for the fail-closed certification pass; you do NOT put them here for
# Steps 1-3.
# CERTIFICATION_MODE=false
# CACHE_ENABLED=true
```

> **Security:** the code reads secrets from env only — never hard-code or echo
> `OPENROUTER_API_KEY` / `N8N_WEBHOOK_SECRET`. The boundary test prints only
> `Secret Set : YES (masked)`, and the LLM client redacts the key in error text.

---

## 1. Unit tests (offline — proves the wiring)

Run the full suite first, then the integration-focused files. `asyncio_mode=auto`
is already set in `pytest.ini`, so `pytest-asyncio` must be installed.

```powershell
# Full current test suite (spec §29: "Run the full current test suite.")
pytest -q

# Verifier agent only (fast iteration on the stabilization work):
pytest -q agents/verifier_agent/tests

# The integration-focused files that carry the §29 cases:
pytest -v `
  agents/verifier_agent/tests/test_n8n_retrieval.py `
  orchestration/tests/test_llm_detector_verifier_slice.py `
  agents/verifier_agent/tests/test_slice_integration_supplement.py

# The new hardening regression suite (§39 — one test per discovered bug):
pytest -v agents/verifier_agent/tests/test_verifier_hardening_regression.py
```

**Expected:** all pass. `test_slice_integration_supplement.py` adds the only §29
case the existing suites did not assert directly — **Case 7: BGE receives the real
claim + real evidence** — plus an explicit `adapter_score != bge_score`
end-to-end guard (§9) and a four-state-verdict guard (§10). These are hermetic
(n8n/BGE/NLI mocked); they prove wiring, not real model execution.

`test_verifier_hardening_regression.py` pins the fixes from this pass and is
**hermetic** (no torch/network — models are faked and fallbacks are forced):

- detector diagnostics: baseline fallback is `degraded / not-executed`; real
  inference is `executed / not-degraded` with provenance (§6);
- BGE fallback is flagged and its hybrid score is **not** presented as a BGE
  score; real inference records `executed` (§13);
- NLI engine-level execution proof on every batch call (§15);
- certification helpers fail-closed on detector/BGE/NLI fallback and on
  mock/empty/malformed evidence, and are strict no-ops when disabled (§28);
- `RelationCheckTrace` round-trips its flat fields (the pre-fix schema silently
  dropped them) and `RetrievalTrace` carries the execution blocks (§26);
- the result cache is disabled whenever certification mode is on (§30).

The certification-helper subset is dependency-light and was already verified in
the sandbox (29 assertions); the model/pydantic/pipeline tests run here for real.
If a group is `SKIPPED`, its message names the missing dependency.

> **Python 3.13 wheel note:** if `pytest -q` errors during import with a NumPy 2.x
> ABI message (`numpy.dtype size changed` / a package compiled against NumPy 1.x),
> pin `pip install "numpy<2"` in this env, or upgrade the offending package to a
> 3.13 / NumPy-2 compatible wheel. This is an environment/ABI issue, not a code
> defect — resolve it before trusting the suite result.

Real model execution is proven in Steps 3-4, not here.

---

## 2. n8n boundary test (proves the FROZEN retrieval boundary is live)

This calls the real n8n V2 webhook (POST only — no health call) and normalizes
the response. It does **not** load BGE/DeBERTa.

```powershell
python scripts/test_n8n_retrieval.py --claim "Paris is the capital of France."
python scripts/test_n8n_retrieval.py --claim "The Eiffel Tower is located in London."
python scripts/test_n8n_retrieval.py --claim "Aspirin is used to relieve mild to moderate pain." --domain healthcare
```

**Confirm in the output:**

- `Success : True` and `HTTP Status : 200`
- `Secret Set : YES (masked)` (auth header applied)
- `Workflow Version : 2.x`, a non-zero `Latency`, and `Primary Sources` (e.g. `['wikipedia']`)
- `Normalized Evidence Passages Count > 0` with provenance records
- For each passage: `adapter_score` is populated from the n8n hint, and
  `bge_score : 0.0000 (Reserved for Python BGE inference)` — this **0.0 is correct
  here**; the real BGE score is computed later in the full chain, not by n8n.

If this step fails, **stop and report the exact HTTP status / error** before
proceeding — the downstream chain depends on this boundary.

---

## 3. Full vertical slice — live tests A–H (the actual completion criterion)

This is the run that satisfies the spec: real LLM → Detector → n8n → BGE → DeBERTa
→ EvidenceScorer → verdict. `--all-tests` runs the full **A–H** matrix. All cases
force the verifier, so the chain runs regardless of the detector's own risk tier.

```powershell
# Run the whole A-H matrix:
python scripts/test_llm_detector_verifier_slice.py --all-tests

# Or run one claim at a time (useful for capturing per-claim latency):
python scripts/test_llm_detector_verifier_slice.py --query "Who is the father of Allu Arjun?"
python scripts/test_llm_detector_verifier_slice.py --query "Allu Arjun's father is Chiranjeevi."
python scripts/test_llm_detector_verifier_slice.py --query "Paris is the capital of France."
python scripts/test_llm_detector_verifier_slice.py --query "The Eiffel Tower is located in London."
python scripts/test_llm_detector_verifier_slice.py --query "Aspirin is used to relieve mild to moderate pain." --domain healthcare
python scripts/test_llm_detector_verifier_slice.py --query "The moon is made of green cheese and orbits Jupiter."
python scripts/test_llm_detector_verifier_slice.py --query "Who is the current president of the United States?"
python scripts/test_llm_detector_verifier_slice.py --query "Tell me three facts about the Eiffel Tower."
```

Cases **F–H** exercise, respectively: a nonsense claim (must **never** come back
`VERIFIED`), a time-sensitive claim (tests live retrieval recency), and a
multi-sentence answer that must **decompose into multiple per-claim verdicts**
(you should see more than one `Claim #` block in the trace for H).

To capture everything (including the model-load log lines) to a file:

```powershell
python scripts/test_llm_detector_verifier_slice.py --all-tests *>&1 |
  Tee-Object -FilePath live_validation_output.txt
```

---

## Live-validation checklist (verify per the diagnostic trace)

For **every** claim, the `FULL VERTICAL SLICE EXECUTION TRACE` must show each link
of the chain. Tick these off:

**Chain visibility (spec §30 / §32 — all runtime inputs must be visible):**

- [ ] **LLM** — a non-empty `LLM Draft Response` is printed.
- [ ] **Detector** — `Hallucination Prob`, `Risk Tier`, `Decision`, `Model Source` printed.
- [ ] **n8n** — `Sources Used`, `Retrieved Count` (> 0), `Tavily Fallback`, and counts/performance printed.
- [ ] **Evidence + NLI** — at least one decision-grade citation with `adapter_score`, `bge_score`, `NLI Label`, `nli_entailment`, `nli_contradiction`, `nli_neutral`, `classification`, and a real `Snippet`.
- [ ] **Final Verdict** — one of `VERIFIED / CONTRADICTED / UNVERIFIED / CONFLICTED`, with support/contradiction/trust/confidence scores.

**Proof that the real models ran (not fail-soft fallbacks) — this is the crux.**
This pass added explicit diagnostics so you no longer have to infer this from log
lines alone. In each claim's trace, look for the `MODEL EXECUTION PROOF` block and
the detector fields:

- [ ] **Detector model really loaded** — the `DETECTOR:` block shows
  `Model Loaded : True`, `Inference Executed : True`, `Degraded (baseline) : False`,
  and a real `Detector Provenance` (a model dir / HF id, **not** `baseline-heuristic`).
  If instead every claim shows `Inference Executed : False` / `Degraded : True` (and
  a flat `0.0800` probability), the HaluEval model failed to load — fix
  `HALUEVAL_MODEL_PATH`. (Cross-check: `Hallucination Prob` should vary across claims.)
- [ ] **BGE really ran** — `MODEL EXECUTION PROOF → BGE` shows `status=executed`,
  `executed=True`, `degraded=False`, a real `device` (e.g. `cuda`/`cpu`) and
  `scored=<n>`. Corroborate in the citations: `bge_score` should differ from
  `adapter_score` (real cross-encoder relevance, not the passed-through hint). If
  BGE shows `status=unavailable`/`degraded` (or `bge_score == adapter_score` for every
  passage), BGE fell back — investigate before trusting the verdict.
- [ ] **DeBERTa really ran** — `MODEL EXECUTION PROOF → NLI` shows `status=executed`,
  `executed=True`, `degraded=False`, and `pairs=<n>`. The three NLI probabilities
  should form a genuine distribution and evidence must not be flagged degraded.
  (Degraded NLI is excluded from decision-grade evidence, so degraded ⇒ few/zero
  citations and `UNVERIFIED`.)

**Sanity check on outcomes** (these depend on live retrieval + model outputs, so
treat them as expectations to confirm, **not** guarantees — report whatever
actually prints):

| Test | Claim | Truth | Expected verdict |
|------|-------|-------|------------------|
| A | "Who is the father of Allu Arjun?" | LLM should answer *Allu Aravind* | VERIFIED (if answer is correct & supported) |
| B | "Allu Arjun's father is Chiranjeevi." | False | CONTRADICTED |
| C | "Paris is the capital of France." | True | VERIFIED |
| D | "The Eiffel Tower is located in London." | False | CONTRADICTED |
| E | "Aspirin is used to relieve mild to moderate pain." | True | VERIFIED |
| F | "The moon is made of green cheese and orbits Jupiter." | False / nonsense | CONTRADICTED or UNVERIFIED — **never** VERIFIED |
| G | "Who is the current president of the United States?" | Time-sensitive | VERIFIED iff live evidence supports the LLM's answer; else UNVERIFIED |
| H | "Tell me three facts about the Eiffel Tower." | Multi-claim | Multiple `Claim #` blocks, each with its own verdict (decomposition proof) |

If a real result disagrees with the expectation, that is a genuine finding to
report — do **not** adjust code to force an expected verdict.

**Latency:** note `latency_ms` per claim (first run includes one-time model
downloads/loads; re-run for steady-state numbers).

---

## 4. Certification run — fail-closed runtime proof (§28/§29/§30)

Steps 1–3 run the **resilient** production path (a degraded model still yields a
best-effort verdict). Certification mode is the opposite: it exists to *prove* the
chain ran for real, so a detector/BGE/NLI fallback or mock/empty/malformed
evidence becomes a **controlled `CertificationError`** instead of a verdict, and
the result cache is bypassed so a cached row can never stand in as proof.

First, confirm model readiness explicitly (this actually loads BGE + NLI — it is
not inferred from imports):

```powershell
python -c "import sys, os; sys.path.insert(0, os.path.join('agents','verifier_agent')); from api.pipeline import VerificationPipeline; import json; print(json.dumps(VerificationPipeline().health_check(probe=True), indent=2))"
```

Expect `BGE_READY: true`, `NLI_READY: true`, `N8N_READY: true`. `CERTIFICATION_MODE`
and `CACHE_ENABLED` reflect the current env (here: `false` / `true`).

Then run the slice in certification mode. The `--certify` flag sets
`CERTIFICATION_MODE=true` and `CACHE_ENABLED=false` for the process:

```powershell
# One claim, fail-closed:
python scripts/test_llm_detector_verifier_slice.py --certify --query "Paris is the capital of France."

# The whole A-H matrix, fail-closed:
python scripts/test_llm_detector_verifier_slice.py --certify --all-tests
```

**How to read it:**

- **Success (certified):** the normal execution trace prints, and the
  `MODEL EXECUTION PROOF` block shows `status=executed` for BGE and NLI with the
  detector `Inference Executed : True`. This is a genuinely runtime-validated claim.
- **Controlled failure:** you see a `CERTIFICATION FAILURE (controlled,
  fail-closed)` block naming the stage (`detector` / `bge_reranker` / `deberta_nli`
  / `retrieval`). This is **not** a crash — it means that stage could not be proven
  to have run for real (e.g. the HaluEval model isn't loaded, or BGE fell back).
  Fix the named stage (usually a model path / download / network issue), then
  re-run. **Do not** disable certification to "pass"; the controlled failure is the
  correct, honest outcome until the model actually runs.

Certification mode is **off** by default, so Steps 1–3 and all existing callers
are unaffected.

---

## If anything fails (spec §33)

Report the **exact** failure (the printed HTTP status, exception, or log line)
and stop at that step. Do not work around it, do not fabricate NLI/BGE numbers,
and do not declare production readiness. Common first-run issues:

- `MISSING_API_KEY` → `OPENROUTER_API_KEY` not in `.env`.
- n8n `403` → wrong/empty `N8N_WEBHOOK_SECRET` or wrong `N8N_HEADER_NAME`.
- Reranker/NLI fallback warnings → model download blocked; ensure
  `ALLOW_MODEL_DOWNLOADS=true` for the first run and that the machine has network.
- Detector stuck at `0.0800` → `HALUEVAL_MODEL_PATH` not resolvable.
- `CERTIFICATION FAILURE [certification:<stage>]` in Step 4 → the named stage did
  not run for real. Fix the root cause (model path / download / network) and
  re-run `--certify`; never remove `--certify` just to obtain a verdict.
