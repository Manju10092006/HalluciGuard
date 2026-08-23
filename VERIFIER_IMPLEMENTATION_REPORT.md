# HalluciGuard — Verifier Agent Implementation Report (§44)

**Task:** Audit + implement-only-what's-missing + integrate + test + instrument + live-validate + document, on the **existing** HalluciGuard Verifier Agent.
**Branch:** `verifier-v2-stabilization`
**Report date:** 2026-08-23
**Scope boundary:** Verifier Agent only. No Judge / Corrector / Memory work; no n8n redesign; no architecture rewrite.
**Environment where this report was produced:** offline sandbox — **no external network, no torch/transformers, no pytest, Python 3.13-target repo compiled under the sandbox interpreter.** Every network/GPU/model-dependent fact is therefore labeled **LIVE-PENDING** and must be produced by running `LIVE_VALIDATION_WINDOWS.md` on the Windows machine (Python 3.13, RTX 3050).

---

## Verification legend

Every claim in this report carries one of these labels:

- **STATIC-VERIFIED** — confirmed by reading source + `py_compile` + `grep`/AST in the sandbox.
- **UNIT-VERIFIED** — exercised by a test that actually ran (in-sandbox, dependency-light) or is authored to run hermetically on Windows.
- **LIVE-VERIFIED** — proven with the real LLM + n8n + BGE + DeBERTa on the Windows machine. **Nothing in this report is LIVE-VERIFIED yet** (the sandbox cannot reach the models/network).
- **LIVE-PENDING** — requires the Windows run to confirm; the plan is specified here.
- **KNOWN-ISSUE** — a defect found and either fixed (with a regression test) or explicitly flagged.

> **Completion status (§32/§43).** This report does **not** declare the task complete and does **not** declare production readiness. Per §43 the Verifier is "RUNTIME-VALIDATED" only after ≥1 real claim traverses the full chain with each stage proven to have executed. That proof is **LIVE-PENDING** (Section 20/21).

---

## 1. Executive summary

The Verifier was already a working retrieval→rerank→NLI→scoring→four-state-verdict pipeline. The audit found the pipeline **functionally complete but under-instrumented**: several fail-soft fallbacks (detector baseline, BGE fallback, NLI fallback) could not be distinguished from real model execution by any downstream consumer, so a silently-degraded run could *look* identical to a genuine one. This pass adds **real-execution-only proof** at every model stage, a structured **runtime trace contract**, and an opt-in **certification mode** that converts any such fallback (or mock/empty evidence) into a **controlled failure** instead of a best-effort verdict. All changes are **additive and default-off / default-safe**; the resilient production path is unchanged. One real observability bug (`RelationCheckTrace` schema mismatch) was discovered and fixed. **STATIC-VERIFIED** for all code; **UNIT-VERIFIED** for the certification helpers (29 in-sandbox assertions); model execution itself is **LIVE-PENDING**.

## 2. Task interpretation and guardrails

This was explicitly **not** a greenfield rewrite. The work was constrained to: audit the existing code, implement only what is genuinely missing, integrate it without changing frozen contracts, instrument for provable execution, and prepare live validation. The following were treated as **DO-NOT-TOUCH** and were **not** modified (**STATIC-VERIFIED** by diff inspection): the n8n Retrieval Service V2 workflow and its URL/auth contract; the four-state verdict semantics and thresholds; the LangGraph orchestration topology; the detector's external `DetectionResult` contract (extended additively only); and the Judge/Corrector/Memory agents. No thresholds were changed to make any benchmark label pass (§36).

## 3. Architecture recap (frozen — for context only)

Chain: `Base LLM (OpenRouter/Qwen) → Detector (HaluEval DistilBERT) → [risk gate] → Verifier: n8n Retrieval V2 → BGE reranker → DeBERTa NLI → EvidenceScorer → four-state verdict (VERIFIED / CONTRADICTED / UNVERIFIED / CONFLICTED)`. The n8n V2 boundary is retrieval-only (no verdict field); BGE and DeBERTa run in-process (torch). NLI orientation is Premise = evidence, Hypothesis = claim (FEVER/SciFact ordering) — **STATIC-VERIFIED** as already correct in `nli/entailment.py`; not changed.

## 4. Gap analysis (what already existed vs. what was missing)

**Already present (STATIC-VERIFIED — not reimplemented):** four-state verdict logic; evidence classification (SUPPORTING/CONTRADICTING/NEUTRAL/IRRELEVANT); per-NLI-item `degraded` flag; retrieval trace scaffolding; relevance/diversity gates; n8n V2 client + normalization; fail-soft fallbacks throughout.

**Genuinely missing (the 7 gaps this pass closed):**

1. Detector could not report whether the real classifier ran or the hardcoded baseline was used (§6).
2. BGE fallback scores were passed through as if they were real BGE relevance scores — no signal to the contrary (§13).
3. NLI had per-item degradation but **no engine-level** proof that the model loaded and a forward pass ran (§15).
4. No structured model-execution-proof block in the runtime trace (§26).
5. No fail-closed verification mode to *prove* the chain ran for certification (§28).
6. No explicit model-health signal — readiness was inferred from imports, not probed (§29).
7. Cache could serve a stored verdict as if it were a fresh model-backed run (§30).

Full anchored gap analysis is retained in the project memory file `project_halluciguard_verifier_hardening.md`.

## 5. Change set (precise, whitespace-insensitive deltas vs. branch HEAD)

**STATIC-VERIFIED.** Modified files (additive):

| File | +added | −removed | Purpose |
|------|-------:|--------:|---------|
| `agents/detector_agent/models.py` | 26 | 1 | 4 additive `detector_*` diagnostic fields on `DetectionResult` (§6) |
| `agents/detector_agent/detector.py` | 27 | 3 | populate diagnostics; baseline path flagged degraded (§6) |
| `agents/verifier_agent/rerankers/cross_encoder.py` | 128 | 1 | `diagnostics()` + per-run BGE execution proof (§13) |
| `agents/verifier_agent/nli/entailment.py` | 73 | 0 | `diagnostics()`/`is_loaded()` + engine-level NLI proof (§15) |
| `agents/verifier_agent/schemas/retrieval_trace.py` | 69 | 4 | `ModelExecutionTrace`; **RelationCheckTrace flat-schema fix** (§26) |
| `agents/verifier_agent/config/settings.py` | 22 | 0 | `certification_mode`, `cache_enabled` flags (§28/§30) |
| `agents/verifier_agent/api/pipeline.py` | 192 | 4 | cert enforcement, trace population, `health_check`, cache gating (§28/§29/§30) |
| `services/llm_detector_verifier_service.py` | 97 | 29 | surface detector diagnostics; detector-stage certification (§6/§28) |
| `scripts/test_llm_detector_verifier_slice.py` | 280 | 59 | **display-only** proof surfacing + `--certify` + A–H matrix (§35) |

New files:

| File | Lines | Purpose |
|------|------:|---------|
| `agents/verifier_agent/api/certification.py` | 139 | fail-closed enforcement helpers (§28/§29/§30) — dependency-light |
| `agents/verifier_agent/tests/test_verifier_hardening_regression.py` | 553 | §39 regression suite (one test per discovered bug) |

> Note on repo-wide git noise: `git status` reports ~400 files changed and tens of thousands of line deltas. This is **line-ending / whitespace normalization churn** across the whole tree and is **not** part of this work; the table above is the real, whitespace-insensitive change set for this task.

## 6. §6 — Detector execution diagnostics

**STATIC-VERIFIED.** `DetectionResult` gained four additive, safe-default fields: `detector_model_loaded`, `detector_inference_executed`, `detector_degraded`, `detector_model_source`. `detector.py` now computes `model_loaded` from the inference engine's real `_loaded` flag *before* inference, sets `inference_executed=True` only inside the real-forward-pass branch, and sets `degraded=True` with `model_source="baseline-heuristic"` on the `0.08/0.92` baseline branch. `_default_result()` (edge cases) reports `degraded=True`, `inference_executed=False`, `detector_model_source="default:<reason>"`. Result: **a failed detector load can no longer masquerade as real ML inference** — the crux requirement of §6. **UNIT-VERIFIED** by `TestDetectorDiagnostics` (baseline→degraded; loaded→executed with provenance; edge-case default; additive defaults on bare construction) — runs on Windows. Routing (HIGH→VERIFY) is unchanged; diagnostics are informational in the resilient path.

## 7. §13 — BGE reranker real-execution-only

**STATIC-VERIFIED.** `CrossEncoderReranker` gained `diagnostics()` and six per-run fields reset at the start of every `rerank()`. Success records `status="executed"`, `inference_executed=True`, `degraded=False`, device, latency, `scored_count`. The "model unavailable" path records `status="unavailable"`/`degraded=True`; an inference exception records `status="degraded"`. The fallback still preserves hybrid ordering (fail-soft) **but is now explicitly flagged so it can never be presented as a genuine BGE score** (§13). Return type is unchanged (`List[Passage]`); diagnostics ride on the instance, so no caller breaks. `score_gate_candidates()` was intentionally left uninstrumented (it is a bounded pre-gate, not the decision-grade rerank). **UNIT-VERIFIED** by `TestRerankerDiagnostics`, including an explicit anti-masquerade assertion that the fallback's `relevance_score` equals the input hybrid score while `inference_executed` is `False`.

## 8. §15 — DeBERTa NLI engine diagnostics

**STATIC-VERIFIED.** `NLIEngine` gained `is_loaded()`, `diagnostics()`, and per-run fields reset at the start of `batch_classify()`. Success records `status="executed"`, `inference_executed=True`, device, latency, `batch_size`; the unavailable and exception paths record `unavailable`/`degraded` and still return per-item degraded neutrals (fail-soft). The pre-existing per-item `degraded` flag is retained. Orientation was already correct and unchanged. **UNIT-VERIFIED** by `TestNLIDiagnostics` (unavailable→degraded with one result per evidence; executed via a fake pipeline; exception→degraded; empty→`not_run`).

## 9. §26 — Runtime trace contract (model-execution proof)

**STATIC-VERIFIED.** Added `ModelExecutionTrace` (component, model, loaded, inference_executed, degraded, status, device, latency_ms, scored_count|batch_size) to `retrieval_trace.py`, and wired `RetrievalTrace.reranker_execution` and `RetrievalTrace.nli_execution` (both `Optional`, default `None`). The pipeline populates them from `reranker.diagnostics()` / `nli_engine.diagnostics()` per sub-claim. `ModelExecutionTrace(**diag)` accepts both diagnostics dicts unchanged. **UNIT-VERIFIED** by `TestTraceContract`. The live runner now prints these as a `MODEL EXECUTION PROOF` block (Section 20).

## 10. §28 — Certification mode (fail-closed)

**STATIC-VERIFIED + UNIT-VERIFIED.** New `api/certification.py` (stdlib-only, so importable and testable even without pydantic/torch) provides: `CertificationError(stage, reason)`; `certification_enabled_from_env()`; and four enforcers — `enforce_detector`, `enforce_reranker`, `enforce_nli`, `enforce_retrieval_evidence`. Semantics (all **UNIT-VERIFIED**, 29 in-sandbox assertions + the pytest suite):

- Detector: fails if `detector_degraded` or not `detector_inference_executed`.
- Reranker: fails on `status ∈ {degraded, unavailable}` or `executed & degraded`; `not_run` (no passages) is allowed (owned by the evidence check) — this avoids over-failing legitimate empty-evidence UNVERIFIED verdicts.
- NLI: fails only when `evidence_present` and NLI degraded/unavailable/not-executed.
- Evidence: fails on `MOCK_MODE`, empty passages, or all-mock/placeholder/malformed passages.
- **Every enforcer is a strict no-op when `enabled=False`** — the default. Normal production is untouched.

## 11. §29 — Explicit model health

**STATIC-VERIFIED.** `VerificationPipeline.health_check(probe=True)` actually attempts to load BGE and NLI (real readiness, not import-inferred) and returns `{BGE_READY, NLI_READY, N8N_READY, CERTIFICATION_MODE, CACHE_ENABLED}`. `N8N_READY` reflects enablement + URL + auth presence. `DETECTOR_READY` is owned by the detector/service layer (the detector diagnostics of §6), not this method. Real readiness values are **LIVE-PENDING** (require torch/models). The Windows guide invokes `health_check` before the certification run.

## 12. §30 — Cache disabled under certification

**STATIC-VERIFIED.** `pipeline.__init__` computes `cache_enabled = settings.cache_enabled AND verifier_cache_enabled AND NOT certification_mode`. Cache reads and writes are both gated on `self.cache_enabled`. Therefore a cached verdict can never stand in as proof that the models executed this run. **UNIT-VERIFIED** by `TestCertificationDisablesCache` (Windows: constructs the pipeline with `CERTIFICATION_MODE=true` and asserts `cache_enabled is False`; and the normal-mode inverse).

## 13. Certification enforcement placement (control-flow correctness)

**STATIC-VERIFIED** (by reading `pipeline.py`). Enforcement points inside the per-claim loop: evidence check after retrieval/dedup; reranker check after diagnostics capture + trace population; NLI check after `record_nli_inference` with `evidence_present=bool(relevant_passages)`. Critically, an `except CertificationError: … raise` handler sits **before** the broad `except Exception` that would otherwise emit an `UNVERIFIED` error report — so a certification failure **propagates as a controlled failure and is never laundered into a fake verdict** (§28/§36/§45). Detector-stage certification is enforced in the service layer *before* routing to the verifier.

## 14. Discovered bugs and fixes

**KNOWN-ISSUE → FIXED. `RelationCheckTrace` three-way schema mismatch.** The producer (`api/pipeline.py`) constructed `RelationCheckTrace(claim_subject=…, check_result=…)` and the consumer (`scripts/verify_claim.py`) read `claim_subject`/`check_result`, but the model defined `claim_triple`/`evidence_triples`/`comparison_result`. Because Pydantic v2 defaults to `extra='ignore'`, the flat kwargs were **silently dropped** and this trace was **always empty** — a real observability defect. Fixed by rewriting `RelationCheckTrace` to the flat schema the producer/consumer actually use; a docstring records the prior bug. **STATIC-VERIFIED** the old field names have no remaining readers (the identically-named fields on `RelationCheckResult` in `relation_verifier.py` are a different class). Regression: `TestTraceContract.test_relation_check_trace_flat_fields_round_trip`.

**KNOWN-ISSUE → FIXED. Detector silent baseline masquerade.** Addressed by §6 diagnostics (Section 6); regression: `TestDetectorDiagnostics`.

## 15. §39 — Regression test suite

**STATIC-VERIFIED (compiles) + UNIT-VERIFIED (cert subset ran in sandbox).** New hermetic suite `tests/test_verifier_hardening_regression.py` (553 lines, no torch/network — models faked, fallbacks forced). Test→bug mapping:

| Class | Guards |
|-------|--------|
| `TestCertificationEnvFlag` | env parsing of `CERTIFICATION_MODE` |
| `TestEnforceReranker` / `…Nli` / `…Detector` / `…RetrievalEvidence` | fail-closed on fallback/mock/empty; no-op when disabled (§28) |
| `TestDetectorDiagnostics` | §6 baseline-vs-executed + safe defaults |
| `TestRerankerDiagnostics` | §13 fallback flagged + anti-masquerade + reset-between-runs |
| `TestNLIDiagnostics` | §15 engine execution proof |
| `TestTraceContract` | §26 `ModelExecutionTrace` round-trip + `RelationCheckTrace` fix |
| `TestCertificationDisablesCache` | §30 cache disabled under certification |

The certification-helper classes are dependency-light and were executed in the sandbox (**29/29 assertions passed**). The pydantic/model/pipeline classes skip cleanly where deps are absent and run for real on Windows (the skip message names the missing dependency).

## 16. §34 — Existing test suite preservation

**STATIC-VERIFIED.** No existing test file was modified or deleted for this pass; the new suite is purely additive. `agents/verifier_agent/tests/` still contains all **23** `test_*.py` files (verified by listing). The full suite (`pytest -q`) and per-agent runs are documented in the Windows guide.

## 17. §7 — Static verification results

**STATIC-VERIFIED.** Full-repo `py_compile` sweep: **328/328** Python files compile cleanly (excluding vendored/cache dirs). All nine modified files and both new files compile. Behavioral wiring in `pipeline.py` (imports, `__init__` cache/cert gating, enforcement calls, trace population, `health_check`, the `except CertificationError: raise` ordering, and cache read/write gates) was confirmed by `grep` + direct read.

## 18. Unit verification results (in-sandbox)

**UNIT-VERIFIED (reproducible).** A committed, zero-dependency smoke test — `scripts/verify_certification_sandbox.py` (imports only the stdlib + `api/certification.py`, so it runs even without torch/transformers/pydantic/pytest) — exercises every branch of the fail-closed guards and prints a real PASS/FAIL per case. Latest sandbox run (Python 3.10): **41/41 cases passed, 0 failed, exit code 0.** Coverage: `CertificationError` shape (`[certification:{stage}] {reason}` + stage/reason attrs); env-flag parsing (`true/1/yes/on/TRUE`→enabled, `false/off/""`/unset→disabled); reranker (degraded/unavailable/executed+degraded fail; executed-clean & `not_run` pass; disabled & empty-diag no-op; failure stage `bge_reranker`); NLI (degraded/unavailable/not-executed-with-evidence fail; executed+evidence pass; no-evidence & disabled no-op; stage `deberta_nli`); evidence (mock-mode/empty/all-mock/all-malformed fail; real & real+mock-mix pass; disabled no-op; stage `retrieval`); detector (degraded/not-executed fail; real pass; disabled & empty no-op; stage `detector`). Reproduce with `python scripts/verify_certification_sandbox.py`. These cases mirror the pytest classes one-to-one; the pytest suite additionally covers the pydantic/model/pipeline paths on Windows. (This committed script supersedes the earlier ad-hoc 29-assertion harness.)

The dependency probe run alongside it confirms **torch, transformers, sentence-transformers, pydantic, pytest, and httpx are all absent in this sandbox** — the concrete reason the model chain (Sections 20–21) is **LIVE-PENDING** and must be validated on Windows.

## 19. Instrumentation of the live runner (display-only)

**STATIC-VERIFIED.** `scripts/test_llm_detector_verifier_slice.py` was enhanced **display-only** (no verifier logic touched): the `DETECTOR` block now prints the §6 diagnostics; each claim prints a `MODEL EXECUTION PROOF` block from `reranker_execution`/`nli_execution`; a `--certify` flag sets `CERTIFICATION_MODE=true` + `CACHE_ENABLED=false` and prints a banner; certification failures render as a clean, controlled `CERTIFICATION FAILURE` block (not a traceback); and the matrix was extended to **A–H**. Runner compiles cleanly.

## 20. §35 — Live validation plan and matrix A–H (LIVE-PENDING)

**LIVE-PENDING.** Full procedure in `LIVE_VALIDATION_WINDOWS.md`. Matrix: A (Allu Arjun father — question), B (Chiranjeevi false relation), C (Paris capital — true), D (Eiffel Tower in London — false), E (aspirin — healthcare true), F (nonsense — must never be VERIFIED), G (time-sensitive current office-holder), H (multi-sentence answer proving decomposition → multiple per-claim verdicts). The operator confirms, per claim: non-empty LLM draft; detector diagnostics show real inference; `MODEL EXECUTION PROOF` shows `status=executed` for BGE and NLI; `bge_score ≠ adapter_score`; decision-grade citations with full NLI distribution; and a four-state verdict. Expected verdicts are documented as expectations to confirm, **not** guarantees — a disagreement is a genuine finding to report, never a reason to alter thresholds (§36).

## 21. §43 — Acceptance criteria status

**LIVE-PENDING.** The Verifier is "RUNTIME-VALIDATED" only when ≥1 real claim demonstrates the full chain (LLM → Detector inference → n8n V2 → real evidence → BGE inference → DeBERTa inference → EvidenceScorer → four-state verdict) **and** the trace proves each stage executed. The instrumentation to *prove* this now exists (Sections 6–9, 19); executing it requires the Windows machine. Until that run passes, this report asserts **STATIC-VERIFIED / UNIT-VERIFIED** only and explicitly does not claim runtime validation or production readiness.

## 22. §41 — Security and secret handling

**STATIC-VERIFIED.** No secrets are printed or hardcoded by any change in this pass. `certification.py` handles only diagnostic dicts and passage metadata — no keys. The runner reads secrets from env only; the n8n boundary test prints `Secret Set : YES (masked)`. No `.env` or secret material is committed by this work. Certification error messages carry a stage + reason, never secret values. `OPENROUTER_API_KEY`, `N8N_WEBHOOK_SECRET`, `TAVILY_API_KEY`, `NVD_API_KEY` remain env-only.

## 23. §45 — Stop-condition check, known issues, and sign-off

**Absolute stop conditions (§45) — none triggered by this pass:** no architecture redesign was required; the data contract was extended additively (no incompatibility); the n8n contract was untouched; no test required fabricated data; and no verdict is produced without evidence (certification enforces the opposite). The one place the spec says to *stop and report* — a model that cannot load, or BGE/NLI/detector falling back — is exactly what certification mode now surfaces as a controlled failure rather than silently working around.

**Known issues / limitations (honest):**

- **All model execution is LIVE-PENDING.** The sandbox cannot run torch/transformers or reach OpenRouter/n8n, so BGE/NLI/detector real execution, the A–H verdicts, and `health_check` readiness values are unproven here and must be generated on Windows.
- **Pydantic-dependent tests are skipped in-sandbox** (no pydantic). They are authored to run on Windows; only the stdlib-only certification subset was executed here.
- **`git status` shows repo-wide EOL/whitespace churn** unrelated to this task; the real change set is Section 5.
- **Time-sensitive case G** depends on live retrieval recency and on the base LLM's own answer; treat its verdict as a live finding.

**Sign-off:** All in-scope hardening (§6, §13, §15, §26, §28, §29, §30, §39) is implemented, **STATIC-VERIFIED**, and — for the dependency-light certification core — **UNIT-VERIFIED** in the sandbox. Certification mode is **off by default**; the resilient production path is behavior-preserving. The task is **not** declared complete: it awaits the **LIVE-PENDING** Windows validation run (Sections 20–21) to reach §43 RUNTIME-VALIDATED status.

---

# Appendix A — Live Results (fill in from the Windows run)

**How to use this appendix.** Everything above labeled **LIVE-PENDING** becomes **LIVE-VERIFIED** by running `LIVE_VALIDATION_WINDOWS.md` Steps 1–4 on the Windows machine (Python 3.13, RTX 3050) and pasting the printed output into the tables below. Do **not** hand-edit expected verdicts to make a row "pass"; paste what actually printed. If a row fails, that is a real finding — record it as **KNOWN-ISSUE** and fix the named stage (usually a model path / download / network issue), then re-run. Fill each **Status** cell with one of: **LIVE-VERIFIED** / **KNOWN-ISSUE** / **not run yet**.

Set the run header once:

```
Run date       : __________________________
Machine        : Windows __ / Python 3.13.__ / GPU: RTX 3050 6GB
Commit / branch: verifier-v2-stabilization @ __________
```

## A.0 — In-sandbox pre-check (already done — real, reproducible)

These two rows were executed in the offline sandbox and are **not** pending. Re-run the smoke test anywhere to reproduce.

| Check | Command | Result | Status |
|-------|---------|--------|--------|
| Certification guards | `python scripts/verify_certification_sandbox.py` | **41/41 passed, exit 0** | **UNIT-VERIFIED** |
| Compile touched files | `python -m py_compile <12 files>` | **12/12 clean** | **STATIC-VERIFIED** |

> The same run's dependency probe showed torch / transformers / sentence-transformers / pydantic / pytest / httpx absent in the sandbox — that is why A.1–A.5 below can only be produced on Windows.

## A.1 — Step 1: offline unit tests (`pytest`)

Paste the summary line from each command (e.g. `120 passed, 3 skipped in 42.1s`):

| Suite | Command | Result (paste) | Status |
|-------|---------|----------------|--------|
| Full suite | `pytest -q` | `__________________` | ☐ |
| Verifier only | `pytest -q agents/verifier_agent/tests` | `__________________` | ☐ |
| Hardening regression (§39) | `pytest -v agents/verifier_agent/tests/test_verifier_hardening_regression.py` | `__________________` | ☐ |

If any test is **SKIPPED**, note the dependency it named: `__________________`. (NumPy-2 ABI error on 3.13? see the guide's `pip install "numpy<2"` note — environment issue, not a code defect.)

## A.2 — Step 2: n8n boundary (FROZEN retrieval V2 is live)

| Claim | HTTP | Success | Workflow ver | Passages > 0 | Latency | Status |
|-------|-----:|:------:|:------------:|:------------:|--------:|--------|
| "Paris is the capital of France." | ☐ 200 | ☐ | `2.__` | ☐ | ___ ms | ☐ |
| "The Eiffel Tower is located in London." | ☐ 200 | ☐ | `2.__` | ☐ | ___ ms | ☐ |
| "Aspirin is used to relieve mild to moderate pain." (healthcare) | ☐ 200 | ☐ | `2.__` | ☐ | ___ ms | ☐ |

`Secret Set : YES (masked)` seen on each call? ☐  (Do not paste the secret itself.)

## A.3 — Step 3: full vertical slice, matrix A–H (the completion criterion)

Command: `python scripts/test_llm_detector_verifier_slice.py --all-tests`. For each test read the trace and record the **model-execution proof** plus the verdict. A row is **LIVE-VERIFIED** only when the detector shows `Inference Executed: True` (not baseline), BGE shows `status=executed`, NLI shows `status=executed`, and a four-state verdict printed.

| Test | Claim | Detector `Inference Executed` / `Degraded` | BGE `status` / `device` | NLI `status` | `bge_score ≠ adapter_score`? | Verdict (actual) | Expected | Match? | Status |
|------|-------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|--------|
| A | Who is the father of Allu Arjun? | ☐ True / ☐ False | ☐ executed / ___ | ☐ executed | ☐ | `__________` | VERIFIED (if answer correct+supported) | ☐ | ☐ |
| B | Allu Arjun's father is Chiranjeevi. | ☐ True / ☐ False | ☐ executed / ___ | ☐ executed | ☐ | `__________` | CONTRADICTED | ☐ | ☐ |
| C | Paris is the capital of France. | ☐ True / ☐ False | ☐ executed / ___ | ☐ executed | ☐ | `__________` | VERIFIED | ☐ | ☐ |
| D | The Eiffel Tower is located in London. | ☐ True / ☐ False | ☐ executed / ___ | ☐ executed | ☐ | `__________` | CONTRADICTED | ☐ | ☐ |
| E | Aspirin is used to relieve mild/moderate pain. (healthcare) | ☐ True / ☐ False | ☐ executed / ___ | ☐ executed | ☐ | `__________` | VERIFIED | ☐ | ☐ |
| F | The moon is made of green cheese and orbits Jupiter. | ☐ True / ☐ False | ☐ executed / ___ | ☐ executed | ☐ | `__________` | **never** VERIFIED | ☐ | ☐ |
| G | Who is the current president of the United States? | ☐ True / ☐ False | ☐ executed / ___ | ☐ executed | ☐ | `__________` | VERIFIED iff live evidence supports; else UNVERIFIED | ☐ | ☐ |
| H | Tell me three facts about the Eiffel Tower. | ☐ True / ☐ False | ☐ executed / ___ | ☐ executed | ☐ | `__________` (≥2 `Claim #` blocks?) ☐ | multiple per-claim verdicts (decomposition) | ☐ | ☐ |

**Detector sanity:** does `Hallucination Prob` vary across A–H (not flat `0.0800`)? ☐ Yes ☐ No → if flat, HaluEval didn't load (`HALUEVAL_MODEL_PATH`).

Optional — paste one full trace (recommended: Test B or D, where a CONTRADICTED verdict best proves the whole chain):

```
<paste one FULL VERTICAL SLICE EXECUTION TRACE here — LLM draft, detector block,
 n8n retrieval, MODEL EXECUTION PROOF (BGE + NLI), decision-grade citations with
 adapter_score/bge_score/NLI distribution, and the final verdict>
```

## A.4 — Step 4: certification run (fail-closed proof, §28/§29/§30)

Health probe (actually loads BGE + NLI): `python -c "... VerificationPipeline().health_check(probe=True) ..."`

| Field | Expected | Actual (paste) |
|-------|----------|----------------|
| `BGE_READY` | true | ☐ |
| `NLI_READY` | true | ☐ |
| `N8N_READY` | true | ☐ |
| `CERTIFICATION_MODE` | (env) | ☐ |
| `CACHE_ENABLED` | (env) | ☐ |

Certified run: `python scripts/test_llm_detector_verifier_slice.py --certify --all-tests`. Each claim is either **certified** (trace prints with BGE/NLI `status=executed`) or a **controlled failure** (`CERTIFICATION FAILURE [certification:<stage>]`). Both are honest outcomes — do **not** drop `--certify` to force a verdict.

| Test | Outcome (`certified` / `controlled-failure:<stage>`) | Status |
|------|------------------------------------------------------|--------|
| A | `__________` | ☐ |
| B | `__________` | ☐ |
| C | `__________` | ☐ |
| D | `__________` | ☐ |
| E | `__________` | ☐ |
| F | `__________` | ☐ |
| G | `__________` | ☐ |
| H | `__________` | ☐ |

## A.5 — §43 acceptance (flips the whole report to RUNTIME-VALIDATED)

The Verifier is **RUNTIME-VALIDATED** once **at least one** real claim demonstrates the full chain with every stage proven to have executed. Certify one row here:

```
Claim used for acceptance : ______________________________________
LLM draft printed         : ☐ non-empty
Detector inference        : ☐ Inference Executed=True, Degraded=False, provenance=<model id/dir>
n8n retrieval             : ☐ Retrieved Count > 0, Workflow ver 2.x
BGE                       : ☐ MODEL EXECUTION PROOF -> status=executed, device=____
DeBERTa NLI               : ☐ MODEL EXECUTION PROOF -> status=executed
EvidenceScorer + verdict  : ☐ one of VERIFIED/CONTRADICTED/UNVERIFIED/CONFLICTED
Certified (--certify)     : ☐ certified (not a controlled failure)

ACCEPTANCE (§43): ☐ RUNTIME-VALIDATED   ☐ still LIVE-PENDING (reason: __________)
```

When A.5 is checked **RUNTIME-VALIDATED**, update Sections 20–21 and the Section 23 sign-off from LIVE-PENDING to LIVE-VERIFIED, and record any A.3 mismatch or A.4 controlled-failure as a KNOWN-ISSUE with its root cause. Until then, per §32/§36/§46 the task remains **not complete** and **not production-ready**.
