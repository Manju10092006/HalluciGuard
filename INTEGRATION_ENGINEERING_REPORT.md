# HalluciGuard — Integration Engineering Report

**Task:** Detector → n8n Retrieval V2 → BGE → DeBERTa → Final Verdict (vertical slice)
**Branch:** `verifier-v2-stabilization`
**Report date:** 2026-08-23
**Environment where this report was produced:** offline sandbox (no external network, no torch, Python 3.10.12). All network/model-dependent facts are therefore labeled **LIVE-PENDING** and must be produced by running `LIVE_VALIDATION_WINDOWS.md` on the Windows machine.

> **Completion status (spec §32).** This report does **not** declare the task complete and does **not** declare production readiness. Per §32, completion requires at least one real claim to traverse the full runtime chain (real LLM → Detector → n8n V2 → real evidence → real BGE → real DeBERTa → EvidenceScorer → four-state verdict) with all runtime inputs visible. That runtime proof can only be generated on the Windows machine and is captured in items **8, 9, 13, 14, 15** below as **LIVE-PENDING**. Everything labeled **STATIC-VERIFIED** was confirmed by reading the source and by `py_compile`/grep in the sandbox.

---

## 1. Files inspected (STATIC-VERIFIED)

Core integration path read end-to-end:

- `services/base_llm_service.py` — OpenRouter/Qwen client, retries, secret redaction.
- `agents/detector_agent/detector.py`, `config.py`, `models.py`, `gate.py` — HaluEval detector, thresholds, routing enums.
- `services/llm_detector_verifier_service.py` — the slice orchestrator (LLM → Detector → gate → Verifier).
- `services/n8n_retrieval_client.py` — n8n V2 boundary client + response normalization.
- `agents/verifier_agent/api/pipeline.py` — verification pipeline (retrieval → rerank → NLI → scoring → verdict).
- `agents/verifier_agent/models/model_manager.py` — BGE/NLI/embedding/zero-shot loaders.
- `agents/verifier_agent/formatters/citation_formatter.py` — passage → EvidenceItem mapping.
- `agents/verifier_agent/schemas/models.py` — `VerdictLabel`, `EvidenceClassification`, `Passage`, `VerifierInputV2`.
- `agents/verifier_agent/config/settings.py` — model names, n8n config, gates.
- `scripts/test_n8n_retrieval.py`, `scripts/test_llm_detector_verifier_slice.py` — the two runners.
- Existing tests: `agents/verifier_agent/tests/test_n8n_retrieval.py`, `orchestration/tests/test_llm_detector_verifier_slice.py`.
- `pytest.ini`.

**Static verification result:** 325/325 Python files compiled cleanly (`py_compile`) in the sandbox on Python 3.10.12.

## 2. Files modified (STATIC-VERIFIED)

Deliberately minimal — nothing on the DO-NOT-TOUCH list (§31) was changed. No architecture rewrite, no new agents, no change to models, thresholds, LangGraph topology, or the n8n workflow.

- `scripts/test_n8n_retrieval.py` — **docstring only.** Removed the stale "calls n8n health endpoint (GET)" description and renumbered so the docstring matches the code, which already performed a POST-only retrieval test (§6/§26). No executable code changed.
- `agents/verifier_agent/tests/test_slice_integration_supplement.py` — **new test file** (additive; no existing test touched). See items 12–13.

## 3. Existing components reused (STATIC-VERIFIED)

The slice is wiring over components that already existed; nothing was reimplemented:

- **Base LLM:** `BaseLLMService` (OpenRouter/Qwen).
- **Detector:** `DetectorAgent` (HaluEval DistilBERT) with its existing thresholds.
- **Retrieval boundary:** `N8NRetrievalClient` against the frozen V2 webhook.
- **Reranker:** `BAAI/bge-reranker-large` via `ModelManager.load_reranker_model` (CrossEncoder).
- **NLI:** `cross-encoder/nli-deberta-v3-base` via `ModelManager.load_nli_model` (HF pipeline).
- **Scoring/verdict:** existing `EvidenceScorer` / pipeline aggregation and `VerdictLabel` taxonomy.
- **Decomposition:** existing `ClaimDecomposer` (sub-claims capped at 5).

## 4. N8N integration point (STATIC-VERIFIED)

- Client: `services/n8n_retrieval_client.py`, method `retrieve_evidence(...)`.
- Endpoint (frozen): `https://manjusogala.app.n8n.cloud/webhook/halluciguard-verify-v2`.
- Auth: header mode, `X-API-Key: <secret>`; secret read from `N8N_WEBHOOK_SECRET` (never hard-coded; boundary test prints only `YES (masked)`).
- Called from the pipeline exactly once per claim at `api/pipeline.py:317` (`n8n_res = await self.n8n_client.retrieve_evidence(...)`). n8n performs **retrieval only**; there is no verdict field on `N8NRetrievalResult`.

## 5. N8N request schema (STATIC-VERIFIED)

`retrieve_evidence` serializes a V2 request carrying: claim text, `domain`, `retrieval_mode` (default `hybrid`), and a `request_id`, sent as JSON with the `X-API-Key` header and a `N8N_TIMEOUT_SECONDS` (default 60s) timeout. Request serialization is covered by `test_request_payload_serialization` in the existing boundary test.

## 6. N8N response mapping (STATIC-VERIFIED)

The raw V2 response is normalized into `Passage` objects. The score-separation contract is enforced at the boundary (`n8n_retrieval_client.py:182-183`):

```python
relevance_score=0.0,                     # reserved for the Python BGE reranker
source_confidence_hint=round(max(0.0, min(1.0, adapter_score)), 4),  # n8n adapter hint
```

So the n8n "adapter" score is carried as `source_confidence_hint` and the BGE slot (`relevance_score`) is left at 0.0 for Python to fill. A retrieval trace (latency, workflow version, sources, counts, Tavily fallback, stage traces) is also captured. Schema/trace covered by `test_exact_v2_response_schema`; failure modes (403/500/timeout/invalid-JSON/missing-evidence/malformed-items) covered by the existing boundary tests.

## 7. Detector routing behavior (STATIC-VERIFIED)

- Thresholds (`detector.py:128-137`): `prob ≤ 0.30 → LOW`, `prob ≥ 0.50 → HIGH`, else `MEDIUM`.
- Effective gate (`llm_detector_verifier_service.py:152-155`):

  ```python
  should_verify = force_verifier or decision == "VERIFY" or risk_tier in {"MEDIUM", "HIGH"}
  ```

  Net behavior: **LOW → accept (verifier skipped); MEDIUM/HIGH → verifier runs (n8n retrieval).** This matches the spec's risk gate.
- **Known silent-degradation path (documented, not a defect introduced here):** if the HaluEval model fails to load, `detect()` returns a hardcoded baseline `hallucination_probability=0.08 / confidence=0.92` (`detector.py:100-107`) → LOW → verifier skipped. The A–E runner sets `force_verifier=True`, so the chain runs regardless; the live checklist still requires confirming the detector loaded (probabilities vary; not a constant `0.0800`).

## 8. BGE execution evidence — **LIVE-PENDING**

- **Wiring (STATIC-VERIFIED):** reranker invoked at `api/pipeline.py:402` with the real claim and the real retrieved passages; loader is `ModelManager.load_reranker_model` (`CrossEncoder("BAAI/bge-reranker-large")`). Unit test `test_bge_reranker_receives_real_claim_and_evidence` asserts BGE receives the real claim + real snippet.
- **Real-model output (LIVE-PENDING):** the actual BGE relevance scores for A–E cannot be produced in the sandbox (no torch/model/network). To be filled from the Windows run; must show the `Loaded reranker model 'BAAI/bge-reranker-large'` log line, no reranker fallback warning, and `bge_score` values that differ from `adapter_score`. **Not fabricated here.**

## 9. DeBERTa execution evidence — **LIVE-PENDING**

- **Wiring (STATIC-VERIFIED):** NLI invoked at `api/pipeline.py:427` (`batch_classify`) over real (claim, evidence-snippet) pairs; loader is `ModelManager.load_nli_model` (`cross-encoder/nli-deberta-v3-base`). Existing test `test_bge_and_deberta_receive_real_claim_and_evidence_pairs` asserts NLI receives the real claim + snippet.
- **Real-model output (LIVE-PENDING):** the actual entailment/contradiction/neutral distributions for A–E cannot be produced in the sandbox. To be filled from the Windows run; must show the NLI model loaded (not `degraded: True`). **Not fabricated here.**

## 10. Evidence scoring result (STATIC-VERIFIED wiring; values LIVE-PENDING)

- `adapter_score` and `bge_score` are preserved as distinct fields on each `EvidenceItem` (`citation_formatter.py:62-63, 74-76`); `bge_score` is never set equal to `adapter_score`. End-to-end separation asserted by `test_adapter_and_bge_scores_remain_separate_end_to_end`.
- Classification uses the four evidence states `SUPPORTING / CONTRADICTING / NEUTRAL / IRRELEVANT`; neutral/irrelevant are preserved (not coerced to contradiction) — asserted by the existing `test_evidence_scorer_preserves_irrelevant_and_neutral`.
- Actual per-claim scores for A–E are **LIVE-PENDING**.

## 11. Final verdict result (STATIC-VERIFIED taxonomy; verdicts LIVE-PENDING)

- Verdict is computed in Python from the four-state `VerdictLabel` enum (`schemas/models.py:20-24`: `verified / contradicted / unverified / conflicted`); `N8NRetrievalResult` carries no verdict. Enforced by `test_final_verdict_is_python_four_state_taxonomy`.
- Actual A–E verdicts are **LIVE-PENDING** (expected sanity check: A VERIFIED, B CONTRADICTED, C VERIFIED, D CONTRADICTED, E VERIFIED — to be confirmed, not assumed).

## 12. Test cases (STATIC-VERIFIED — §29's 12 cases)

Eleven of the twelve §29 cases were already covered by the two existing suites (kept intact per "do NOT delete existing tests"); the twelfth plus reinforcing invariants were added in the new supplement:

| § | Case | Location |
|---|------|----------|
| 1 | Detector LOW → verifier skipped | `orchestration/tests/test_llm_detector_verifier_slice.py` |
| 2 | Detector MEDIUM/HIGH → verifier runs | same |
| 3 | n8n success → evidence normalized | `agents/verifier_agent/tests/test_n8n_retrieval.py` |
| 4 | n8n HTTP failure (403/500/timeout) | same |
| 5 | n8n invalid JSON | same |
| 6 | empty evidence | same |
| 7 | **BGE receives real claim + evidence** | **`test_slice_integration_supplement.py` (new)** |
| 8 | DeBERTa receives real claim + snippet | `test_n8n_retrieval.py` |
| 9 | scoring preserves neutral/irrelevant | same |
| 10 | verdict computed in Python (4-state) | same (+ reinforced in supplement) |
| 11 | no duplicate retrieval on n8n success | same |
| 12 | slice result JSON-serializable | `orchestration/tests/test_llm_detector_verifier_slice.py` |

The supplement also adds the §9 end-to-end `adapter_score != bge_score` guard. All supplement tests are hermetic (n8n/BGE/NLI mocked) and `py_compile`-clean.

## 13. Unit test results — **LIVE-PENDING (run on Windows)**

The sandbox lacks `pytest`/`torch`/`transformers`, so the suite was not executed here — only `py_compile` was run (clean). Run `pytest -q` on Windows and paste the pass/fail summary here. Expected: full suite green, including the three integration files.

## 14. Live test results (A–E) — **LIVE-PENDING (run on Windows)**

Cannot be produced in the sandbox. Run `python scripts/test_llm_detector_verifier_slice.py --all-tests` and record, per claim: LLM draft, detector prob/tier/decision, n8n sources + retrieved count, per-evidence `adapter_score`/`bge_score`/NLI scores/classification, and the final four-state verdict. **No live results are asserted in this report.**

## 15. Latency — **LIVE-PENDING (run on Windows)**

Record `latency_ms` per claim from the runner (first run includes one-time model download/load; re-run for steady-state). n8n boundary latency is reported separately by `scripts/test_n8n_retrieval.py`.

## 16. Remaining limitations (STATIC-VERIFIED observations)

- **Live chain unproven in this environment.** Items 8, 9, 13, 14, 15 require the Windows machine; this report intentionally does not fabricate them (§33).
- **Detector baseline fallback is silent.** If the HaluEval model doesn't load, queries are accepted as LOW with `prob=0.08` and the verifier is skipped. Mitigated for A–E by `force_verifier=True`; still requires confirming the detector loaded in general use.
- **BGE/NLI fail-soft.** `cross_encoder.py` and `entailment.py` degrade rather than error; degraded NLI is flagged `degraded: True` and excluded from decision-grade evidence, which can push a verdict to `UNVERIFIED`. The live checklist requires confirming neither fell back.
- **Model download gating asymmetry.** `load_reranker_model` and the NLI primary path do not pass `local_files_only`, so they download from HF when uncached regardless of `allow_model_downloads` (fine on Windows with network); only embedding/zero-shot loaders honor that flag. Set `ALLOW_MODEL_DOWNLOADS=true` for the first Windows run.
- **Python version drift.** Sandbox is 3.10.12; project targets 3.13. `py_compile` passed under 3.10, but the authoritative run must be on the project's 3.13 environment.

---

### Bottom line

The integration is **statically complete and internally consistent**: the components are correctly wired, scores are kept separate, retrieval happens once, and the four-state verdict is computed in Python. It is **not yet runtime-proven** — per §32 that requires one real claim through the full live chain on Windows. Follow `LIVE_VALIDATION_WINDOWS.md`, then fill items 8, 9, 13, 14, 15 with the real output. Do not declare production readiness until those are green and the models are confirmed to have run for real.
