# HalluciGuard — Remaining work & documented limitations

**Branch:** `claude/integration-20261001` (16 fix commits on top of `origin/main` + the Codex baseline).
**Status at this snapshot:** Critical + all High + most Medium bugs fixed and green; the items below remain.

---

## A. LOW code items — status

**Fixed this pass (7):** #21 (removed the dead unreachable unknown-label counter loop),
#26 (tokenization diagnostic gated behind `HG_DETECTOR_TRACE_TOKENS` so the hot path no
longer double-tokenizes), #32 (an n8n HTTP-200 with no recognizable evidence field is now
flagged in the trace, distinct from a genuine empty result), #34 (the Judge VERIFY_AGAIN
route is computed from the post-increment retry count, matching the edge), #37
(`score_gate_candidates` now sets execution diagnostics so a fallback is distinguishable from
a real BGE run), #42 (NLI `_canonical_label` tolerates known head decorations like
`entailment_score` without loose substring matching), M12 (`legacy_n8n_output` is populated
as a pure diagnostic from any n8n verdict/confidence block).

**Documented, not code-changed (rationale):**

| # | File | Why not changed now |
|---|---|---|
| #15 | `verifier_agent/formatters/citation_formatter.py` | per-item `bge_score` fallback provenance needs the reranker's degraded status threaded into the formatter; today it lives at request level (`RetrievalTrace.reranker_execution.degraded`). Deferred to avoid a schema + threading change — the degraded signal IS available at request level. |
| #19 | `n8n/halluciguard_n8n_v2_updated.json` | the "Production" workflow's missing Analyze-Claim node is a workflow-JSON change that only takes effect on an n8n **re-import** (unvalidatable here), and n8n retrieval is **disabled by default** (`N8N_RETRIEVAL_ENABLED=false`), bounding impact. |
| #33 | `services/llm_detector_verifier_service.py` | wiring `enforce_detector` into a production node is a **behavior change** (it would hard-reject degraded grounded runs); left as the documented shadow/opt-in design pending a measured acceptance policy. |

None of the above affects a verdict, acceptance decision, or security posture.

## B. Documented LIMITATIONS (works-as-designed / needs data or infra — NOT code defects)

- **#23 — Dev-split reuse.** The dev split selects early-stopping, temperature, and operating thresholds, then is also reported. The honest out-of-sample number is `evaluate()` on the untouched test split with frozen thresholds; dev metrics are labelled `selected_on: dev`. Accepted; the real metric is the test-split evaluation.
- **#24 — Phase-1 UQ head inert in production.** No production entrypoint supplies a same-generation `generation_trace`, so the shadow head always returns `unavailable`/`disabled`. Correct by design: it never fabricates a number. Activating it needs real generation traces + labels (the standing calibration-data blocker).
- **#25 — Phase-1 artifact "integrity" is self-referential.** The hashes live in the files they certify (guards accidental corruption, not tampering). **Non-exploitable today** because the fast path is dead-code-unreachable. Defense-in-depth only; would need a signed artifact to harden.
- **#29 — Numeric/entity conflicts don't change the detector label.** By design (HG-007): they set `requires_verification` and warn, never flip the neural label. This is a triage-layer boundary; the Verifier/Judge adjudicate. Accepted.
- **#36 — `_normalize_class_mapping` hard-raises.** This is a defensible fail-closed hardening (a malformed distribution raises instead of silently coercing); production callers wrap `detect()` and fail closed to the Verifier. Not a weakening.
- **#41 — Certification doesn't fail-closed on dense (hybrid) degradation.** Documented in `certification.py`: it gates the reranker, NLI, and evidence presence, not dense recall. Dense degradation is a recall (not correctness) loss, recorded truthfully in the trace. Accepted boundary.
- **#42 — Narrowed `_canonical_label`.** Exact-set NLI label matching could drop a decorated label from a *different* model; the pinned model emits clean labels, and any mismatch degrades to NEUTRAL (safe). Suspected-only; low risk under the pinned model.

## C. Deliverable phases still to run

1. **Full-repo test suite** on this snapshot (in progress) — expected to confirm the 2 previously-failing Paris stabilization tests now PASS.
2. **Live end-to-end** on ~6 diverse hard claims (supported / contradicted / numeric / negation / temporal / conflicting-source) with real retrieval + models.
3. **Frontend ↔ backend** wiring validation.
4. **Push `claude/integration-20261001` + open PR to `main`** (gh not installed → push + compare link; merge when validated).

> Deployment BUILD validation (Docker image / HF Space) remains **BLOCKED** in this environment (no container/HF build host) — see H5 commit.
