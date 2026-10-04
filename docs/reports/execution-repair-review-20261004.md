# Seven-agent execution repair and Memory inspection review — 2026-10-04

## Scope and publication boundary

This is a review of the `docs/restructure-layout` working branch, starting at `623d618d3e856511357a2c14662c061a0e3a10ad`. It is not a merge of the other Detector or Memory worktrees. At inspection, `origin/main` was `82cdc6f`; the branches had divergent histories. Publishing this branch does not update main or deploy the application. A separate integration review must preserve the newer main implementation.

The existing production checkpoint, tokenizer and calibration are retained. No new model was trained, no probability thresholds were tuned, and no measured improvement in hallucination-detection accuracy is claimed. Verification-first defaults remain enabled. Missing scores are unavailable, not fabricated zeros; real zero scores remain zero.

## What changed and why

1. Generation receives the actual question and a focused-answer instruction, rather than a demonstration answer. Extra generated facts are still subject to verification.
2. Claim extraction preserves explicitly stated creator names, years, Unicode text and coordinated claims. JSON truncation/errors and fallback extraction are visible. Ambiguous qualifiers and negation are not silently turned into affirmative claims.
3. Passage deduplication preserves distinct excerpts from one URL, including conflicting dates or negation. Only duplicate source/excerpt records are collapsed; independent sources retain provenance. Early ranking no longer removes a whole source family before neural relevance scoring.
4. Wikipedia founding/creation retrieval selects contiguous relevant source excerpts and retains real source identifiers and links. Supplementary retrieval is recorded separately from n8n. This does not make every selected passage sufficient evidence.
5. Sparse/dense retrieval and BGE reranking expose actual attempts, execution, failure and fallback. Fusion contributions are recorded. Cache hits separate historical origin diagnostics from components executed on the current request, and claim/query bindings prevent cross-claim attribution.
6. Narrow relation guards avoid unsupported decisions from negated, qualified, different-organization or compound-name evidence. Existing deterministic relation assumptions and score adjustments are not a substitute for neural accuracy. Raw neural NLI scores remain distinguishable from adjusted decision scores.
7. Detector lazy model ownership is per instance, protected by a lock, and assigned only after successful initialization so a failed load can be retried. Bridge/service failures with a valid draft still reach verification. Malformed, missing or nonfinite scores cannot silently authorize a bypass.
8. Direct-service certification is checked after the Verifier attempt, not before evidence-free triage. A service without grounded Detector inference remains honestly uncertified; this change does not invent a grounded score.
9. ReVerifier extracts the complete corrected answer, checks report coverage, and requires successful verification before release. A narrow founded/established topical alias avoids rejecting an otherwise on-topic Microsoft repair. Conditional repair stages are not forced to run for correct answers.
10. The narrated runner prints observed handoffs, claims, source URLs, BGE and raw NLI scores, decisions, conditional stages, Memory outcomes and final response. Live calls and replay are explicitly different. Generic validation runs have generic case labels rather than misleading Dell/Microsoft labels.
11. Validation runs use fresh diagnostic Memory/cache files and never overwrite old reports or model artifacts. Runtime observations remain local, not committed.
12. A standard-library saved-file Memory inspector reads JSON and immutable/read-only SQLite without initializing agents, models, cache-hit counters or expiry deletion. Text is hidden by default. It compares sampled graph fact/claim, vector and cache states, reports malformed/missing data and refuses non-empty SQLite WAL to avoid a stale or mutating view. It does not repair Memory.

## Actual live observations

These are individual recorded requests, not a performance benchmark. They executed real retrieval, BGE/NLI inference and production graph routing with isolated diagnostic Memory.

- Normal Microsoft query: real Groq generation returned “Microsoft was founded by Bill Gates and Paul Allen.” Five downstream agents fired, Corrector and ReVerifier were not required, no agent failed. Two claims were reported verified and two facts were stored. Elapsed time was approximately 48.7 seconds.
- Deliberately wrong Microsoft draft: initial generation was intentionally skipped for “Snehith is the founder of Microsoft.” All seven downstream agents fired. Initial n8n returned zero passages; real supplementary Wikipedia retrieval supplied a decision-grade contradiction. Corrector produced “Gates and Allen established Microsoft on April 4, 1975.” Fresh re-verification reported the corrected claim verified with complete coverage; final Judge accepted and Memory stored one corrected fact. Elapsed time was approximately 116.0 seconds. Raw re-verification NLI entailment was approximately 0.9981. This establishes one repair, not general reliability.
- LinkedIn query: real generation produced six claims; the pipeline reported them verified. However, the year claim used a poorly ordered Rabois excerpt that is not independently satisfactory grounding. This is a remaining evidence-quality concern, not a claimed successful accuracy test.

For the Microsoft facts, graph fact/claim confidence, vector metadata and read-only cache rows were independently read back and agreed. This checks those fresh diagnostic writes, not distributed production consistency or an old quarantine record.

The normal Microsoft run also exposed disagreement between a raw NLI contradiction and an existing adjusted support score. Reports display both. A pipeline verdict must not be described as proof that the raw model was correct.

## Commands and local reports

See [the executable guide](../seven-agent-validation.md) for ordinary questions, supplied-draft diagnostics, all TXT filenames and Memory viewing. The terminal prints each actual absolute output directory. Shared documentation uses portable paths; local execution records may contain private user text and must not be published indiscriminately.

The local combined Microsoft package includes the eleven numbered/summary reports, reproduction commands, independent Memory read-back and a detailed implementation report. Its two raw recordings retain their original run origins. Reformatting a recorded run does not make new live calls.

## Validation evidence and its limits

Before the documentation/inspector addition, the current dirty checkout completed:

```text
python -m pytest -q --tb=short --disable-warnings --show-capture=no -ra
1369 passed, 14 skipped, 1 warning, 0 failed; exit 0; 519.59 seconds
```

The explicitly selected Detector/integration regression command completed 397 tests, exit 0. The main pytest configuration does not include the Detector package, so explicit Detector coverage matters.

That dirty-checkout total includes user-owned tests excluded from this publication. It is not the result for the exact proposed commit. Publication requires a fresh exact-index snapshot test; its outcome is reported separately in the publication response. Do not reuse this historical count as proof of the clean commit.

For the newly added viewer and case-label regression:

```text
python -m pytest -q agents/memory_agent/tests/test_readonly_inspector.py orchestration/tests/test_walkthrough_source_scores.py --tb=short --disable-warnings --show-capture=no
50 passed; exit 0; 1.95 seconds
```

The first run was 48 passed and 2 failed because SQLite fixture connections were left open on Windows. Explicit closing fixes the fixture resource leak; no failed assertion was removed. Viewer tests use isolated saved-file fixtures and subprocesses, not live Memory mutations.

The earlier full-suite skips were two opt-in n8n tests, two opt-in web tests, four unavailable legacy Detector-module tests and six missing Corrector curriculum checks. The warning was a Starlette/httpx TestClient deprecation. Passing regression tests do not establish model accuracy, calibration or production readiness.

## Complete intended file inventory

Modified implementation files:

- `agents/verifier_agent/adapters/general.py`: contiguous founding excerpts and real source fallback.
- `agents/verifier_agent/aggregation/duplicate_remover.py`: source/excerpt duplicate identity.
- `agents/verifier_agent/api/pipeline.py`: passage preservation, execution traces, claim binding and cache-origin distinction.
- `agents/verifier_agent/claims/claim_decomposer.py`: preserve year claims across separators.
- `agents/verifier_agent/rerankers/cross_encoder.py`: truthful execution/error telemetry and score validation.
- `agents/verifier_agent/retrievers/dense.py`: dense availability/load/inference/failure telemetry.
- `agents/verifier_agent/retrievers/hybrid.py`: actual backend contributions, fusion and fallback telemetry.
- `agents/verifier_agent/schemas/retrieval_trace.py`: backward-compatible diagnostic fields.
- `agents/verifier_agent/scorers/relation_verifier.py`: narrow admissibility guards and supplementary creation evidence.
- `halluciguard_detector/agent.py`: isolated locked lazy model initialization and retry.
- `orchestration/detector_bridge.py`: safe invalid-risk handoff.
- `orchestration/graph.py`: actual-question generation, fail-safe routing and complete corrected-claim coverage.
- `services/claim_analyzer.py`: atomic claim extraction and visible fallback behavior.
- `services/evidence_ranker.py`: preserve distinct factual candidate excerpts.
- `services/llm_detector_verifier_service.py`: output validation, safe handoff, sanitized errors and deferred certification.

Modified tests:

- `agents/verifier_agent/tests/test_final_stage_hardening.py`
- `agents/verifier_agent/tests/test_retrieval_quality_gate.py`
- `orchestration/tests/test_llm_detector_verifier_slice.py`

Added regression tests:

- `agents/verifier_agent/tests/test_claim_decomposer_years.py`
- `agents/verifier_agent/tests/test_creation_source_fallback.py`
- `agents/verifier_agent/tests/test_creator_list_grounding.py`
- `agents/verifier_agent/tests/test_general_adapter_founder_excerpt_selection.py`
- `agents/verifier_agent/tests/test_hybrid_passage_identity.py`
- `agents/verifier_agent/tests/test_passage_dedup_integrity.py`
- `agents/verifier_agent/tests/test_qualified_relation_admissibility.py`
- `agents/verifier_agent/tests/test_retrieval_execution_truth.py`
- `agents/verifier_agent/tests/test_verifier_trace_binding.py`
- `halluciguard_detector/tests/test_instance_isolation.py`
- `orchestration/tests/test_claim_analyzer_atomic_regression.py`
- `orchestration/tests/test_detector_failure_handoff.py`
- `orchestration/tests/test_direct_service_fail_safe.py`
- `orchestration/tests/test_evidence_ranker_preservation.py`
- `orchestration/tests/test_generation_answer_scope.py`
- `orchestration/tests/test_reverifier_claim_coverage.py`
- `orchestration/tests/test_walkthrough_source_scores.py`
- `agents/memory_agent/tests/test_readonly_inspector.py`

Added executable tools:

- `scripts/run_seven_agent_validation.py`: live production observer and separate per-agent TXT exports.
- `scripts/run_seven_agent_walkthrough.py`: narrated live/replay presentation without fabricated scores.
- `test_walkthrough.py`: root executable entrypoint.
- `scripts/inspect_memory.py`: read-only saved-file inspector.

Documentation modified: `README.md`, `docs/README.md`, `docs/agents/memory.md`, `agents/memory_agent/README.md`, `orchestration/README.md`, `scripts/README.md`.

Documentation added: `docs/seven-agent-validation.md` and this review.

No files are intentionally deleted. No model weights, prepared datasets, `.env`, runtime stores, backups or raw live reports belong in the commit. The untouched user changes in `test_relation_grounding_guard.py` and `scripts/test_verifier_stabilization.py`, the user's independent benchmark/audit and local TXT artifacts are excluded.

## Remaining defects and release blockers

There is no trustworthy total count of every project bug. Zero failing tests does not mean zero bugs. In this checkout, at least four directly inspected lifecycle/integration limitations remain:

1. `MemoryAgent.update_fact()` changes fact properties but not the linked claim or fact top-level confidence. It can also derive an unchanged confidence from an absent fact property rather than the existing top-level value. The quarantine repair in another worktree is not automatically integrated here.
2. `MemoryAgent.delete_fact()` can leave the linked claim behind.
3. Vector deletion rebuilds from in-memory embeddings, but loaded JSON deliberately omits embeddings; remaining entries can consequently disappear from the rebuilt search index. This path was inspected, not exercised against live production stores.
4. The direct LLM–Detector–Verifier service still limits its input to five claims; it is not full-answer coverage for long answers.

Additional constraints need evaluation, not invented fixes: incomplete/poorly ordered evidence can be admitted; raw NLI can err on correct relations; creator-list exhaustivity and partial-name assumptions remain; oversized excerpts and some tokenizer-level survival are not fully observed; weak contradiction performance and calibration are not improved by these changes; no validated Phase 1 remote-LLM uncertainty predictor exists; concurrent requests and distributed Memory were not production load-tested; missing curriculum and opt-in tests remain unavailable; individual request latencies need broader measurement. Source-disjoint provenance is required before strong calibration claims.

## Next work, in order

1. Review this branch against latest main and integrate compatible fixes without replacing newer architecture or losing the approved quarantine repair.
2. Reconcile Memory update/delete/vector-index lifecycle behavior, add isolated failure/retry tests, and independently verify any specifically authorized live record repair.
3. Audit evidence admission with adversarial temporal, entity, negation and structured-record fixtures; validate fresh live correction across more domains.
4. Run source-disjoint held-out real inference with unchanged criteria, calibration and false-accept/risk–coverage evaluation. Decide on research changes from measured failures, not test counts.
5. Validate concurrency, distributed consistency, timeouts, cache provenance and latency/memory budgets. Keep low-risk bypass disabled until separately justified.

This work is ready for review-branch preservation only after its exact-snapshot checks pass. It is not a production-readiness or high-accuracy certification.
