# Detector release validation, 2026-10-02

Base: `ca838528455aaf41d13deb4e63362f9768145e41`.
Review branch: `agents/detector-release-validation-20261002`.
This is an engineering/diagnostic review, not a production-readiness certificate.
No training, checkpoint replacement, temperature fitting or threshold tuning occurred.

## Confirmed changes

* Detector fallback now ranks whole normalized passages, rather than splitting
  them into isolated sentences. This preserves dates, negation and sibling facts
  within a selected passage. Selecting one of several passages can still omit
  relevant context; normalization and tokenizer bounds still apply.
* The shared Detector evidence selector serializes lazy initialization,
  decomposition, retrieval and trace capture with a reentrant thread lock.
  Shared model weights remain shared; mutable indexes cannot interleave between
  calls to this facade. This serializes work rather than increasing throughput.
* Verifier sparse/dense/fusion/reranker telemetry distinguishes real execution,
  empty results, failures, timeouts and fallbacks, with timings and selected
  contributors. Tests use controlled failure stubs; separate smoke diagnostics
  execute the cached real retrieval models.
* NLI observation attaches premise/hypothesis hashes and retained token offsets
  only when the independently obtained fast-tokenizer IDs exactly match the
  actual preprocessing IDs. Slow/opaque tokenizers, mismatches and observation
  failures are explicitly unobserved. Shared pipelines are not modified; each
  invocation uses a shallow local copy. Per-engine inference is serialized.
* Relationship search queries precede entity-only resolver queries. A finance
  fixture with the actual resolver demonstrates that the relation query, rather
  than `Microsoft Corp MSFT`, becomes the first topicality anchor. This does not
  decide truth and did not resolve the separate live Python retrieval failure.
* Direct-service certification runs after independent verification and genuine
  grounded Detector inference. Initial evidence-free triage is never certified.
  Missing/invalid/degraded results fail certification without blocking the
  otherwise valid Verifier attempt. Certification is execution proof, not truth.
* New Memory stores save and independently read back graph, vector metadata and
  cache before reporting `stored=True`. Duplicate reuse checks the existing
  fact's durable state, original text and confidence; a partial failed store
  cannot silently become a successful duplicate. Legacy ordinary verified facts
  may use their persisted top-level confidence when the redundant property is
  absent. Quarantine/update read-back remains strict.
* Diagnostic pipeline requests use fresh, isolated Memory and cache paths.
  No diagnostic request writes the previously quarantined production stores.

## Frozen-checkpoint paired experiment

Command (paths supplied explicitly to the locally available artifacts):

```text
python -m scripts.evaluate_paired_detector --data <prepared-RAGTruth> --checkpoint <existing-detector-best> --raw-responses <RAGTruth-response.jsonl> --output artifacts/paired-evaluation-20261002-context-preservation --batch-size 32 --exclude-cross-split-duplicates
```

18,770 genuine prepared test claims: 17,251 supported, 592 contradicted and
927 not-enough-info. Source groups are disjoint and official test-response
membership was checked. Seven supported test rows matching train/dev input
pairs were explicitly excluded. Five train/dev exact pairs remain: this
experiment does not recreate historically clean calibration or training.
Near-duplicate overlap was not measured. Prepared span labels were not
independently reannotated for selected snippets.

Weights SHA256: `470261f81870f5c4f97fe85918d02aa08673ac08744b5cae4d4621ee9d631bf9`.
Calibration SHA256: `7f1b411d4943141f8ab80fc4f8696917828dd41b85a9ea783c37039b9a7911a8`.
Temperature 0.7586129307746887; contradiction threshold 0.46; operational risk
threshold 0.625; maximum sequence length 256. All 14 protected input files were
hash-checked after evaluation and remained unchanged.

Actual results (same IDs, labels, weights and saved calibration in every arm):

* Historical prepared evidence: accuracy 83.8253%; contradiction precision
  24.4866%, recall 52.3649%, F1 33.3692%; inference 150.42 seconds.
* Production representation: accuracy 83.7666%; same contradiction precision,
  recall and F1; inference 142.17 seconds.
* Previous sentence-fragment fallback: accuracy 72.9622%; contradiction precision
  21.6710%, recall 56.0811%, F1 31.2618%; inference 100.53 seconds.
* Whole-passage lexical fallback: accuracy 83.7880%; contradiction precision
  24.4866%, recall 52.3649%, F1 33.3692%; inference 144.99 seconds.

Whole-passage fallback recovers 10.8258 accuracy percentage points relative to
the sentence-fragment arm on this cohort. It does NOT improve the existing
checkpoint's weak contradiction precision, demonstrate general live accuracy,
or establish safe low-risk acceptance. All fast-path release gates stay off.

At the existing grounded-risk threshold, the production arm's hypothetical
bypass would include 16,174 claims, of which 716 have non-supported gold labels
(226 contradicted): conditional error 4.4269%. This is a diagnostic using prepared
claim labels and correlated source claims, NOT a remote Phase 1 evaluation or
permission to enable bypass. No threshold was selected from these test results.

## Live evidence and unresolved release requirements

Live Python/Elon Musk diagnostic: real retrieval/reranker/NLI executed, but no
decision-grade evidence remained. Human review, zero repairs, zero stored facts;
46.08 seconds. Fast-tokenizer survival offsets were observed on actual NLI inputs.
Live France/Berlin diagnostic: human review, zero repairs, zero stored facts;
53.12 seconds. These are safety observations, not successful repair benchmarks.

A fresh generated Microsoft answer incorrectly asserted incorporation in 1975.
The pipeline accepted it using passages about founding/establishment; four
facts were stored in an isolated diagnostic store, not production Memory. This
is a confirmed false acceptance, not a successful correct-answer demonstration.
Microsoft's [investor FAQ](https://www.microsoft.com/en-us/investor/faq) and
[1981 history](https://learn.microsoft.com/en-us/shows/history/history-of-microsoft-1981)
give June 25, 1981 for incorporation. A narrowly scoped event-comparability veto
now prevents founding-only snippets from supporting or contradicting an
incorporation claim. It removes non-comparable evidence; it neither changes
neural probabilities nor manufactures a corrected date. Ambiguous attribution
and event-bearing passages still require actual NLI and further validation.
The Verifier cache key version is bumped so historical decisions from the older
event-scope policy cannot silently bypass this new relevance check.

The incorrect incorporation fact in that isolated diagnostic store was subsequently
quarantined by its exact fact, claim, vector and cache identities. Six new backup
files were copied and hash-checked before the scoped write. Independent persisted
graph, vector and SQLite read-backs confirmed unverified status and confidence
zero, including the graph's top-level confidence. Production Memory and the
earlier quarantine backups were not modified. The original experiment trace
remains unchanged to preserve the evidence of false acceptance.

The post-veto incorporation-date diagnostic was withheld, with zero stored facts
and no speculative correction; 26.26 seconds. A final live Hanoi answer was
accepted and stored one verified fact; 50.26 seconds. These individual observations
are not a reliability estimate.

A post-change controlled Bangkok/Vietnam repair used a fixed, explicitly cited
government excerpt and real inference/LLM correction. It produced Ha Noi,
passed ReVerifier, and stored one fact in isolated Memory; 22.37 seconds.
It is not a live-retrieval or cross-domain reliability demonstration.

The following remain blockers, not silently resolved promises:

1. Contradiction precision and independent held-out calibration remain weak or
   uncertain. Historical training provenance cannot be reconstructed by editing
   metadata; no replacement model has been validated.
2. Remote generation lacks the exact raw-logit features required by the current
   Phase 1 head. No compatible, trained remote uncertainty predictor exists.
   Grounded Detector risk is operational triage, never a truth certificate.
3. Reliable live corrective evidence is still not guaranteed. Insufficient
   evidence must remain withheld, rather than triggering speculative repair.
4. Token survival is observable only where actual preprocessing and matching
   offsets are exposed, not universally across opaque wrappers.
5. Thread/async regression tests are not distributed load tests. Memory stores
   still lack a distributed transaction, cross-process snapshot coordination and
   crash recovery. A read-back failure is explicit, not an atomic rollback.
6. Four real `contract_v4` Corrector curriculum artifacts are unavailable here.
   Six artifact-contract tests remain skipped; no replacement data was fabricated.
7. The earlier four-claim 123.44-second request spent 63.724 seconds in Verifier
   and 49.340 seconds in grounded Detector. Cold model loads, sequential retrieval
   and repeated evidence scoring need controlled performance experiments; these
   fixes do not establish a production latency SLO.
8. The existing Judge's moderate/relaxed policy can tolerate peripheral unknown
   claims when core claims answer the question. Rejecting non-comparable evidence
   does not make this policy an all-sentences-verified guarantee. Judge behavior
   was not changed in this task; mixed supported/unknown answers require a
   separate release-policy review.

Generated predictions, datasets, checkpoint files, isolated Memory stores and
raw live traces are ignored local artifacts, not publication contents.

## Final regression validation

Tests used cached/offline models (`HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`), CPU-only execution (`CUDA_VISIBLE_DEVICES=-1`)
and two OpenMP/MKL threads. The benchmark separately used the available GPU.

```text
python -m pytest -q halluciguard_detector/tests orchestration/tests/test_grounded_certification.py orchestration/tests/test_diagnostic_store_isolation.py orchestration/tests/test_final_integration_safety.py orchestration/tests/test_llm_detector_verifier_slice.py agents/verifier_agent/tests/test_retrieval_execution_status.py agents/verifier_agent/tests/test_execution_trace_repair.py agents/verifier_agent/tests/test_token_survival.py agents/verifier_agent/tests/test_relation_query_priority.py agents/verifier_agent/tests/test_incorporation_event_scope.py agents/verifier_agent/tests/test_port_backend_trace.py agents/verifier_agent/tests/test_verifier_hardening_regression.py agents/memory_agent/tests/test_update_consistency.py agents/memory_agent/tests/test_store_durability.py --tb=short --disable-warnings --show-capture=no -rs
```

436 passed, four skipped; 101.67 seconds; exit 0. These tests include controlled
backend failures, isolated persistence fixtures and synthetic tokenizer/model
contracts. They do not collectively certify live model accuracy. The default
pytest configuration does not include `halluciguard_detector/tests`, hence the
explicit focused invocation. Counts overlap the full suite and must not be added.

An initial final-suite run found one exact NLI diagnostics-schema assertion that
had not yet included the new `requested` and `total_duration_ms` fields: 1,292
passed, one failed, 14 skipped; exit 1. The schema assertion was extended without
removing its exact-key check; type checks and real-execution timing assertions
were added. The focused follow-up was:

```text
python -m pytest -q agents/verifier_agent/tests/test_nli_canonicalization.py agents/verifier_agent/tests/test_token_survival.py --tb=short --disable-warnings --show-capture=no
```

Six passed; 63.40 seconds; exit 0. The final full-suite rerun on that unchanged
Python snapshot was:

```text
python -m pytest -q --tb=short --disable-warnings --show-capture=no -rs
```

1,293 passed, 14 skipped, one warning; 457.63 seconds; exit 0. Skips were two
opt-in n8n tests, two opt-in web tests, four obsolete HaluEval import-dependent
tests and six unavailable Corrector curriculum checks. Separate live pipeline
diagnostics ran, but do not make those skipped tests passing tests. The warning
was the existing Starlette/httpx deprecation, not a suppressed failing test.

```text
python -m compileall -q halluciguard_detector agents/verifier_agent agents/memory_agent orchestration services scripts/evaluate_paired_detector.py scripts/smoke_retrieval_execution.py scripts/trace_pipeline_execution.py
git diff --check
git diff --cached --check
```

All exited 0. Final protected-artifact hash checks found zero mismatches across
14 inputs. Earlier production quarantine-backup hashes remained unchanged.
The publication contains only reviewed source, tests and this report: no model
weights, prepared data, raw traces, local credentials, stores or backups.
