# HalluciGuard full-system engineering audit — 2026-10-01

## Snapshot and boundaries

Worktree: `HalluciGuard.worktrees/full-system-audit-20261001`.
Branch: `agents/full-system-audit-20261001`.
Base: `9eac20724086b1c97e37b0dc650c1c86ea8f444e`.
Changes remain uncommitted. No push, deployment, main merge, checkpoint replacement,
production training, threshold fitting, or additional hosted LLM calls were performed.
The production Detector remains `halluciguard_detector`, RAGTruth/DeBERTa.
The legacy HaluEval implementation was not brought back.

Useful changes were selectively adapted from the independent Detector worktrees;
branches were not merged. Only contextual creator/title parsing fixes were adapted
from the original checkout's uncommitted Verifier work. Its probability overrides
were not copied. Original worktrees, artifacts and source checkpoint remain intact.

## Bug ledger

These IDs identify distinct investigated finding groups, not individual failing
assertions. Related cases sharing a boundary/root cause are recorded together.
This is not an exhaustive proof that the repository has no other defects.

HG-001 — High; correctness; confirmed, repaired.
`structured_evidence.normalize_evidence`, DetectorAgent evidence adapter.
Structured dictionaries/wrappers could discard sibling facts; punctuation-free long
prose could disappear at a truncation boundary. Bounded normalization preserves
field paths, source IDs, null/malformed counters and separate field/character/document
truncation. Duplicate equal text with different sources is explicitly ambiguous.
Tests: `test_port_safety.py` structured, long-text, provenance cases.

HG-002 — High; observability; confirmed, repaired.
`HybridRetriever.retrieve`, `DenseRetriever`, `CrossEncoderReranker`, retrieval trace.
Configured backends could be mistaken for executed hybrid retrieval. Diagnostics now
separate availability, attempts, execution, contribution and initialization/inference
failure. Sparse-only fallback is degraded, not hybrid success. Fusion/scoring
thresholds were not changed for telemetry. Tests: `test_port_backend_trace.py`.

HG-003 — High; correctness; confirmed, repaired.
`DetectorAgent._get_detector`.
Class-level lazy state could mix checkpoints. Per-instance ownership plus a lock
ensures compatible repeated reuse and retry after unsuccessful initialization.
Different instances do not share weights; identical instances may consume duplicate
memory. Tests: cache/isolation/concurrency/retry in `test_port_safety.py`.

HG-004 — Critical; safety; confirmed, repaired.
Detector bridge, graph routing, direct Detector/Verifier service.
Exceptions or incomplete results could block verification or appear low-risk.
Unavailable risk is None, not zero. Missing/nonfinite/degraded/uncalibrated/ungrounded
results require verification. Evidence-free certification remains pending and cannot
prevent the Verifier call. Defaults remain ALWAYS_VERIFY=true and fast path disabled.
Tests: bridge and slice contract tests, `test_port_safety.py`.

HG-005 — High; data; confirmed, repaired.
`halluciguard_detector.data` and training preflight.
Malformed labels, IDs, offsets, insufficient groups and train/dev overlap previously
could reach training. Validation rejects unsupported labels/invalid spans, tracks
malformed rows and uses deterministic source grouping. Existing correct interval
union/decorated-label logic is preserved. Tests: `test_port_data_integrity.py`.

HG-006 — High; reproducibility; confirmed, repaired.
Detector calibration/training/evaluation and Phase 1 output commands.
Output files could overwrite saved results; invalid logits/temperatures or mismatched
evaluation length could corrupt metrics. Existing outputs are refused; finite values
and checkpoint length are validated; single-class ROC/PR metrics are explicitly
undefined. Tests: data-integrity and Phase 1 lifecycle fixtures. Existing artifact
hashes were checked before/after real evaluation and were unchanged.

HG-007 — Critical; score provenance; confirmed, repaired.
`Detector._guard_signals`, Verifier decision selection/EvidenceScorer.
Entity/relation rules rewrote neural probabilities and injected 0.95 contradiction
or raised relevance floors. Guards now return warnings/verification requirements
without changing neural scores. Verifier NLI predictions are copied, not mutated;
neutral results cannot be promoted solely by relation keywords. Tests:
`test_audit_model_signal_integrity.py`, updated Detector semantics regressions.

HG-008 — High; correctness; confirmed, repaired.
`RelationVerifier.extract_triples/verify_relation`.
Titles prepended to prose corrupted subjects; plural creators were lost; negated,
reported, time-qualified or compound statements were treated as timeless atomic
relations. Titles are processed separately; explicit creator pairs retained;
unsupported comparisons abstain. Duplicate templates for the same semantic triple
are deduplicated. Relation diagnostics do not establish calibrated truth.
Tests: signal-integrity, contextual creation and grounding guard tests.

HG-009 — Critical; integration; confirmed, repaired.
Graph canonical Verifier conversion and `_verifier_node`.
Substring aliases could read "unsupported" as support; source IDs were truncated;
top-level failure status and incomplete requested-claim coverage could be lost.
Aliases are exact; IDs/status preserved; missing/duplicate/wrong-text claim reports
fail the initial verification contract. Tests: `test_full_audit_safety.py`.

HG-010 — Critical; correction safety; confirmed, repaired.
`graph._reverifier_node`.
Partial correction coverage could pass. Every expected correction claim must have
one matching verified report, actual supporting evidence and a completed run.
Incomplete coverage is INCOMPLETE_VERIFICATION, not acceptance. Tests: five
ReVerifier cases in `test_full_audit_safety.py`; updated step9 compound fixture.

HG-011 — High; memory; confirmed, repaired.
`graph._memory_node`, `MemoryAgent.store_fact`.
Unsupported/degraded results could become established facts or penalize source
trust despite no contradiction. Graph ingestion requires completed grounded
verification; provenance prefers source ID. Unverified/uncertain audit records do not
alter source trust. Direct MemoryAgent callers still own their input policy.
Tests: memory gating cases in `test_full_audit_safety.py`; existing memory suite.

HG-012 — Critical; Judge safety; confirmed, repaired.
`JudgeAgent.evaluate/_normalize_verifier_result`.
Decisive bare verdicts lacked proof; legacy adapters fabricated confidence and
inferred truth from numbers/keywords; "unverified" matched "verified".
Decisive reports require real correspondingly labelled snippets. Legacy adapters
preserve explicit values only, exact aliases and zero defaults for absent scores.
Missing graph Judge decisions escalate. Tests: Judge defects and audit safety tests.

HG-013 — High; configuration; confirmed, repaired.
CorrectorConfig, model_client, GroqGenerator.from_config.
Generator read a nonexistent config.provider (13 baseline Corrector failures).
Provider, model, timeout and bounded transport retries are now explicit validated
fields. Local remains default; Groq is explicit. Tests: provider-contract tests and
existing Corrector suite. No real hosted correction call was executed.

HG-014 — High; security; confirmed, repaired at reviewed boundaries.
Graph/shared error helper, deep health API, Corrector, ClaimAnalyzer, n8n client,
direct Detector services and runtime validation.
Raw provider/error text could contain prompts/credentials. External errors retain
classification but not exception bodies; dummy-secret regressions check this.
This is not a whole-repository formal proof of absence of sensitive logging.

HG-015 — Medium; integration; confirmed, repaired.
Verifier settings imports across adapters/cache/container/model manager/scorer/router.
Unqualified `config.settings` collided with legacy Judge config under reordered
imports. Production now imports the canonical Verifier settings namespace; tests
that configure those settings use that same module. No import-path redesign.

HG-016 — Medium; operations; confirmed, repaired.
`runtime_validation.validate_base_llm_configuration`.
Readiness required OpenRouter even for configured Groq/Gemini chains. Validation
now checks the generation provider chain without making requests or claiming
connectivity. The legacy OpenRouter-specific helper remains for compatibility.
Test: Groq-only readiness/dummy-secret case in audit safety tests.

HG-017 — High; architectural limitation; blocked/unverified.
Phase 1 same-generation uncertainty inputs and human labels are not available in
the hosted production path. An isolated reproducible five-feature linear head,
trace collector, preparation, calibration and evaluation pipeline are provided.
Default disabled; shadow requires genuine compatible traces; fast_path is explicitly
unavailable even if a fixture artifact says release_validated. No production head
was trained. Tiny fixture training establishes serialization only, not accuracy.

HG-018 — High; model-quality limitation; empirically observed, unresolved.
Existing DeBERTa frequently supports wrong creator claims. Real curated Snehith
example support=0.829354; contradiction recall on the prepared test at P(C)>=0.5
is 0.508446. Guards do not hide this with fabricated probabilities. Verification
remains mandatory. No measured before/after improvement or retraining was done.

HG-019 — Medium; configuration/provenance limitation; unresolved.
Original local checkpoint calibration differs from tracked current-main metadata.
Original temperature=0.75861293, risk threshold=0.62162162, length=256, no saved
contradiction threshold (existing runtime default 0.5 used). Current-main metadata
contains a different contradiction/risk threshold. Both were preserved. Measurements
identify the exact original artifact, not interchangeably any production configuration.

HG-020 — High; operational limitation; unverified.
No real complete hosted Base LLM→n8n→Judge→Corrector→ReVerifier→persistent Memory
transaction or deployment was performed. API cost/credentials, ngrok availability,
container resource requirements and production concurrency need separate validation.
Actual local DeBERTa, cached dense retrieval and BGE reranking did execute.

HG-021 — Medium; temporal evidence alignment; fixed for the demonstrated narrow case.
Both baseline Paris tests returned CONFLICTED. A real diagnostic retrieval found a
Wikipedia historical passage about the seat of government at Versailles from
1682–1789: NLI contradiction 0.994932, versus entailment 0.979044 for the current
Paris article. The scorer combined support 0.6607 and contradiction 0.6748.
EvidenceScorer._historical_only_against_present_claim now abstains on contradiction
from explicitly bounded past-tense premises against undated present-tense claims.
It strips adapter headings, requires a from/to or between/and four-digit interval
and past copula, and refuses abstention when the premise asserts present tense or
the claim contains a year. Model probabilities and thresholds are unchanged.
Six deterministic regressions cover the historical fixture and retained present,
dated and past refutations. Both real Paris tests passed after this change.
This is not a complete temporal reasoner or proof of improved benchmark accuracy.

HG-022 — Medium; confidence accounting; fixed.
ConflictResolver.resolve used entailment_score even for contradiction-labelled
citations. It now weights the corresponding nli_contradiction/contradiction_score,
with finite in-range score and credibility checks. Missing contradiction scores
are not replaced by invented 0.5 confidence. Regression tests cover actual formatted
EvidenceItem objects, dictionaries and six invalid probability values.

HG-023 — Medium; model-loader configuration; confirmed remaining defect.
ModelManager.load_nli_model's primary hf_pipeline load and load_reranker_model's
CrossEncoder constructor do not consistently pass the configured offline-only
policy. The embedding loader does. This was found during continuation inspection;
these loader paths were not changed in the temporal/confidence patch. Cached model
execution is established, not safe missing-cache offline behavior. Strict local
model/tokenizer loading and CPU retry policy require their own regression coverage.

Ledger totals: 23 investigated finding groups, comprising 19 confirmed engineering
defect groups (18 repaired in their tested scope, HG-023 outstanding) and four
architectural/experimental limitations (HG-017 through HG-020). These are not an
exhaustive count of every possible bug; broader privacy, parsing and deployment
claims remain bounded by the coverage described here.

## Genuine existing-checkpoint measurements

Command: `python -m scripts.audit_detector_measurements --checkpoint ABS_CHECKPOINT
--raw-data ABS_RAGTRUTH_DATASET --prepared-data ABS_PROCESSED --retrieval-smoke`.
The executed absolute paths are in the final response. No output artifacts written.
Checkpoint weights SHA256: 470261f81870f5c4f97fe85918d02aa08673ac08744b5cae4d4621ee9d631bf9.
CUDA execution; max_length=256; no new temperature/threshold selection.

Raw inventory: 2,965 source records, 17,790 response records (15,090 train, 2,700
test). JSON parsing/IDs valid in this inventory; it is not an annotation validation.
Local RAGTruth distribution includes an MIT license; retain dataset attribution and
independently assess source-material rights for intended deployment.

Prepared train/dev/test: 29,832/10,873/18,777 claims; source groups 2,263/252/450;
all group intersections zero, no missing groups. This checks current files, not a
complete historical provenance proof of how the checkpoint was trained.
Test labels: 17,258 SUPPORTED, 592 CONTRADICTED, 927 NOT_ENOUGH_INFO.
Fresh inference exactly reproduced the archived prediction metrics.
Test SHA256: 32808798f449275daa9833ca4ad967653ad0d3dfc2352d0f38d7ec6ccc802503.

Three-class accuracy=0.838313, macro-F1=0.507246. Majority share=0.919103, so
accuracy alone would give a misleading impression of useful contradiction detection.
Confusion matrix (true rows/predicted columns S,C,N):
15046 954 1258
211 325 56
427 130 370

P(CONTRADICTED)>=0.5: precision=0.251043, recall=0.508446, F1=0.336125,
TN=17287 FP=898 FN=291 TP=301; PR-AUC=0.231139, ROC-AUC=0.850047,
Brier=0.040943, ECE=0.051337.
Verification-risk>=0.62162162: precision=0.308337, recall=0.533246, F1=0.390738,
TN=15441 FP=1817 FN=709 TP=810; PR-AUC=0.288640, ROC-AUC=0.821198,
Brier=0.108324, ECE=0.129556.
These are existing-checkpoint measurements, not improvements caused by the audit.

Real two-passage retrieval smoke: sparse and dense contributed, route=hybrid,
degraded=false; BGE reranker executed on CUDA and scored two passages in 3024 ms.
That latency is one local smoke measurement, not a throughput/production SLA.
Loading reported an unexpected position_ids buffer and deprecated device property;
neither was suppressed as proof of checkpoint equivalence.

## Remaining engineering boundaries

Normalization is bounded (64 documents/fields, depth 8, 8,000 characters); later
fields may be excluded and token limits still apply. Only one selected evidence
snippet per Detector claim is classified. Fast tokenizer sequence IDs permit
measured token consumption; other tokenizers report unavailable rather than guessing.
Multiple conflicting sources are adjudicated by the Verifier, not one Detector pair.
Direct slice processes at most five claims and now reports truncation/degradation;
it is not equivalent to the complete graph or a grounded Detector certification.
Regex claim/relation fallbacks cannot guarantee perfect semantic decomposition.
Memory expiration/conflict policy and live external services require further review.
No frontend changes, model/data deletion, deployment changes or main integration.

## Final validation

Previous application/test snapshot (superseded by the continuation below):

`python -m pytest -q --tb=short --disable-warnings --show-capture=no -rs`
Exit 1: 1,099 passed, 14 skipped, 2 failed, 1 warning; 491.63 seconds.
Failures: `TestVerifierStabilization.test_single_claim_end_to_end_verification`
and `TestVerifierV1Stabilization.test_1_paris_capital_of_france` (CONFLICTED vs VERIFIED).
The identical two-test command on untouched baseline a9e9ed9 exited 1 with two
identical failures in 84.77 seconds. That baseline and this base have identical
backend/test files; their differences are frontend-only. Historical full baseline
was 1,026 passed, 14 skipped, 15 failed; it was not rerun in full during this audit.

Final focused command:
`python -m pytest -q halluciguard_detector/tests orchestration/tests/test_full_audit_safety.py
agents/verifier_agent/tests/test_audit_model_signal_integrity.py
agents/verifier_agent/tests/test_port_backend_trace.py
agents/corrector_agent/tests/test_audit_provider_contract.py
orchestration/tests/test_llm_detector_verifier_slice.py --tb=short --disable-warnings --show-capture=no`
Exit 0: 285 passed in 68.79 seconds. The subsequent explicit-route compatibility
fix was separately checked with the following graph test command and final full suite.

`python -m pytest -q orchestration/tests/test_graph_contract.py
orchestration/tests/test_step9_canonical_orchestration.py
orchestration/tests/test_full_audit_safety.py --tb=short --disable-warnings --show-capture=no`
Exit 0: 60 passed in 47.94 seconds.
`python -m pytest -q halluciguard_detector/tests --tb=short --disable-warnings --show-capture=no`
Exit 0: 222 passed in 73.34 seconds (before the later graph-only route fix).
`python -m pytest -q halluciguard_detector/tests/test_port_data_integrity.py
--tb=short --show-capture=no -rs`: exit 0, 15 passed, 43.57 seconds.

`python -m compileall -q halluciguard_detector agents services orchestration scripts/audit_detector_measurements.py`
and `git diff --check`: exit 0 on final code.
No configured Python lint tool was found. Frontend dependencies/build were not run.
Docker CLI exists; image build/deployment were not performed.

`python -m pytest --collect-only -q --tb=short`: exit 0, 1,115 collected in
34.96 seconds. It identified the full-suite warning as StarletteDeprecationWarning:
the installed FastAPI TestClient uses deprecated httpx integration. Dependencies
were not changed merely to silence this warning.

Skipped: two opt-in live n8n, two opt-in live web, four legacy removed HaluEval
Detector import tests, six unavailable Corrector contract_v4 curriculum checks.
The actual production Detector tests run separately; legacy skips are not passes.
Two superseded full-suite attempts were explicitly terminated after code corrections;
neither is counted as successful/final validation. Development failures and their
causes are reported separately in the final response.

Local environment: Python 3.13.2, torch 2.13.0+cu126, Transformers 5.2.0,
NumPy 2.2.5, NVIDIA GeForce RTX 3050 6GB Laptop GPU. Docker requirements target
Python 3.11, Transformers <5 and NumPy <2; that pinned environment remains untested.

Current-main weights are real 283,348,948-byte safetensors, not an LFS pointer;
their SHA256 equals the independently evaluated original artifact. Class mapping
is exactly 0=SUPPORTED, 1=CONTRADICTED, 2=NOT_ENOUGH_INFO. Current-main calibration
was preserved. Recomputing the verified archived logits using its existing thresholds
(no inference/training/refitting in this comparison) gives:
P(C)>=0.46: precision=0.244866, recall=0.523649, F1=0.333692,
TN=17229 FP=956 FN=282 TP=310.
Risk>=0.625: precision=0.309588, recall=0.529296, F1=0.390671,
TN=15465 FP=1793 FN=715 TP=804.
Threshold-independent AUC/calibration/three-class results are unchanged.
This is a configuration comparison of the same saved predictions, not a measured
model improvement. Neither existing calibration file was modified.


## Exact changed-file inventory

All paths are relative to the audit worktree identified above. 61 tracked files
are modified and 16 files are newly added; none are deleted or migrated.

agents/corrector_agent/corrector/__init__.py — Modified: Sanitized invalid-input/pipeline results.
agents/corrector_agent/corrector/config.py — Modified: Explicit local/Groq provider and validated transport settings.
agents/corrector_agent/corrector/groq_client.py — Modified: Config-driven generator transport and sanitized response errors.
agents/corrector_agent/corrector/model_client.py — Modified: Valid provider loading and sanitized initialization/generation failures.
agents/corrector_agent/corrector/validation.py — Modified: Validation diagnostics omit raw model response.
agents/corrector_agent/tests/test_prompt_model.py — Modified: regressions/fixtures aligned with real evidence, preserved model scores, canonical settings or sanitized errors; no skips introduced.
agents/judge_agent/judge_agent.py — Modified: Evidence-required decisions, safe legacy normalization and missing Detector score handling.
agents/judge_agent/tests/test_judge_defects.py — Modified: regressions/fixtures aligned with real evidence, preserved model scores, canonical settings or sanitized errors; no skips introduced.
agents/memory_agent/memory/memory_agent.py — Modified: Unverified records do not penalize source trust.
agents/verifier_agent/adapters/ai_research.py — Modified: canonical Verifier settings import; no provider/scoring redesign.
agents/verifier_agent/adapters/cybersecurity.py — Modified: canonical Verifier settings import; no provider/scoring redesign.
agents/verifier_agent/adapters/finance.py — Modified: canonical Verifier settings import; no provider/scoring redesign.
agents/verifier_agent/adapters/healthcare.py — Modified: canonical Verifier settings import; no provider/scoring redesign.
agents/verifier_agent/adapters/legal_general.py — Modified: canonical Verifier settings import; no provider/scoring redesign.
agents/verifier_agent/adapters/web_enhanced.py — Modified: canonical Verifier settings import; no provider/scoring redesign.
agents/verifier_agent/api/pipeline.py — Modified: Backend trace propagation; preserve real NLI scores and remove rule-injected probabilities.
agents/verifier_agent/cache/sqlite_cache.py — Modified: canonical Verifier settings import; no provider/scoring redesign.
agents/verifier_agent/container.py — Modified: canonical Verifier settings import; no provider/scoring redesign.
agents/verifier_agent/formatters/citation_formatter.py — Modified: Keep original pre-tokenizer snippet separate from display preview/provenance.
agents/verifier_agent/models/model_manager.py — Modified: canonical Verifier settings import; no provider/scoring redesign.
agents/verifier_agent/nli/robust_entailment.py — Modified: Exact labels and complete finite NLI probability validation; sanitized errors.
agents/verifier_agent/rerankers/cross_encoder.py — Modified: Reranker availability/attempt/execution/failure metadata.
agents/verifier_agent/retrievers/dense.py — Modified: Initialization/indexing/inference execution diagnostics.
agents/verifier_agent/retrievers/hybrid.py — Modified: Actual contributing route, failures and degraded fallback diagnostics.
agents/verifier_agent/routers/domain_validator.py — Modified: canonical Verifier settings import; no provider/scoring redesign.
agents/verifier_agent/schemas/models.py — Modified: Additive evidence provenance/input-snippet fields.
agents/verifier_agent/schemas/retrieval_trace.py — Modified: Additive actual backend diagnostics.
agents/verifier_agent/scorers/evidence_scorer.py — Modified: Remove synthetic relation/refutation score promotion; abstain on rule-model disagreement.
agents/verifier_agent/scorers/relation_verifier.py — Modified: Title/creator/qualifier parsing, semantic duplicate handling and conservative diagnostic abstention.
agents/verifier_agent/tests/test_contextual_creation_relation.py — Modified: regressions/fixtures aligned with real evidence, preserved model scores, canonical settings or sanitized errors; no skips introduced.
agents/verifier_agent/tests/test_official_integrations.py — Modified: regressions/fixtures aligned with real evidence, preserved model scores, canonical settings or sanitized errors; no skips introduced.
agents/verifier_agent/tests/test_relation_grounding_guard.py — Modified: regressions/fixtures aligned with real evidence, preserved model scores, canonical settings or sanitized errors; no skips introduced.
agents/verifier_agent/tests/test_v2_regression_failures.py — Modified: regressions/fixtures aligned with real evidence, preserved model scores, canonical settings or sanitized errors; no skips introduced.
agents/verifier_agent/tests/test_verifier_hardening_regression.py — Modified: regressions/fixtures aligned with real evidence, preserved model scores, canonical settings or sanitized errors; no skips introduced.
halluciguard_detector/agent.py — Modified: Per-instance loading/locking, evidence/consumption diagnostics and optional Phase 1 trace.
halluciguard_detector/calibration.py — Modified: Finite calibration/logit validation and protected calibration output.
halluciguard_detector/data.py — Modified: RAGTruth parsing/offset/identifier/split validation and safe preparation outputs.
halluciguard_detector/detector.py — Modified: Exact label mapping, finite complete model output, unchanged probabilities and separate guard warnings.
halluciguard_detector/evidence.py — Modified: Truthful lexical-only selection route and actual selected/model input.
halluciguard_detector/nli_input.py — Modified: Optional actual encoded token ownership/truncation diagnostics; input format unchanged.
halluciguard_detector/schemas.py — Modified: Additive model-input and evidence-selection trace fields.
halluciguard_detector/tests/test_agent.py — Modified: regressions/fixtures aligned with real evidence, preserved model scores, canonical settings or sanitized errors; no skips introduced.
halluciguard_detector/tests/test_claim_evidence.py — Modified: regressions/fixtures aligned with real evidence, preserved model scores, canonical settings or sanitized errors; no skips introduced.
halluciguard_detector/tests/test_data_conversion.py — Modified: regressions/fixtures aligned with real evidence, preserved model scores, canonical settings or sanitized errors; no skips introduced.
halluciguard_detector/tests/test_detector_semantics.py — Modified: regressions/fixtures aligned with real evidence, preserved model scores, canonical settings or sanitized errors; no skips introduced.
halluciguard_detector/training.py — Modified: Preflight, artifact protection, length consistency and honest gold-label/evaluation metadata.
orchestration/api.py — Modified: Sanitized deep health exception output.
orchestration/detector_bridge.py — Modified: Validate scores/provenance and preserve unavailable risk and Phase 1 metadata.
orchestration/graph.py — Modified: Fail-safe routing, full claim coverage, source/status preservation and correction/Memory gates.
orchestration/runtime_validation.py — Modified: Actual provider-chain/Corrector readiness and canonical Verifier settings.
orchestration/schemas.py — Modified: Additive canonical source ID.
orchestration/state.py — Modified: Sanitized shared state error messages.
orchestration/tests/test_detector_bridge.py — Modified: regressions/fixtures aligned with real evidence, preserved model scores, canonical settings or sanitized errors; no skips introduced.
orchestration/tests/test_llm_detector_slice.py — Modified: regressions/fixtures aligned with real evidence, preserved model scores, canonical settings or sanitized errors; no skips introduced.
orchestration/tests/test_llm_detector_verifier_slice.py — Modified: regressions/fixtures aligned with real evidence, preserved model scores, canonical settings or sanitized errors; no skips introduced.
orchestration/tests/test_step9_canonical_orchestration.py — Modified: regressions/fixtures aligned with real evidence, preserved model scores, canonical settings or sanitized errors; no skips introduced.
services/character_regenerator.py — Modified: Sanitized generation errors.
services/claim_analyzer.py — Modified: Sanitized analysis errors; decomposition strategy unchanged.
services/llm_detector_service.py — Modified: Detector errors/incomplete output cannot bypass verification.
services/llm_detector_verifier_service.py — Modified: Initial certification pending; safe Verifier fallback and partial-claim diagnostics.
services/n8n_retrieval_client.py — Modified: Sanitized retrieval/health/batch errors.
agents/corrector_agent/tests/test_audit_provider_contract.py — Added: Corrector config, bounded transport settings and sanitized loading errors.
agents/verifier_agent/tests/test_audit_model_signal_integrity.py — Added: no synthetic model-score promotion and adversarial relation/NLI inputs.
agents/verifier_agent/tests/test_port_backend_trace.py — Added: mocked dense/reranker backend execution and fallback telemetry.
docs/reports/full-system-audit-20261001.md — Added: ledger, measurements, boundaries and validation record.
halluciguard_detector/local_generation.py — Added: isolated cached local generation with actual token/logit traces.
halluciguard_detector/phase1.py — Added: compatible five-feature shadow scoring; missing inputs unavailable and fast path disabled.
halluciguard_detector/phase1_cli.py — Added: reproducible collect/annotate/prepare/preflight/train/evaluate CLI.
halluciguard_detector/phase1_collect.py — Added: isolated cached local generation with actual token/logit traces.
halluciguard_detector/phase1_data.py — Added: human-label trace ingestion and four independent grouped splits.
halluciguard_detector/phase1_train.py — Added: research head training, separate calibration/evaluation, artifact hashes and overwrite guards.
halluciguard_detector/structured_evidence.py — Added: bounded structured/prose normalization and provenance/truncation metadata.
halluciguard_detector/tests/test_phase1.py — Added: same-generation compatibility, disabled/blocked bypass and fixture-only training lifecycle.
halluciguard_detector/tests/test_port_data_integrity.py — Added: malformed data, split leakage, output preservation, lengths, logits and gold-label guards.
halluciguard_detector/tests/test_port_safety.py — Added: structured evidence/provenance, instance isolation, retry and fail-safe routing.
orchestration/tests/test_full_audit_safety.py — Added: coverage, canonical labels/status/IDs, Judge/Memory gates, error privacy and readiness.
scripts/audit_detector_measurements.py — Modified: Added: read-only real checkpoint/data/retrieval measurement command.

Continuation file addition to the modified-file manifest:
agents/verifier_agent/scorers/conflict_resolver.py — Modified: use the reported
evidence direction's actual probability rather than entailment for contradictions;
ignore invalid weights. There are now 62 modified tracked and 16 added files,
78 total; no deletions or migrations.

Continuation validation:
The temporal fix and existing model-signal regressions plus the two real Paris
integration tests: exit 0, 24 passed in 83.53 seconds. Seven later confidence
accounting regressions were added after that run and require the subsequent
focused/full validation recorded below.
Final focused continuation command is the same six-area focused command above:
exit 0, 298 passed in 52.66 seconds. Compileall and diff-check also exited 0.
The standalone public Paris diagnostic disabled n8n and cache for that process,
used Wikipedia retrieval and actual cached dense, reranker and NLI models, and
exited 0. It did not run the hosted Base LLM, change configuration files, fit
thresholds, modify weights, or write evaluation artifacts.
Known boundary found during inspection: the Verifier NLI/reranker model manager
does not consistently enforce allow_model_downloads=false on its primary load
paths (the embedding loader does). Cached models were available in these runs;
strict offline/no-download behavior for missing caches remains unverified and
requires a separate loader-policy regression. No absence of network requests is
claimed for those primary load paths.

Final unchanged application/test code after both continuation fixes:
`python -m pytest -q --tb=short --disable-warnings --show-capture=no -rs`
Exit 0: 1,114 passed, 14 skipped, 0 failed, 1 warning in 452.62 seconds.
The earlier 1,099-pass/two-failure run is superseded, not the final result.
Both baseline Paris failures are resolved on this snapshot; no assertions were
weakened, tests skipped, model probabilities replaced, or thresholds adjusted.
The 14 skips remain the two opt-in n8n, two opt-in web, four legacy HaluEval import
and six missing Corrector curriculum cases described above. One existing
Starlette/TestClient dependency deprecation warning remains.
Focused continuation: 298 passed, exit 0; compileall and git diff --check: exit 0.
Documentation updates after the full run do not change application/test code.
This full suite result does not establish live complete production execution,
strict missing-cache offline behavior, deployment readiness, or improved held-out
Detector accuracy. Changes remain uncommitted on agents/full-system-audit-20261001
at HEAD 9eac20724086b1c97e37b0dc650c1c86ea8f444e.

