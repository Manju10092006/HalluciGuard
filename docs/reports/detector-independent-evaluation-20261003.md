# Independent Detector evaluation — 2026-10-03

**Release conclusion: INDEPENDENT_EVALUATION_BLOCKED.** The implementation and synthetic framework tests are complete; a real independent cohort is not available. Production readiness is **not established**, `production_ready=false`, and fast-path acceptance stays **disabled**. No model training, real-data calibration, threshold tuning, checkpoint replacement or Prompt3 work occurred.

## Repository and frozen environment

Working branch: `agents/production-readiness-20261003`. Initial remote HEAD: `4ee6fd23547a5f87f419ca1d00f53554d418536d`. Exact implementation commit tested: **`39b7522bb68a139b130bb1deaf10b944a5dcd1cb`**. `origin/main` and merge-base are both `82cdc6fb734f5fa31cd345a4060f60178bea2118`. The branch was fetched, confirmed through the GitHub plugin, checked out and baseline ancestry verified. The report/status publication is a documentation/metadata child of the tested implementation; no source history or main was reset or rewritten. No PR is opened.

Prompt1 report/manifest conclusions are preserved. The existing isolated venv remains valid; no new dependency installation was required. Python **3.12.14**, Ubuntu **24.04.3** on x86_64 Linux, CPU-only (`CUDA_VISIBLE_DEVICES=-1`; torch CUDA unavailable), two OpenMP/MKL threads. Imports of Pydantic, torch, Transformers and Detector dependencies succeeded; `pip check` exited 0. Root requirements and the recorded Prompt1 resolved freeze remain the declared environment basis. The repo expects Python 3.11.9, so this is not an exact recreation of the laptop interpreter.

Core versions: Pydantic 2.13.5; torch 2.14.1+cu130; Transformers 5.18.0; sentence-transformers 6.1.0; NumPy 2.5.3; scikit-learn 1.9.1; pytest 9.1.1; pytest-asyncio 1.4.0. Prompt1 installed root requirements plus pytest/pytest-asyncio, then socksio 1.0.0 for the supplied proxy and Trio 0.34.0 for all AnyIO backend variants. Complete freeze/install command and dependency hashes remain in the Prompt1 manifest and are referenced from this manifest.

Checkpoint bytes came from Prompt1 Git LFS fetch; the cached public Memory model remains at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. No models were downloaded in this task. Final tests use HF offline flags and cached Memory embeddings. Verifier BGE/DeBERTa weights remain absent; the owner laptop n8n workflow remains unavailable.

## Existing baseline reproduction — partial, not independent

Checkpoint SHA256 matches `470261f81870f5c4f97fe85918d02aa08673ac08744b5cae4d4621ee9d631bf9`. Linux/Git calibration SHA256 is `3ce4724edd70d41cbf5975cf65643ddf1a8f3479bb983e324fb19789a27a3f32`; LF-to-CRLF conversion exactly yields the recorded `7f1b411d4943141f8ab80fc4f8696917828dd41b85a9ea783c37039b9a7911a8`. No parameter or byte changes were made.

The approved checkout contains frozen checkpoint/tokenizer/config/calibration files, archived `dev_predictions.npz` and `test_predictions.npz`, metadata and reports. The archives contain **only logits and integer labels**, with no input text, row/source/response IDs or split-generation membership. Numerical recomputation of archived predictions is legitimate; it is not fresh checkpoint inference, reproduction of the paired arms, or evidence of dataset independence.

The 18,777-row archive reproduces 83.8313% three-class accuracy and contradiction-threshold precision/recall/F1 of 24.4866% / 52.3649% / 33.3692%. The last paired experiment used **18,770** claims and reported historical prepared accuracy 83.8253% and production representation accuracy 83.7666%. Those paired values remain historical: prepared rows/raw responses/paired outputs are missing, so fresh numerical reproduction is **BLOCKED**. No synthetic examples replaced the missing real benchmark.

Missing approved artifacts (not hypothetical substitutes):

- `halluciguard_detector/data/processed/train.jsonl`
- `halluciguard_detector/data/processed/dev.jsonl`
- `halluciguard_detector/data/processed/test.jsonl`
- `halluciguard_detector/third_party/RAGTruth/dataset/source_info.jsonl`
- `halluciguard_detector/third_party/RAGTruth/dataset/response.jsonl`
- `artifacts/paired-evaluation-20261002-context-preservation`
- `halluciguard_detector/data/train.jsonl`
- `halluciguard_detector/data/dev.jsonl`
- `halluciguard_detector/data/test.jsonl`

## Provenance and leakage audit

The existing RAGTruth converter checks source/response identities, human span vocabulary and quality, groups train/dev by source and downsamples **training** supported rows. Its labels describe sentence/span-derived examples, not independently reannotated runtime claims. Official test membership is retained, but converter grouping alone does not prove exact/near isolation. The paired runner checks source groups/exact pairs and can record exact exclusions; it previously did not measure near duplicates. The new framework checks those missing dimensions.

MODEL_CARD describes 29,832 training examples, seed 42 and batch size 16. Actual model metadata has null training seed/batch size; the legacy training report contains one epoch and best threshold 0.6216216216216217, whereas the frozen calibration records risk 0.625 and contradiction ~0.46. These references cannot reconstruct historical exposure. No metadata claim was silently upgraded to verified provenance, and no threshold was changed. Phase1 trace/data/training code was inspected separately; its same-event features and unavailable head do not replace the grounded checkpoint cohort.

Exact real-data duplicate/source/response overlap counts: **UNMEASURED/BLOCKED**. Real near-duplicate counts: **UNMEASURED/BLOCKED**, not zero. The preserved release report describes five exact train/dev pairs and seven excluded supported test matches; those are historical findings, not measurements in this run. Prior near-duplicate findings were absent. Archived labels cannot support text or identity audits. Corrector data, legacy HaluEval, 13 handwritten probes and a 600-row truncated historical subsample were not substituted for independent data.

Implemented audit: NFKC → casefold → collapse Unicode whitespace, preserving punctuation, numbers and negation; normalization/Unicode versions are locked. Exact checks cover claims, evidence, pairs, row IDs, source/document IDs, response IDs and source+response IDs. Near pairs use complete shared-claim character-5-gram candidates, with **both** claim/evidence Jaccard ≥ **0.85**, default 10,000,000 candidate budget. Exhaustion blocks rather than returns a partial audit. This lexical method does not detect all semantic/translated paraphrases; absence of lexical matches alone would not prove semantic independence. Pair identities/similarities, counts and exclusions are explicit. No rows are silently deleted.

## Three data roles and locked manifests

| Role | Real rows in newly established split | Source/response groups | Status |
|---|---:|---|---|
| Training | Unavailable | Unavailable | Historical fitting membership missing |
| Development/calibration | Unavailable | Unavailable | Historical temperature/threshold membership missing |
| Final locked test | Unavailable | Unavailable | **Not created; no legitimate lock hash** |

Available archival label counts (not new split counts):

| Archive | Rows | Supported | Contradicted | NEI | IDs/text/source groups |
|---|---:|---:|---:|---:|---|
| Dev predictions | 10,873 | 9,678 | 530 | 665 | Absent |
| Test predictions | 18,777 | 17,258 | 592 | 927 | Absent |

The machine-readable manifest explicitly marks each real role blocked, with null row/source IDs, counts and hashes. No zero-row “locked test” was manufactured. A **locked-test manifest SHA256 is unavailable**, and independently established split counts cannot be reported. The metadata artifact’s own checksum is linked from readiness status; it is not a hash of a real test set.

The reusable framework either freezes supplied official roles or groups source/document/response/exact/near connected components, assigns whole components deterministically, preserves every label/row and refuses fewer than three independent components. Per-role manifests have stable sorted identities, class/exclusion counts, timestamp/version/git/seed, source hashes and immutable data/manifest hashes. Verification requires an externally pinned bundle hash. No final-split balancing/oversampling occurs. Recreating a clean split cannot erase checkpoint exposure: evaluation additionally requires complete historical membership linked to **both checkpoint and calibration hashes**, rechecks those files, and rejects historical train/dev or locked-test overlaps. Completeness of that declaration requires genuine records/owner attestation; hashes alone cannot prove it.

## Evaluation runner and calibration safeguards

The runner writes a hashed evaluation plan before inference, uses local frozen weights and the canonical NLI contract, and never fits/calibrates/selects a model. It records raw logits/classes, calibrated probabilities and risk decisions per stable row, separately from aggregate model quality and hypothetical routing. Locks/checkpoint/exposure are rechecked after inference. Partial failed runs never write a successful `evaluation.json`.

Model-quality metrics: accuracy, confusion matrix, per-class supported/contradiction/NEI P/R/F1/support, macro/micro F1, top-label 10-bin ECE and multiclass Brier. Policy metrics: saved-threshold contradiction P/R/F1 and FP/FN counts, risk-based hypothetical eligibility/coverage, contradicted and non-supported bypass counts, conditional and population false-accept rates, supported routing/false-reject and abstention/routing fractions. Bypass diagnostics do not represent the full production Judge/routing policy and never enable acceptance.

Implemented CIs use deterministic source-cluster percentile bootstrap: seed **20261003**, default **2,000 replicates**, **95%** intervals; preserve all claims of each sampled source cluster. At least two clusters are required. Undefined denominators stay null with replicate counts. CI limitations include few clusters, dependence across sources, fixed-model uncertainty and historical provenance. **Real locked-test confidence intervals are BLOCKED**, since the cohort and source IDs are missing. Synthetic CI tests are software tests, not performance evidence.

Standard `fit_temperature`/`best_threshold` now require development provenance and refuse locked role-bound labels even under a contrary explicit declaration. Unmarked arrays fail closed unless deliberately declared development. Standard training verifies managed manifests and role markers before model loading; renamed/copied locked rows in dev are rejected. The existing threshold-selection tests were updated to explicitly declare their synthetic development role; assertions were retained. This protects ordinary tooling against accidental use, not deliberate stripping/relabeling or a false provenance declaration by a researcher. Actual research must restrict final-label access too. No production inference thresholds or Judge rules changed.

## Detector metric table — independent cohort BLOCKED

The following numbers are **archived-logit recomputation only** on 18,777 old examples; every new locked-cohort metric is blocked. They are not a second independent evaluation.

| Metric | Locked independent cohort | Archived recomputation |
|---|---|---:|
| Three-class accuracy | BLOCKED | 83.831283% |
| Macro F1 | BLOCKED | 0.507246 |
| Micro F1 | BLOCKED | 0.838313 |
| Argmax SUPPORTED precision | BLOCKED | 0.959322 |
| Argmax SUPPORTED recall | BLOCKED | 0.871828 |
| Argmax SUPPORTED f1 | BLOCKED | 0.913484 |
| Argmax CONTRADICTED precision | BLOCKED | 0.230660 |
| Argmax CONTRADICTED recall | BLOCKED | 0.548986 |
| Argmax CONTRADICTED f1 | BLOCKED | 0.324838 |
| Argmax NOT_ENOUGH_INFO precision | BLOCKED | 0.219715 |
| Argmax NOT_ENOUGH_INFO recall | BLOCKED | 0.399137 |
| Argmax NOT_ENOUGH_INFO f1 | BLOCKED | 0.283416 |
| Contradiction saved-threshold precision | BLOCKED | 0.24486571879936808 |
| Contradiction saved-threshold recall | BLOCKED | 0.5236486486486487 |
| Contradiction saved-threshold f1 | BLOCKED | 0.333692142088267 |
| Contradiction saved-threshold fp | BLOCKED | 956 |
| Contradiction saved-threshold fn | BLOCKED | 282 |
| Top-label ECE (10 bins) | BLOCKED | 0.047256 |
| Multiclass Brier (sum per row; range 0–2) | BLOCKED | 0.211270 |

Archived argmax confusion matrix (true rows / predicted columns, SUPPORTED / CONTRADICTED / NEI):

| True class | Supported | Contradicted | NEI |
|---|---:|---:|---:|
| SUPPORTED | 15046 | 954 | 1258 |
| CONTRADICTED | 211 | 325 | 56 |
| NEI | 427 | 130 | 370 |

Archived hypothetical bypass at risk < 0.625: **16,180/18,777** rows (86.169250%); **715** non-supported eligible, including **226** contradicted; conditional false accepts **4.419036%**. Supported rows routed: **1793** (10.389385%). These archived scores are not the prior 18,770-row production-representation arm; do not mix their counts or claim that arm was reproduced. Locked-test policy/coverage results remain **BLOCKED**. No CIs were fabricated without source identities.

## Validation on the exact implementation snapshot

| Scope | Passed | Failed | Errors | Skipped | Exit |
|---|---:|---:|---:|---:|---:|
| framework | 42 | 0 | 0 | 0 | 0 |
| focused | 478 | 0 | 0 | 4 | 0 |
| full | 1566 | 0 | 0 | 25 | 0 |
| compileall | — | — | — | — | 0 |

Counts overlap and must not be added. The full command includes both the default backend roots and Detector tests. New fixtures are **SYNTHETIC_FRAMEWORK_TEST_ONLY**. Controlled Verifier routing/integration assertions are **MOCKED_VERIFIER_PATH**, never live passes. Default native live n8n/web opt-ins remain disabled. The exact execution-only prerequisite plugin already documented in Prompt1 is reused unchanged for six unmocked live-retrieval checks and five absent real-model checks; its source/hash is pinned in the manifests. No repository live tests were deleted or weakened.

The framework tests cover exact/group/cross-split/near detection, deterministic component assignment, stable manifests, every locked artifact mutation, no label remapping, calibration misuse/renamed locked rows, bootstrap reproducibility/undefined denominators, missing artifacts/exposure, historical membership leakage, calibration-history linkage and checkpoint mutation during evaluation. End-to-end synthetic stubs can never claim real independent success. No failures/errors remain in these feasible suites.

Executed commands:

```text
python -m pytest -q halluciguard_detector/tests/test_independent_evaluation.py --tb=short --disable-warnings -p baseline_blocked --junitxml=<scratch>/prompt2-framework.xml
python -m pytest -q halluciguard_detector/tests orchestration/tests/test_grounded_certification.py orchestration/tests/test_diagnostic_store_isolation.py orchestration/tests/test_final_integration_safety.py orchestration/tests/test_llm_detector_verifier_slice.py agents/verifier_agent/tests/test_retrieval_execution_status.py agents/verifier_agent/tests/test_execution_trace_repair.py agents/verifier_agent/tests/test_token_survival.py agents/verifier_agent/tests/test_relation_query_priority.py agents/verifier_agent/tests/test_incorporation_event_scope.py agents/verifier_agent/tests/test_port_backend_trace.py agents/verifier_agent/tests/test_verifier_hardening_regression.py agents/memory_agent/tests/test_update_consistency.py agents/memory_agent/tests/test_store_durability.py --tb=short --disable-warnings --show-capture=no -rs -p baseline_blocked --junitxml=<scratch>/prompt2-focused.xml
python -m pytest -q halluciguard_detector/tests orchestration/tests agents/verifier_agent/tests agents/memory_agent/tests agents/judge_agent/tests agents/corrector_agent/tests --tb=short --disable-warnings --show-capture=no -rs -p baseline_blocked --junitxml=<scratch>/prompt2-full.xml
python -m compileall -q halluciguard_detector agents/verifier_agent agents/memory_agent orchestration services scripts/evaluate_paired_detector.py scripts/smoke_retrieval_execution.py scripts/trace_pipeline_execution.py
git diff --check
git diff --cached --check
```

Every remaining full-suite skip:

| Exact nodeid | Classification | Exact reason |
|---|---|---|
| `orchestration/tests/test_verifier_stabilization.py::TestVerifierStabilization::test_single_claim_end_to_end_verification` | BLOCKED_LIVE_VERIFIER_RETRIEVAL | BLOCKED/UNVERIFIED: live Verifier retrieval requires unavailable owner laptop n8n or external network; no live check is credited |
| `orchestration/tests/test_verifier_stabilization.py::TestVerifierStabilization::test_wikipedia_adapter_retrieval` | BLOCKED_LIVE_VERIFIER_RETRIEVAL | BLOCKED/UNVERIFIED: live Verifier retrieval requires unavailable owner laptop n8n or external network; no live check is credited |
| `orchestration/tests/test_verifier_v1_stabilization.py::TestVerifierV1Stabilization::test_1_paris_capital_of_france` | BLOCKED_LIVE_VERIFIER_RETRIEVAL | BLOCKED/UNVERIFIED: live Verifier retrieval requires unavailable owner laptop n8n or external network; no live check is credited |
| `orchestration/tests/test_verifier_v1_stabilization.py::TestVerifierV1Stabilization::test_2_eiffel_tower_in_london` | BLOCKED_LIVE_VERIFIER_RETRIEVAL | BLOCKED/UNVERIFIED: live Verifier retrieval requires unavailable owner laptop n8n or external network; no live check is credited |
| `orchestration/tests/test_verifier_v1_stabilization.py::TestVerifierV1Stabilization::test_3_moon_made_of_green_cheese` | BLOCKED_LIVE_VERIFIER_RETRIEVAL | BLOCKED/UNVERIFIED: live Verifier retrieval requires unavailable owner laptop n8n or external network; no live check is credited |
| `orchestration/tests/test_verifier_v1_stabilization.py::TestVerifierV1Stabilization::test_4_aspirin_mild_pain_healthcare` | BLOCKED_LIVE_VERIFIER_RETRIEVAL | BLOCKED/UNVERIFIED: live Verifier retrieval requires unavailable owner laptop n8n or external network; no live check is credited |
| `agents/verifier_agent/tests/test_n8n_retrieval.py::test_pipeline_uses_n8n_evidence_and_runs_python_bge_nli` | BLOCKED_REAL_VERIFIER_MODELS | BLOCKED/UNVERIFIED: real Verifier BGE/DeBERTa weights are absent from the fresh offline cache; mock contract checks do not establish real-model success |
| `agents/verifier_agent/tests/test_n8n_retrieval.py::test_pipeline_fallback_to_python_adapters_on_n8n_failure` | BLOCKED_REAL_VERIFIER_MODELS | BLOCKED/UNVERIFIED: real Verifier BGE/DeBERTa weights are absent from the fresh offline cache; mock contract checks do not establish real-model success |
| `agents/verifier_agent/tests/test_n8n_retrieval.py::test_no_duplicate_retrieval_when_n8n_succeeds` | BLOCKED_REAL_VERIFIER_MODELS | BLOCKED/UNVERIFIED: real Verifier BGE/DeBERTa weights are absent from the fresh offline cache; mock contract checks do not establish real-model success |
| `agents/verifier_agent/tests/test_n8n_retrieval.py::test_live_n8n_health_endpoint` | BLOCKED_LIVE_N8N_UNAVAILABLE | Live n8n tests require RUN_LIVE_N8N_TESTS=true environment variable |
| `agents/verifier_agent/tests/test_n8n_retrieval.py::test_live_n8n_retrieval_webhook` | BLOCKED_LIVE_N8N_UNAVAILABLE | Live n8n tests require RUN_LIVE_N8N_TESTS=true environment variable |
| `agents/verifier_agent/tests/test_pipeline_hardening.py::TestDirectModelDiagnostics::test_direct_deberta_nli_execution` | BLOCKED_REAL_VERIFIER_MODELS | BLOCKED/UNVERIFIED: real Verifier BGE/DeBERTa weights are absent from the fresh offline cache; mock contract checks do not establish real-model success |
| `agents/verifier_agent/tests/test_pipeline_hardening.py::TestDirectModelDiagnostics::test_direct_bge_reranker_execution` | BLOCKED_REAL_VERIFIER_MODELS | BLOCKED/UNVERIFIED: real Verifier BGE/DeBERTa weights are absent from the fresh offline cache; mock contract checks do not establish real-model success |
| `agents/verifier_agent/tests/test_verifier_hardening_regression.py::TestDetectorDiagnostics::test_baseline_fallback_is_flagged_degraded` | OBSOLETE_HALUEVAL_IMPORT | detector deps unavailable: ModuleNotFoundError("No module named 'agents.detector_agent'") |
| `agents/verifier_agent/tests/test_verifier_hardening_regression.py::TestDetectorDiagnostics::test_real_inference_is_flagged_executed` | OBSOLETE_HALUEVAL_IMPORT | detector deps unavailable: ModuleNotFoundError("No module named 'agents.detector_agent'") |
| `agents/verifier_agent/tests/test_verifier_hardening_regression.py::TestDetectorDiagnostics::test_default_result_edge_case_is_degraded` | OBSOLETE_HALUEVAL_IMPORT | detector deps unavailable: ModuleNotFoundError("No module named 'agents.detector_agent'") |
| `agents/verifier_agent/tests/test_verifier_hardening_regression.py::TestDetectorDiagnostics::test_diagnostic_fields_default_safe_on_bare_construction` | OBSOLETE_HALUEVAL_IMPORT | detector deps unavailable: ModuleNotFoundError("No module named 'agents.detector_agent'") |
| `agents/verifier_agent/tests/test_web_retriever.py::TestLiveIntegration::test_live_tavily_search` | UNVERIFIED_OPT_IN_WEB | Live web tests disabled (set RUN_LIVE_WEB_TESTS=true) |
| `agents/verifier_agent/tests/test_web_retriever.py::TestLiveIntegration::test_live_full_chain` | UNVERIFIED_OPT_IN_WEB | Live web tests disabled (set RUN_LIVE_WEB_TESTS=true) |
| `agents/corrector_agent/tests/test_step4a_contract_alignment.py::test_dataset_files_exist_and_non_empty` | UNRESOLVED_CORRECTOR_CURRICULUM | contract_v4 curriculum dataset not present; build it via the training pipeline before enabling these checks |
| `agents/corrector_agent/tests/test_step4a_contract_alignment.py::test_all_training_targets_are_strict_json` | UNRESOLVED_CORRECTOR_CURRICULUM | contract_v4 curriculum dataset not present; build it via the training pipeline before enabling these checks |
| `agents/corrector_agent/tests/test_step4a_contract_alignment.py::test_sentence_id_authorization_in_training_data` | UNRESOLVED_CORRECTOR_CURRICULUM | contract_v4 curriculum dataset not present; build it via the training pipeline before enabling these checks |
| `agents/corrector_agent/tests/test_step4a_contract_alignment.py::test_prompt_fencing_integrity` | UNRESOLVED_CORRECTOR_CURRICULUM | contract_v4 curriculum dataset not present; build it via the training pipeline before enabling these checks |
| `agents/corrector_agent/tests/test_step4a_contract_alignment.py::test_zero_data_leakage_between_train_and_test` | UNRESOLVED_CORRECTOR_CURRICULUM | contract_v4 curriculum dataset not present; build it via the training pipeline before enabling these checks |
| `agents/corrector_agent/tests/test_step4a_contract_alignment.py::test_edge_case_coverage` | UNRESOLVED_CORRECTOR_CURRICULUM | contract_v4 curriculum dataset not present; build it via the training pipeline before enabling these checks |

The two native live-n8n checks are **BLOCKED_LIVE_N8N_UNAVAILABLE**. Live-Verifier checks remain unverified. Six Corrector curriculum checks remain unresolved; four obsolete HaluEval imports do not imply failure of the real Detector suite. Separate real-model/web prerequisites remain blocked/unverified. None count as release-validation success.

Protected artifacts: all 12 checkpoint/Prompt1 files remain unchanged, and 729 other original tracked files retain their hashes (only the three intentional offline fitting/test files are excluded from that original-source comparison). Compilation, JSON metadata/hash linkage and diff checks pass. Only code/tests/docs/metadata are committed; no weights, restricted raw datasets, credentials, private traces, production Memory/cache or backup stores are added. New fit-time role guards change offline tooling only; Detector inference, thresholds, weights, Judge and fast-path behavior remain frozen.

## Blockers before model research

- Approved prepared train/dev/test claim+evidence rows with immutable row/source/response IDs are absent.
- Raw RAGTruth source_info.jsonl/response.jsonl and the prior paired-experiment artifacts are absent.
- Historical complete checkpoint-linked training and calibration exposure, source hashes and dataset version are not reconstructed by the saved logits.
- Prior report records five exact train/dev pairs and seven excluded test matches; actual rows are absent, so current exact/near counts cannot be measured.
- No legitimate final locked-test artifact/manifest exists; no independent metrics, source-cluster intervals or locked-test bypass results can be computed.

Recover approved historical inputs and complete exposure provenance, run exact/source/response/near audits, preserve/declare all exclusions, then lock a legitimate source-isolated final cohort before metrics. If historical calibration contamination is confirmed, the existing checkpoint cannot satisfy this clean gate by merely editing metadata or reshuffling rows. The owner must resolve the research/provenance plan before fitting new models. Live n8n is a separate release prerequisite; it is not needed for this offline Detector runner.

[Evaluation/split status manifest](detector-independent-evaluation-20261003.manifest.json) · [Machine-readable readiness status](production-readiness-status-20261003.json) · [Tooling and replay instructions](../detector-independent-evaluation.md)
