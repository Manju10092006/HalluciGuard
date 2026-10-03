# Production-readiness baseline — 2026-10-03

**Production readiness is NOT established. Fast-path acceptance remains disabled.**
The owner’s laptop-hosted n8n Verifier workflow is unavailable: every live-Verifier check is **BLOCKED/UNVERIFIED**, never a passed live validation. Successful mock/contract tests are engineering evidence only.

## Frozen Git baseline

Exact source commit tested: `82cdc6fb734f5fa31cd345a4060f60178bea2118`. Current `origin/main` at fetch was the same SHA. Both the required baseline and merged release-validation commit `6c6681833023a20d2bc45fa6ee1b47d88757661e` were verified with `git merge-base --is-ancestor` (exit 0).
The remote branch did not exist when checked using the GitHub plugin and `git ls-remote`. A fresh clone was made; Git LFS fetched the already tracked checkpoint. The new branch was created from `origin/main`, with a clean worktree before tests. Main was not reset or rewritten. No PR was opened. This publication changes only this report and its JSON manifest; the tested source is their parent baseline.

## Detector baseline and provenance

| Item | Frozen state |
|---|---|
| Checkpoint SHA256 | `470261f81870f5c4f97fe85918d02aa08673ac08744b5cae4d4621ee9d631bf9` — matches owner/prior report |
| Calibration SHA256 recorded previously | `7f1b411d4943141f8ab80fc4f8696917828dd41b85a9ea783c37039b9a7911a8` |
| Calibration SHA256 actually in this checkout | `3ce4724edd70d41cbf5975cf65643ddf1a8f3479bb983e324fb19789a27a3f32` — LF Git blob/checkout; CRLF conversion exactly matches prior hash |
| Parameters, unchanged | Temperature 0.7586129307746887; contradiction threshold 0.4599999999999999 (~0.46); risk threshold 0.625; maximum sequence length 256 |
| Last paired production-representation metrics | Accuracy 83.7666%; contradiction precision 24.4866%, recall 52.3649%, F1 33.3692% |
| Last paired historical-evidence accuracy | 83.8253% |
| Existing archived test-metrics cohort | 18,777 claims; three-class accuracy 83.8313%; thresholded contradiction P/R/F1 24.4866% / 52.3649% / 33.3692% |
| Acceptance gates | `FAST_PATH_RELEASE_ENABLED = False`; run flag `ALLOW_DETECTOR_FAST_PATH=false` |

The checkpoint bytes were independently hashed. The calibration values agree with the prior report, and the byte-level difference is explained: converting LF line endings to CRLF produces exactly the prior recorded hash. The Git blob hash independently matches this checkout. Neither calibration nor any Detector weights were edited. This deterministic transformation explains the hash difference without claiming access to the original historical file.

The last measured paired results come from [detector-release-validation-20261002.md](detector-release-validation-20261002.md), on 18,770 claims, after seven cross-split duplicates were excluded. They were **not remeasured here**. Five train/dev exact pairs remain in that historical experiment; near duplicates were not audited. Contradiction precision is weak, held-out calibration remains uncertain, and these metrics do not establish safe acceptance.

The JSON manifest hashes every existing Detector artifact, related source/test path, evaluation script, Detector report and all report references. It includes all ten `artifacts/detector-best/` files (weights, calibration, model/tokenizer configuration, metadata, predictions, metrics and training report), the six `reports/detector_v2_*` files, phase-1/paired-evaluation implementation references and dependency files. No preexisting manifest was found in the tracked baseline. The prepared RAGTruth dataset, raw response file and ignored paired-evaluation output directory are absent; they have explicit unavailable status and no invented hashes. Independent evaluation/research must recover those inputs and account for the recorded LF/CRLF normalization before claiming an exact historical replay.

## Clean environment and replay

Fresh virtual environment without system packages; Python **3.12.14**, Ubuntu **24.04.3**, Linux **6.18.44**, x86_64. Repository `.python-version` specifies **3.11.9**; the prior audit described **3.13.2**. This environment does not reproduce either interpreter exactly. CPU execution was requested; CUDA package installation does not mean GPU inference was used.

The canonical root `requirements.txt` was installed with `pytest` and `pytest-asyncio`. These are version ranges, not a full Python lock. The manifest records every resolved package, complete `pip freeze --all`, its SHA256, all dependency-file hashes and the frontend lock hash. The frontend was not built. `socksio==1.0.0` was subsequently installed because the supplied workspace SOCKS proxy is inherited by httpx; it is an execution-environment dependency, not a source change. `pip check` exited 0. Required imports succeeded.

Key resolved versions: torch 2.14.1+cu130, Transformers 5.18.0, sentence-transformers 6.1.0, NumPy 2.5.3, Pydantic 2.13.5, FastAPI 0.142.2, Starlette 1.7.0, pytest 9.1.1 and pytest-asyncio 1.4.0.

Flags:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES=-1
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
ALLOW_DETECTOR_FAST_PATH=false
RUN_LIVE_N8N_TESTS=false
RUN_LIVE_WEB_TESTS=false
HF_HOME=<scratch>/baseline-hf-cache
PYTHONPATH=<scratch>
```

The HF cache started empty. After setup errors, the public Memory embedding model `sentence-transformers/all-MiniLM-L6-v2` was downloaded (resolved revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`) and its 11 files were hash-indexed; offline inference returned a 384-dimensional embedding. Final tests ran offline. The Verifier’s BGE/DeBERTa model prerequisites remained unavailable. Owner API-key/n8n URL variables were removed; existing repository fixtures provide dummy n8n URLs for mocked transports. No owner laptop workflow was contacted successfully.

The initial unmodified release commands are retained in the manifest as unsuccessful attempts, including exact failed/error nodeids. Full suite: **1,196 passed, 24 failed, 59 errors, 14 skipped**. Focused suite: **429 passed, 0 failed, 7 errors, 4 skipped**. The intermediate rerun after socksio had **1,209 passed, 0 failed, 59 errors, 25 skipped**, and **429 passed, 0 failed, 7 errors, 4 skipped** focused; all setup errors required the uncached public Memory model. These are preserved separately. Many initial failures/errors arose when httpx could not construct its SOCKS transport; the final rerun establishes which resolve after installing socksio. Remaining real-model/live-retrieval prerequisites were explicitly blocked. These initial attempts are not counted as successful validation.

For the final offline replay, an execution-only pytest plugin adds skips to exactly six unmocked live-retrieval checks and five real BGE/DeBERTa checks. This is an explicit deviation from the original commands, not a code fix or a passing substitute. Its complete source and hash are embedded in the manifest. To replay, write `validation.execution_only_blocker_plugin.source` to `<scratch>/baseline_blocked.py`, activate the fresh venv, apply the recorded flags and execute the following commands. The manifest’s freeze provides exact resolved versions. JUnit paths are output locations only.

```text
python -m pytest -q --tb=short --disable-warnings --show-capture=no -rs -p baseline_blocked --junitxml=<scratch>/baseline-full.xml
python -m pytest -q halluciguard_detector/tests orchestration/tests/test_grounded_certification.py orchestration/tests/test_diagnostic_store_isolation.py orchestration/tests/test_final_integration_safety.py orchestration/tests/test_llm_detector_verifier_slice.py agents/verifier_agent/tests/test_retrieval_execution_status.py agents/verifier_agent/tests/test_execution_trace_repair.py agents/verifier_agent/tests/test_token_survival.py agents/verifier_agent/tests/test_relation_query_priority.py agents/verifier_agent/tests/test_incorporation_event_scope.py agents/verifier_agent/tests/test_port_backend_trace.py agents/verifier_agent/tests/test_verifier_hardening_regression.py agents/memory_agent/tests/test_update_consistency.py agents/memory_agent/tests/test_store_durability.py --tb=short --disable-warnings --show-capture=no -rs -p baseline_blocked --junitxml=<scratch>/baseline-focused.xml
python -m pytest -q agents/verifier_agent/tests/test_nli_canonicalization.py agents/verifier_agent/tests/test_token_survival.py --tb=short --disable-warnings --show-capture=no -p baseline_blocked --junitxml=<scratch>/baseline-schema.xml
python -m compileall -q halluciguard_detector agents/verifier_agent agents/memory_agent orchestration services scripts/evaluate_paired_detector.py scripts/smoke_retrieval_execution.py scripts/trace_pipeline_execution.py
git diff --check
git diff --cached --check
```

## Final validation results

| Run | Passed | Failed | Errors | Skipped | Exit | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| full | 1282 | 0 | 0 | 25 | 0 | 379.125 |
| focused | 436 | 0 | 0 | 4 | 0 | 16.299 |
| schema | 6 | 0 | 0 | 0 | 0 | 7.879 |
| compileall | — | — | — | — | 0 | 0.044 |

Focused/schema counts overlap the full suite; they must not be added together. The Detector suite is explicitly included in the focused invocation because root pytest defaults omit it. Exact command strings, outcome counts, durations, skip names/reasons and log/JUnit hashes are in the manifest. Raw logs/XML remain local and are not publication content.

Compared with 1,293 passed / 14 skipped, the final full-suite deviation is `{"errors": 0, "failed": 0, "passed": -11, "skipped": 11}`. The additional 11 skips are environment/prerequisite blockers. Trio installation restored 14 AnyIO backend variants that were not collected in the earlier attempts because current AnyIO only collects installed backends. The historical full baseline is **not fully reproduced**; blocked checks are not passes. Acceptance is qualified to the runnable offline/mock suite; full release-validation equivalence remains unresolved.

## Every remaining full-suite skip

| Exact test nodeid | Classification | Exact reason |
|---|---|---|
| `orchestration/tests/test_verifier_stabilization.py::TestVerifierStabilization::test_single_claim_end_to_end_verification` | BLOCKED_LIVE_VERIFIER | BLOCKED/UNVERIFIED: live Verifier retrieval requires unavailable owner laptop n8n or external network; no live check is credited |
| `orchestration/tests/test_verifier_stabilization.py::TestVerifierStabilization::test_wikipedia_adapter_retrieval` | BLOCKED_LIVE_VERIFIER | BLOCKED/UNVERIFIED: live Verifier retrieval requires unavailable owner laptop n8n or external network; no live check is credited |
| `orchestration/tests/test_verifier_v1_stabilization.py::TestVerifierV1Stabilization::test_1_paris_capital_of_france` | BLOCKED_LIVE_VERIFIER | BLOCKED/UNVERIFIED: live Verifier retrieval requires unavailable owner laptop n8n or external network; no live check is credited |
| `orchestration/tests/test_verifier_v1_stabilization.py::TestVerifierV1Stabilization::test_2_eiffel_tower_in_london` | BLOCKED_LIVE_VERIFIER | BLOCKED/UNVERIFIED: live Verifier retrieval requires unavailable owner laptop n8n or external network; no live check is credited |
| `orchestration/tests/test_verifier_v1_stabilization.py::TestVerifierV1Stabilization::test_3_moon_made_of_green_cheese` | BLOCKED_LIVE_VERIFIER | BLOCKED/UNVERIFIED: live Verifier retrieval requires unavailable owner laptop n8n or external network; no live check is credited |
| `orchestration/tests/test_verifier_v1_stabilization.py::TestVerifierV1Stabilization::test_4_aspirin_mild_pain_healthcare` | BLOCKED_LIVE_VERIFIER | BLOCKED/UNVERIFIED: live Verifier retrieval requires unavailable owner laptop n8n or external network; no live check is credited |
| `agents/verifier_agent/tests/test_n8n_retrieval.py::test_pipeline_uses_n8n_evidence_and_runs_python_bge_nli` | BLOCKED_REAL_VERIFIER_MODELS | BLOCKED/UNVERIFIED: real Verifier BGE/DeBERTa weights are absent from the fresh offline cache; mock contract checks do not establish real-model success |
| `agents/verifier_agent/tests/test_n8n_retrieval.py::test_pipeline_fallback_to_python_adapters_on_n8n_failure` | BLOCKED_REAL_VERIFIER_MODELS | BLOCKED/UNVERIFIED: real Verifier BGE/DeBERTa weights are absent from the fresh offline cache; mock contract checks do not establish real-model success |
| `agents/verifier_agent/tests/test_n8n_retrieval.py::test_no_duplicate_retrieval_when_n8n_succeeds` | BLOCKED_REAL_VERIFIER_MODELS | BLOCKED/UNVERIFIED: real Verifier BGE/DeBERTa weights are absent from the fresh offline cache; mock contract checks do not establish real-model success |
| `agents/verifier_agent/tests/test_n8n_retrieval.py::test_live_n8n_health_endpoint` | BLOCKED_LIVE_VERIFIER | Live n8n tests require RUN_LIVE_N8N_TESTS=true environment variable |
| `agents/verifier_agent/tests/test_n8n_retrieval.py::test_live_n8n_retrieval_webhook` | BLOCKED_LIVE_VERIFIER | Live n8n tests require RUN_LIVE_N8N_TESTS=true environment variable |
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

**The six Corrector curriculum checks remain unresolved**: all four generated `contract_v4` curriculum artifacts are missing, and no replacement data was fabricated. The four obsolete HaluEval import skips remain separate from the actual Detector tests. Opt-in live web/n8n checks remain unverified; historical live diagnostics cannot turn them into passes.

## Integrity, scope and follow-up

All **732** preexisting tracked files were SHA256 checked before and after execution: **zero mismatches**. This includes checkpoint, calibration, thresholds, Judge policy and production code. No production behavior changed. `compileall` and both diff checks passed. Only the two report/manifest files are committed; post-commit Git status and remote branch SHA are checked during publication.

No weights, private datasets, credentials, Memory stores, raw private traces or backups were added. Tests may create ignored local diagnostic/cache fixtures in this fresh clone; those are not production stores or committed artifacts.

Remaining prerequisites: preserve the recorded calibration LF/CRLF hash distinction; obtain the actual historical evaluation inputs with hashes; rerun the 11 blocked checks in a controlled environment with the required Verifier models and owner laptop workflow; resolve the six Corrector curriculum artifacts; validate independent Detector accuracy/calibration and production policy/performance. All prior release-report blockers remain open.

[Machine-readable reproducibility manifest](production-readiness-baseline-20261003.manifest.json)
