# Run all seven agents and inspect Memory

The Generator is the Base LLM. The seven downstream agents are Detector, Claim Analyzer, Verifier, Judge, Corrector, ReVerifier and Memory. The Detector also has a later grounded boundary after evidence retrieval; that is not an eighth agent.

## Run a real question

From the repository root, after installing the project dependencies, configuring your private `.env` and starting n8n/tunneling if enabled:

```powershell
python test_walkthrough.py "Who founded Microsoft?" --checkpoint artifacts/detector-best
python test_walkthrough.py "Your own question" --checkpoint artifacts/detector-best
```

The Base LLM actually generates the initial answer. The production graph extracts claims, retrieves evidence, evaluates them, and follows the Judge's route. A Corrector/ReVerifier run is conditional, not mandatory just because their nodes exist. Answers without sufficient evidence can remain withheld.

For a laptop price, specify model/configuration, country, currency and date. An unspecific or unsupported price is not made factual merely to produce a pleasing demonstration.

For a deliberately incorrect supplied-answer diagnostic only:

```powershell
python test_walkthrough.py "Who founded Microsoft?" --draft "Snehith is the founder of Microsoft." --checkpoint artifacts/detector-best
```

This skips initial generation only. Retrieval evidence and repair are not injected. Corrector runs only if the Judge requests it; a corrected candidate requires re-verification before release.

Optional CPU settings used for the recorded laptop runs:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:HALLUCIGUARD_MAX_MODELS = '2'
```

These are session settings, not changes to checkpoint weights or calibration.

## Where the TXT reports are saved

The console prints the actual absolute directory. By default the walkthrough creates a new `Downloads/HalluciGuard-walkthrough-<timestamp>-<suffix>` directory. `--output-dir` must be a fresh nonexistent directory; reports are not overwritten.

Files include:

1. `01_ALL_7_AGENTS_END_TO_END.txt`
2. `02_DETECTOR_AGENT.txt`
3. `03_CLAIM_ANALYZER_AGENT.txt`
4. `04_VERIFIER_AGENT.txt`
5. `05_JUDGE_AGENT.txt`
6. `06_CORRECTOR_AGENT.txt`
7. `07_REVERIFIER_AGENT.txt`
8. `08_MEMORY_AGENT.txt`
9. `09_CONTRADICTION_CORRECTION_TEST.txt`
10. `10_NORMAL_LIVE_QUERY.txt`
11. `SUMMARY.txt`
12. `WALKTHROUGH.txt` from the narrated runner.

The original observations, complete component inputs/outputs and console recording are under `raw/`. A Dell-specific file is created only for an actual Dell query. A report for a category not requested is explicitly not run.

Open a saved text report in Windows:

```powershell
notepad "<report-directory>\01_ALL_7_AGENTS_END_TO_END.txt"
notepad "<report-directory>\08_MEMORY_AGENT.txt"
```

`FIRED` means observed execution, not factual correctness. `NOT_REQUIRED` is not a passing execution. Exit 0 means the runner/export completed; inspect the final decision separately. Source URLs, BGE relevance and raw neural NLI values are distinguished from relation-adjusted decision scores. Unavailable values stay unavailable, genuine zeros stay zero, and selected text does not prove tokenizer survival.

## Inspect what Memory stored

The test runners use fresh diagnostic stores under `<report-directory>/runtime/`; they do not add demo facts to your pre-existing production Memory.

```powershell
python scripts/inspect_memory.py --runtime-dir "<report-directory>/runtime"
python scripts/inspect_memory.py --runtime-dir "<report-directory>/runtime" --show-text --limit 10
```

For default production paths:

```powershell
python scripts/inspect_memory.py --data-dir data
```

The inspector does not load `.env`, embedding models, cache APIs or MemoryAgent. It reads JSON and opens SQLite with a read-only connection. It reports missing/malformed stores and inconsistent representations, without repairing or deleting anything. Claim text is opt-in. Relative paths are selected explicitly; if your service uses different path overrides, inspect the corresponding directory rather than assuming default `data/` is active.

Memory retains claim text, verdict, operational confidence, IDs, graph relationships, cache timestamps, source counts and trust/pattern state. Full source excerpts/URLs and request metadata are not all persisted by current Memory storage; use the raw execution record for citations. Stored verified status is the pipeline's verdict, not proof of current truth.

For a local Memory service UI/API:

```powershell
python -m uvicorn agents.memory_agent.api.main:app --host 127.0.0.1 --port 8005
# Open http://127.0.0.1:8005/docs
```

API startup/shutdown and recall/cache endpoints can mutate storage or counters. This is not strict read-only inspection. Configure API-key protection and do not expose private Memory publicly. See [Memory details and lifecycle limitations](../agents/memory_agent/README.md).

## Validation and future work

Keep `ALWAYS_VERIFY=true` and `ALLOW_DETECTOR_FAST_PATH=false`. No new model, calibration or low-risk bypass was approved by the execution tests. Compare source-disjoint held-out predictions before making precision/recall claims. Current release concerns include weak evidence admission, temporal/entity/model errors, unobserved tokenizer survival, incomplete Memory update/delete consistency and unvalidated concurrent/distributed operation. The [dated review](reports/execution-repair-review-20261004.md) separates tested repairs from those remaining risks.
