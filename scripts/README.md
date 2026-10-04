# Scripts

The scripts directory contains targeted diagnostics and validation utilities, not product import packages.

| Group | Examples |
|---|---|
| Full demonstrations | `prove_all_agents_e2e.py`, `demo_three_agents_e2e.py` |
| LLM/provider checks | `test_openrouter_llm.py`, `test_llm_detector_slice.py` |
| Retrieval/n8n checks | `test_n8n_retrieval.py`, `test_web_evidence.py`, `repair_n8n_workflow.py` |
| Verifier diagnostics | `diagnose_verifier.py`, `diagnose_verifier_e2e.py`, `verify_claim.py` |
| Certification/readiness | `deployment_readiness_check.py`, `verify_certification_sandbox.py` |
| Documentation | `build_master_doc.py`, `generate_diagrams.py` |

Probe and audit scripts created for one-off debugging should remain local unless they are generalized, credential-safe and documented. Runtime output belongs in ignored log/report paths.

## Live seven-agent walkthrough

Run these commands from the repository root, using an environment with the project dependencies installed. Configure the real LLM and retrieval services in your local `.env`; never commit that file or its credentials. Start your local n8n workflow and its tunnel when configured. The existing production checkpoint must be available; the runner does not train or replace it.

The Generator is the Base LLM, separate from the seven downstream agents: Detector, Claim Analyzer, Verifier, Judge, Corrector, ReVerifier and Memory.

For a normal question, the Base LLM generates the initial response. Do not supply `--draft`:

```powershell
python test_walkthrough.py "Who founded Microsoft?" --checkpoint artifacts/detector-best
python test_walkthrough.py "What is the cost of Dell G15 laptop?" --checkpoint artifacts/detector-best
```

`test_walkthrough.py` calls `scripts/run_seven_agent_walkthrough.py`. It prints the narrated production execution and automatically saves reports in a fresh directory such as `Downloads/HalluciGuard-walkthrough-<UTC timestamp>-<unique suffix>`. The actual absolute output path is printed at the end. Optional `--output-dir` must name a directory that does not already exist.

For the deliberate correction diagnostic only, supply the incorrect initial answer. This skips initial Base LLM generation, but does not inject retrieval evidence, force correction or bypass re-verification:

```powershell
python test_walkthrough.py "Who founded Microsoft?" --draft "Snehith is the founder of Microsoft." --checkpoint artifacts/detector-best
```

The Corrector runs only when the production Judge requests it. Insufficient evidence may correctly withhold the answer rather than produce a repair. A laptop price also depends on configuration, seller, location and date; acceptance is not guaranteed merely because the command completed.

To create one package containing the fixed Dell live question and Microsoft supplied-answer diagnostic, use the validation runner:

```powershell
$reportPath = Join-Path ([Environment]::GetFolderPath('UserProfile')) ("Downloads/HalluciGuard-seven-agents-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + "-" + [guid]::NewGuid().ToString('N').Substring(0, 6))
python scripts/run_seven_agent_validation.py --suite --checkpoint artifacts/detector-best --output-dir "$reportPath"
```

For a custom question, replace `--suite` with `--query "Your question"`. The validation runner requires `--output-dir`; the narrated walkthrough does not. Both support `--env-file`, `--checkpoint` and `--timeout-seconds`.

## Report files and execution meanings

Each run exports these human-readable files:

- `01_ALL_7_AGENTS_END_TO_END.txt`
- `02_DETECTOR_AGENT.txt`
- `03_CLAIM_ANALYZER_AGENT.txt`
- `04_VERIFIER_AGENT.txt`
- `05_JUDGE_AGENT.txt`
- `06_CORRECTOR_AGENT.txt`
- `07_REVERIFIER_AGENT.txt`
- `08_MEMORY_AGENT.txt`
- `09_CONTRADICTION_CORRECTION_TEST.txt`
- `10_NORMAL_LIVE_QUERY.txt`
- `SUMMARY.txt`

`10_DELL_G15_LIVE_QUERY.txt` is added only for an actual Dell G15 normal query. `WALKTHROUGH.txt` is additionally written by the narrated walkthrough runner. When one of the two test categories was not requested, its report explicitly says it was not run. Full observed handoffs, evidence and component outputs are retained in `raw/*_execution.json`; the console transcript and per-agent JSON are also in `raw/`.

Statuses are execution facts, not accuracy scores:

- `FIRED`: the agent actually executed; this is not a factual PASS.
- `NOT_REQUIRED`: production routing did not require the conditional work, or Memory persistence was not required.
- `NOT_FIRED`: the stage was not reached; this does not imply success.
- `FAILED`: an observed execution or returned contract reported failure/degradation.

The Verifier report shows actual source URLs, candidate counts, backend availability/execution/fallback diagnostics, BGE ranking and observed NLI outputs where available. Ranking is relevance, not factual support. Exported decision scores can contain deterministic relation-rule adjustments; raw observed neural NLI probabilities are presented separately. Selected text is not proof that all of it survived tokenizer truncation. Unavailable values remain unavailable, and genuine zero probabilities are not replaced with invented values.

Exit code 0 means execution and report export completed, not that all claims were verified. Inspect the terminal decision, verdicts, failures and status counts. No passing-test or successful-run result establishes improved precision, recall or calibration.

## Memory and replay safety

These diagnostic runners execute the real Memory implementation using fresh per-run stores under `<output-dir>/runtime/`, rather than writing the user's existing Memory records. Runtime paths include `knowledge_graph.json`, `memory-cache.db`, `vector_store/`, `patterns.db` and `source-trust.db`. Verification caching is disabled for the diagnostic run; `ALWAYS_VERIFY=true` and `ALLOW_DETECTOR_FAST_PATH=false` are enforced. Models, retrieval evidence and factual verdicts are not mocked.

Inspect `08_MEMORY_AGENT.txt` for the actual storage/skip status and fact IDs. A graph Memory boundary can execute without storing anything; only eligible accepted verified facts are persisted. Do not re-ingest diagnostic stores into production merely to demonstrate persistence.

Replay is explicitly separate from a live test:

```powershell
python test_walkthrough.py --from-json "<report-directory>/raw/QUERY_RUN_execution.json"
```

Replay prints `REPLAY OF RECORDED EXECUTION` and performs no new LLM, retrieval or model calls. `--details` adds the fuller diagnostic view. Reports and runtime stores remain local; do not commit logs, databases, checkpoints, raw responses or personal evidence. Review reports before sharing them even though the runner redacts configured credentials.
