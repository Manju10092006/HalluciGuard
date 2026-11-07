# HalluciGuard reliability revision

The current branch changes the existing implementation. It does not replace the architecture or the frozen 22 page reviewed paper. New outputs are under `results/research_20261106_r1`. The date in the run identifier is the start date; reporting completed on 7 November 2026.

## Configurations

`RELATION_POLICY=structured` retains the original evidence policy. The alternatives are `no_scoped`, `nli_temporal`, and `soft`. `HALLUCIGUARD_RELEASE_POLICY` accepts `default`, `strict_all_verified`, and `slot_aware`. The historical Judge action behavior remains the default. Soft arbitration and alternate release policies are not validated deployment improvements.

Normal routing has `ALWAYS_VERIFY=true` and `ALLOW_DETECTOR_FAST_PATH=false`. A low score does not bypass verification under these defaults. Alternate Judge policies also force verification. The historical optional uncertainty head remains a separately configured experimental path.

No model weights, calibration files, historical datasets or archived predictions were changed. `baseline/inventory.json` gives hashes and original locations. `baseline/source` and `baseline/preexisting_changes.patch` preserve the entry code and relevant user edits. The archived technical candidate is a different Git revision and must not be called the current reliability candidate.

## Commands and evidence

Run scripts from the repository root with the project Python environment. The recorded environment is Python 3.13.2, PyTorch 2.13.0+cu126, Transformers 5.2.0 and scikit-learn 1.7.2. CPU experiments set CUDA_VISIBLE_DEVICES=-1 and Hugging Face offline flags. Each producer writes exclusively to new paths and refuses to overwrite results. To repeat a run, set a new `RUN` directory in a copy of `common.py`; retain the archived source hashes and inputs.

1. `python scripts/research_freeze_20261106.py` created the initial inventory. Do not rerun it over the saved entry state.
2. `python research/reliability_20261106/diagnose_repairs.py` performs fresh CPU NLI over 24 fixed candidates, with original passages and no provider calls.
3. `python research/reliability_20261106/replay_policies.py` executes three Judge policies on 180 saved states and four relation policies on the 100 reference targets. It reuses frozen neural scores and makes no provider calls.
4. `python research/reliability_20261106/evaluate_repairs.py` invokes the actual ReVerifier and Judge gates with the saved neural pairs. Extraction is the deterministic fallback. This is a component replay, not a fresh whole graph experiment.
5. `python research/reliability_20261106/heldout_verifier.py` selects 40 ANLI R2 test examples per class before fresh CPU inference. It records selection IDs and source hashes before producing predictions. No thresholds are selected from the resulting test labels.
6. `python research/reliability_20261106/detector_audit.py` reaggregates archived logits and checks processed splits. Historical logits lack row IDs and must not be joined to the current text rows by assumed order.
7. `python research/reliability_20261106/analyze_results.py` derives per-class metrics, paired bootstrap intervals, a case inventory, and two blank annotation CSVs.
8. `python research/reliability_20261106/run_external_baselines.py` is a paid-capable live runner and is not part of offline reproduction. The approved pilot used Groq only, 512 completion tokens per request, no retries, at most 320 requests, and stopped after three provider errors. It stopped at 42 requests. Do not silently restart or pool it with a later differently configured experiment. Its source was subsequently adjusted to preserve error codes and attempt trails; the archived configuration hash identifies the executed version.

The original external archive locations are centralized in `common.py`. The final evidence package includes the small inputs needed for the new replays where practical; checkpoints and full datasets remain at their inventoried locations. No credentials belong in a reproduction package.

## Validation

The component suite passed 99 cases. The broader suite passed 601 cases. The last targeted suite passed 11 cases, including overlapping Detector checks; these totals must not be added. The final Memory test file passed two cases after adding assertions that reopen the graph, vector store and SQLite cache and recover the same provenance. That run has 22 existing datetime deprecation warnings. XML reports preserve individual test names. The first broad command named a nonexistent test file and collected zero cases; that failed invocation remains recorded rather than being counted as a test run.

Some Windows sandbox runs hung in Python's asyncio loopback socket initialization. Running the same local tests with authorized access to the loopback socket resolved this environment issue. It was not a ReVerifier defect. The broad run produced one Starlette/httpx deprecation warning.

Manuscripts are compiled with Tectonic 0.17.0 using its frozen cache. `main.tex` is the two column article and `supplement.tex` the separate supplement. The original eight raster files remain in the source; only four approved versioned replacements are referenced. The conclusion has exactly two paragraphs, followed immediately by the bibliography. Do not treat successful compilation as proof of scientific validity.

## Interpretation limits

The soft relation rule improves known development cases and the eight diagnostic repairs, but not held-out ANLI agreement. It adds one false contradiction on R2 and stays optional. The two alternate Judge policies remove five known false targets while retaining the same 48 target-passing releases in replay. This is not independent whole answer correctness, and additional retries were not executed.

The new live baseline pilot is incomplete. It produced four basic RAG answers, three CoVe-style answers, two RARR-style answers and three SelfCheck-style score records, with three failed records and 65 unattempted method/question records. There is no new complete matched full-system factuality comparison. The original 32-card graph comparison and 20-question ordinary workflow comparison remain archived evidence, not results of the new source.

Memory provenance and failures are validated at storage boundaries. Cross-store writes are not atomic, duplicates do not merge new provenance, direct storage can bypass graph eligibility, and recall is not connected to generation. Human annotation and human-review outcomes remain pending. No general hallucination reduction, novelty determination, or journal acceptance is claimed.
