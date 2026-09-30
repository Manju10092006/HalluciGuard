# Detector Phase 1: generation-bound uncertainty (experimental)

Phase 1 is an optional **pre-evidence** risk predictor. It estimates the likelihood of a human `HALLUCINATION_RELATED` claim label from the exact local generation's token score features. This is **not** a factual verification verdict. Phase 2 (`artifacts/detector-best`) remains the evidence-grounded DeBERTa detector, and Verifier/Judge retain their roles.

The design is informed by [Shelmanov et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1809/) and its [MIT-licensed implementation](https://github.com/IINemo/llm-uncertainty-head), but **does not reproduce its attention-based LUH architecture or use its published checkpoints**. It is a small logistic head over generated-token negative log-likelihood, entropy, top-two probability margin, and token count, aggregated per sentence. A published head for a different generator must not be loaded here.

## Current status and compatibility

No Phase 1 checkpoint was trained or evaluated in this repository update. Existing RAGTruth labels come from different generation events and **cannot** be matched to this generator's token logits. There is no local human-labeled same-event trace dataset. Consequently no Phase 1 accuracy, calibration, false-accept, or coverage figure is claimed. Normal Groq/Gemini/OpenRouter responses expose text but not the required generation trace: Phase 1 returns `unavailable` and the answer goes to Verifier. Default `disabled` mode also always verifies. The local generator is opt-in; the available 6 GB RTX 3050 laptop is not suitable for the paper's example 7B configuration. A real Qwen2.5-0.5B-Instruct generation on this machine produced an aligned schema-v2 trace. Raw pre-sampling logits are required: sampled `output.scores` in the installed Transformers version were filtered to one finite token and could not supply usable entropy. Pin exact model and tokenizer revisions and gather genuine labels on their own generations.

| Generator path | Exact trace | Compatible Phase 1 head now | Route |
|---|---:|---:|---|
| Groq/Gemini/OpenRouter default | No | No | Verify |
| Opt-in `local_uq` with pinned revisions | Yes, if alignment succeeds | No trained artifact yet | Verify |
| Same local generator with validated head | Yes | Only after training/calibration | Shadow or conditionally bypass |

## Runtime

```powershell
# In the HalluciGuard repository; ordinary deployment needs none of these changes.
$env:HALLUCIGUARD_PHASE1_MODE = "shadow"  # disabled | shadow | fast_path
$env:HALLUCIGUARD_LLM_PROVIDER = "local_uq"
$env:HALLUCIGUARD_PHASE1_MODEL_ID = "<exact local or Hub model ID>"
$env:HALLUCIGUARD_PHASE1_MODEL_REVISION = "<immutable commit revision>"
$env:HALLUCIGUARD_PHASE1_TOKENIZER_ID = "<exact tokenizer ID>"
$env:HALLUCIGUARD_PHASE1_TOKENIZER_REVISION = "<immutable commit revision>"
$env:HALLUCIGUARD_PHASE1_HEAD_DIR = "artifacts/phase1-uq"
# Explicitly opt in only when model files need downloading:
# $env:HALLUCIGUARD_PHASE1_ALLOW_DOWNLOAD = "true"
```

`fast_path` alone does not bypass. A checkpoint must pass schema/checksum/model-identity checks; calibration must authorize a validation-selected response threshold; the response's **maximum claim risk** must be below that threshold; domain must be allowlisted; all claim spans must align; and the existing `ALWAYS_VERIFY=false` plus `ALLOW_DETECTOR_FAST_PATH=true` gates must also be set. Stress-test mode still verifies. Missing or invalid prerequisites return an explicit unavailable reason and verify. The trace is transient orchestration state and is cleared after Detector scoring. `phase1_result` retains score provenance; the Phase 2 `hallucination_probability` has separate evidence-grounded semantics.

## Real training data contract

Input is one JSON object per generation event, with `event_id`, independent `group_id`, `dataset_id`, `label_source: "human"`, exact `response`, an exact `trace` from `local_transformers_generate_raw_logits` (schema `hg-token-logits-v2`), and `claims` with `text`, `start`, `end`, `label`. Labels are `SUPPORTED` (0) or `HALLUCINATION_RELATED` (1), meaning the human annotation targets a hallucination-related event, **not** a calibrated probability of objective falsehood. Declare `label_unit: "sentence"` only when human labels cover every original sentence exactly and mark a sentence positive if any factual assertion in it is hallucination-related. This is the only unit eligible for the current sentence-level fast path. `label_unit: "atomic_claim"` supports research training but cannot authorize bypass until atomic-claim inference is integrated. See `GenerationTrace` in `halluciguard_detector/phase1.py` for token fields and pinned identity. A supplied `label_source` string is an audit declaration, not independent proof of annotator quality. Obtain independent human labels, clear guidelines, adjudication, and representative sampling before training.

The preparation command rejects mismatched response hashes, offsets, mixed generator revisions, duplicate generations, and non-human label declarations. Groups are split into train, validation, calibration, and final test, with group non-overlap. Training selects a checkpoint on validation PR-AUC, fits temperature only on calibration claims, then selects any bypass cutoff on validation **responses**. The conservative policy counts a validation group as a failure if any bypassed response in it is positive, and uses a simultaneous Hoeffding upper bound with a minimum number of independent bypassed groups. If no cutoff meets the constraint, `release_validated=false`, and verification remains mandatory. The final test is not used for model or cutoff selection. This statistical bound still assumes representative, independent validation groups and does not guarantee safety under drift.

To collect labels, create a JSONL prompt file with unique `prompt_id` and `query` fields (and optional `group_id`). The `collect` command generates each answer once with a pinned local model and persists the exact response and raw-logit trace. It resumes by skipping completed prompt IDs. The application does not silently log raw responses or traces. Generate an annotation queue, then have human reviewers check every sentence against independent evidence, label it `SUPPORTED` or `HALLUCINATION_RELATED`, and set `label_source` to `human`. For atomic-claim research, reviewers may split sentences and change `label_unit` to `atomic_claim`; such a head cannot enable the current fast path. The template itself is deliberately **unlabelled** and cannot pass training validation. Protect the source and annotated files as potentially sensitive data.

```powershell
python -m halluciguard_detector.phase1_cli collect --input data/phase1-prompts.jsonl --output data/phase1-generation-events.jsonl --model-id Qwen/Qwen2.5-0.5B-Instruct --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 --tokenizer-id Qwen/Qwen2.5-0.5B-Instruct --tokenizer-revision 7ae557604adf67be50417f59c2c2f167def9a775 --dataset-id human-study-v1
python -m halluciguard_detector.phase1_cli annotation-template --input data/phase1-generation-events.jsonl --output data/phase1-annotation-queue.jsonl
# After independent human review, save data/phase1-human-generations.jsonl.
python -m halluciguard_detector.phase1_cli prepare --input data/phase1-human-generations.jsonl --output-dir data/phase1-prepared
python -m halluciguard_detector.phase1_cli preflight --data-dir data/phase1-prepared --output-dir artifacts/phase1-uq
python -m halluciguard_detector.phase1_cli train --data-dir data/phase1-prepared --output-dir artifacts/phase1-uq --epochs 100 --max-false-accept 0.01 --min-bypass 100
python -m halluciguard_detector.phase1_cli evaluate --data-dir data/phase1-prepared --head-dir artifacts/phase1-uq
```

Evaluate once after model and policy freeze. `test_report.json` includes claim-level PR-AUC, ROC-AUC, Brier and ECE, plus response-level ranking and routing counts when release is validated. Compare those results with Phase 2 and Verifier on the **same request distribution** before enabling any bypass. The head uses `torch`, `numpy`, `scikit-learn` (training/evaluation), and `safetensors`; the optional local generator uses `transformers`. Ordinary remote-provider startup does not load the local model or head.

Run tests with `python -m pytest -q halluciguard_detector/tests/test_phase1.py orchestration/tests/test_detector_bridge.py orchestration/tests/test_graph_contract.py`. Disable Phase 1 by setting `HALLUCIGUARD_PHASE1_MODE=disabled` and keep `ALWAYS_VERIFY=true`.
