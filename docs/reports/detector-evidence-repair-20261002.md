# Detector evidence repair and real model experiments — 2026-10-02

Base: `f8900ca66721971b515674a10e2a17d8d8437cc0` (origin/main).
Branch: `agents/detector-evidence-repair-20261002`.
Production Detector weights, calibration and verification-first defaults are unchanged.

## Engineering repairs

- `orchestration.graph._dump` recursively serializes enum values inside Pydantic models and dataclasses. Canonical evidence conversion recognizes provider labels without treating enum names as neutral. Raw NLI triplets and relation/provenance metadata are retained as optional fields.
- `JudgeAgent.evaluate` no longer accepts neutral-only evidence merely because aggregate support exceeds contradiction. A decisive claim requires a nonempty citation classified in the verdict's direction. This does not remove the existing Verifier relation-rule classifier.
- ModelManager enforces the configured offline policy on reranker weights and on both tokenizer and weights for its HF NLI loader. CPU retries reuse the loaded NLI objects. Failed initialization does not create a healthy cache entry. Retry logs classify errors without printing their messages.
- QueryExpander adds object-independent retrieval for inverse capital statements. The original claim remains unchanged for NLI; this is not a truth rule.
- The direct tester distinguishes operational verification risk from contradiction probability and shows evidence labels separately from P(entailment).
- Training optionally supports source-disjoint calibration, macro-F1 checkpoint selection, local-only initialization and gradient checkpointing. Defaults retain prior behavior. Invalid calibration groups/labels fail before expensive loading. Existing artifact protections remain in force.

## Validation

Clean baseline and final candidate used:
`python -m pytest -q --tb=short --disable-warnings --show-capture=no -rs`

- Baseline: exit 0; 1,127 passed, 14 skipped, 1 warning; 547.90 seconds.
- Final candidate: exit 0; 1,146 passed, 14 skipped, 1 warning; 649.00 seconds.
- An intermediate full run was interrupted after the additional retrieval repair; it is NOT a passing validation.
- Evidence serialization, offline loader, Judge and full-audit safety group: 67 passed (72.97 seconds).
- Calibration, pilot preparation and existing data-integrity group: 22 passed (18.09 seconds).
- Inverse capital retrieval group: 5 passed (45.92 seconds).
- `python -m compileall -q agents/judge_agent agents/verifier_agent halluciguard_detector orchestration scripts/retrain_detector_pilot.py scripts/compare_minicheck_diagnostic.py test_direct.py`: exit 0.
- `git diff --check`: exit 0.

The 14 skips are two opt-in n8n tests, two opt-in live web tests, four legacy removed-Detector tests, and six tests requiring missing Corrector curriculum data. Passing unit/contract tests do not establish factual detection accuracy.

## Real RAGTruth fine-tuning pilot (not promoted)

Runner: `python -u -m scripts.retrain_detector_pilot --data <existing-processed-data> --output artifacts/retraining-pilot-20261002-run1 --baseline <existing-detector-best> --epochs 2`.
HF offline environment flags were enabled. Exit 0; training and both evaluations took 920.50 seconds.

Initialized a fresh three-class head on the cached pretrained `microsoft/deberta-v3-xsmall` encoder. This is fine-tuning, NOT training the encoder from random weights. Two epochs, seed 42, batch 8, length 256, learning rate 2e-5, mixed precision and gradient checkpointing on RTX 3050 Laptop GPU.

Used 6,000 genuine training rows (2,000 per class), 1,200 validation rows and 1,200 independently grouped calibration rows. All 18,777 test rows were unchanged and excluded from selection/calibration: 17,258 SUPPORTED, 592 CONTRADICTED, 927 NOT_ENOUGH_INFO. Source groups do not overlap.

Same-test results (existing checkpoint versus pilot):
- Three-class accuracy: 0.838313 versus 0.597593.
- Macro-F1: 0.507246 versus 0.363988.
- Thresholded contradiction precision: 0.251043 versus 0.173883.
- Thresholded contradiction recall: 0.508446 versus 0.427365.
- Thresholded contradiction F1: 0.336125 versus 0.247191.

Thresholded results use each checkpoint's stored calibration, not one common threshold. Baseline evaluation uses the original checkout's checkpoint/calibration; this does not assert that every current-main calibration copy is identical. The pilot's temperature was fitted only on its independent calibration split. Legacy threshold-report key `dev_f1` denotes calibration F1 when `selected_on=calibration`.

Pilot rejected. No performance improvement demonstrated. The prepared span dataset and joined evidence are not guaranteed atomic claims or production top-one evidence; class-balanced sampling causes a substantial prevalence mismatch. No safe bypass experiment was established. Artifacts remain in a separate ignored directory; original metrics/weights were not overwritten.

## Alternative detector investigation

Used the official cached `lytang/MiniCheck-DeBERTa-v3-Large` snapshot `2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb`. Runner:
`python -u -m scripts.compare_minicheck_diagnostic --checkpoint <cached-snapshot> --output artifacts/minicheck-diagnostic-20261002-run1/results.json --device cpu`.

Exit 0; real CPU inference correctly classified five of six constructed diagnostic cases. The supported Paul Allen co-founder example was missed (raw support score 0.451679). This is NOT a held-out benchmark. A prior incomplete cached snapshot failed tokenizer initialization before inference and produced no result.

The runner matches the upstream encoder's EOS-joined short document/claim format; it rejects long documents rather than claiming upstream long-document aggregation. MiniCheck is binary supported/unsupported and cannot independently distinguish contradiction from insufficient information. Its scores are not calibrated for HalluciGuard. No production substitution was made.

Primary references: [MiniCheck implementation](https://github.com/Liyan06/MiniCheck), [model card](https://huggingface.co/lytang/MiniCheck-DeBERTa-v3-Large), [RAGTruth](https://github.com/ParticleMedia/RAGTruth), [RefChecker](https://github.com/amazon-science/RefChecker), [SelfCheckGPT](https://github.com/potsawee/selfcheckgpt).

## Live pipeline checks and release limits

Loaded existing application environment without copying credentials; HF model downloads were disabled. Real Groq, retrieval and cached model inference were used.
- Vietnam capital query: final Hanoi answer accepted; Memory stored one verified fact; 143.12 seconds, exit 0.
- Supplied wrong Bangkok answer: insufficient decision-grade evidence, human-review escalation, answer withheld, no facts stored; 119.24 seconds, exit 0.
- Same wrong answer after inverse query repair: still insufficient decision-grade evidence, safely withheld; 90.05 seconds, exit 0. Therefore the live correction/ReVerifier path was NOT validated by these runs.

Main release remains withheld. Retrieval coverage needs further trace-level investigation; relation-rule score floors are existing heuristics, not newly measured calibration. Legitimate zero counts, absent probabilities and uncertainty must remain visible. These repairs do not establish that all project defects are resolved, that the Detector is production-ready, or that low-risk bypass is safe. Frontend and deployment remain deferred.
