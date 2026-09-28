# Training, evaluation and demonstrations

This document separates model evaluation, automated tests and live E2E demonstrations. They measure different things and must not be combined into one accuracy claim.

## Detector training artifact

Source: `halluciguard_detector/MODEL_CARD.md`, `artifacts/detector-best/training_report.json`, calibration and test metric files.

| Property | Recorded value |
|---|---|
| Base encoder | `microsoft/deberta-v3-xsmall` |
| Data | RAGTruth human span annotations converted to sentence labels |
| Labels | `SUPPORTED`, `CONTRADICTED`, `NOT_ENOUGH_INFO` |
| Train / dev / test | 29,832 / 10,873 / 18,777 examples |
| Split policy | Train/dev grouped by source; official test split untouched |
| Artifact run | 1 epoch, batch size 16, maximum length 256, seed 42 |
| Optimizer implementation | AdamW; training code default learning rate `2e-5`, weight decay `0.01` |
| Scheduling | Linear schedule with 6% warmup |
| Gradient clipping | `1.0` |
| Precision/device | CUDA recorded; AMP enabled by code on CUDA; exact GPU model not recorded |
| Calibration | Dev-fitted temperature `0.7586129308`; threshold `0.6216216216` |

The artifact report does not record whether every training-function default was overridden beyond values stated in the model card. Missing hardware details are intentionally not inferred.

## Held-out detector evaluation

| Metric | Value |
|---|---:|
| Accuracy | 0.8654737 |
| Precision | 0.3083365 |
| Recall | 0.5332456 |
| F1 | 0.3907381 |
| ROC-AUC | 0.8211981 |
| PR-AUC | 0.2886405 |
| ECE | 0.1295564 |
| False-positive rate | 0.1052845 |
| False-negative rate | 0.4667544 |

Confusion matrix: TN 15,441; FP 1,817; FN 709; TP 810. These figures exclude the runtime entity-conflict guard and do not establish open-world accuracy.

## Live E2E demonstrations — September 25, 2026

| Query | Grounded outcome |
|---|---|
| `What is the capital of India?` | One claim verified; Judge accepted; duplicate memory reuse observed |
| False Microsoft premise naming Snehith | Base LLM corrected the premise; two claims became conflicted, one verified; system withheld output for human review |
| `Who created the Java programming language?` | Core James Gosling/Sun claim verified; peripheral details unverified; Judge accepted core answer; one fact stored |
| `Who founded Microsoft?` after Detector integration | Two claims verified; trained checkpoint loaded and executed; calibrated probability `0.5596`; Judge accepted; two facts stored in that run |

These are live examples under the available providers and sources, not a fixed benchmark.

## Automated validation

The grounded Detector integration change was validated with 44 focused orchestration/contract tests passing. The exact suites were Detector bridge, grounded Detector node, graph execution, graph contract and canonical orchestration tests.

## Known limitations

- Live results can change with provider output, web content, rate limits and model versions.
- Detector recall/F1 remain insufficient for autonomous final decisions.
- Some integration tests require credentials and network access.
- Corrector training artifacts and benchmark reports exist, but claims should be taken only from their explicit recorded files and not generalized to the complete product.
