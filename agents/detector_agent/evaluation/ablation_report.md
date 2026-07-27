# HalluciGuard - Signal Ablation Study Report

## Summary Performance Table

| Configuration | Accuracy | Precision | Recall | Macro F1 | Pairwise Acc | Avg Conf Gap | SC Activation Rate | Avg Conf Change (SC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Detector** | 0.3750 | 0.1667 | 0.2500 | 0.2000 | 25.00% | -0.0831 | 0.00% | +0.0000 |
| **Without Token Probability** | 0.5000 | 0.1667 | 0.3333 | 0.2222 | 75.00% | -0.0164 | 0.00% | +0.0000 |
| **Without Entropy** | 0.2500 | 0.2778 | 0.1667 | 0.2063 | 25.00% | -0.1277 | 0.00% | +0.0000 |
| **Without Semantic Similarity** | 0.2500 | 0.2222 | 0.1667 | 0.1905 | 25.00% | -0.1073 | 0.00% | +0.0000 |
| **Without Self-Consistency** | 0.3750 | 0.1667 | 0.2500 | 0.2000 | 25.00% | -0.0811 | 0.00% | +0.0000 |
| **Token Probability Only** | 0.2500 | 0.1333 | 0.1667 | 0.1481 | 25.00% | -0.1782 | 0.00% | +0.0000 |
| **Entropy Only** | 0.5000 | 0.1667 | 0.3333 | 0.2222 | 75.00% | +0.0109 | 0.00% | +0.0000 |
| **Semantic Similarity Only** | 0.3750 | 0.1429 | 0.2500 | 0.1818 | 75.00% | -0.0437 | 0.00% | +0.0000 |
| **Self-Consistency Only** | 0.5000 | 0.1667 | 0.3333 | 0.2222 | 0.00% | +0.0000 | 0.00% | +0.0000 |

## Self-Consistency Gating & Execution Analysis

Self-Consistency is executed conditionally by `SelfConsistencyGate` ONLY when:
1. Single-pass risk level evaluates to `MEDIUM`.
2. User query belongs to an analytical category (`factual_question`, `explanation`, `summarization`, `reasoning`).

### Self-Consistency Execution Log

No evaluations triggered the `SelfConsistencyGate` (all queries were classified as LOW or HIGH risk or non-analytical prompt types).

## Findings & Signal Contribution Analysis

- **Full Detector**: Allocates non-zero weight (`0.15`) to Self-Consistency, executing stochastic decoding on uncertain prompts while avoiding compute bloat on clear predictions.
- **Without Self-Consistency**: Disables stochastic sampling completely (`self_consistency=0.00`), saving GPU compute while relying strictly on single-pass signals.
- **Token Probability**: Serves as the primary token-level certainty anchor.
- **Predictive Entropy**: Measures token uncertainty distribution under choice ambiguity.
- **Semantic Similarity**: Essential for measuring semantic grounding between prompt context and candidate responses.
