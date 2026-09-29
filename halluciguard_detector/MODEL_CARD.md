# HalluciGuard Detector — Model Card

## Intended use

Triage factual claims in an LLM draft against evidence supplied by a retrieval service. The answer is first decomposed into atomic claims (shared Verifier `ClaimDecomposer`); evidence is selected per claim with the shared hybrid retriever and cross-encoder reranker (reranking on the real claim), then each claim is classified as `SUPPORTED`, `CONTRADICTED`, or `NOT_ENOUGH_INFO`. Output classes and per-claim probabilities keep these three classes separate. The detector is not an open-world truth oracle and is not the final HalluciGuard Judge.

## Training

- Base encoder: `microsoft/deberta-v3-xsmall`
- Detector initialization: fresh classification head and fresh fine-tuning run
- Data: RAGTruth human span annotations converted to sentence-level labels
- Train: 29,832 examples after majority-class downsampling
- Dev: 10,873 natural-distribution examples, grouped away from train by source ID
- Test: 18,777 examples from the untouched official test split
- Run: 1 epoch, seed 42, batch size 16, maximum length 256, AdamW
- Calibration: temperature scaling on dev (`T=0.7586`) and dev-selected threshold (`0.6216`)

## Held-out test results

These metrics cover the learned model only. They do not include the runtime named-entity guard.

| Metric | Value |
|---|---:|
| Accuracy | 0.8655 |
| Precision | 0.3083 |
| Recall | 0.5332 |
| F1 | 0.3907 |
| ROC-AUC | 0.8212 |
| PR-AUC | 0.2886 |
| ECE | 0.1296 |
| False-positive rate | 0.1053 |
| False-negative rate | 0.4668 |

Confusion counts: TN 15,441; FP 1,817; FN 709; TP 810.

## Runtime guards (secondary signals)

These guards never overwrite the model's probabilities with hard-coded values; they reinforce or annotate.

- **Named-entity conflict:** fires only when claim and evidence share a named anchor and each names a different additional named entity. It re-normalizes contradiction confidence upward by a bounded amount (25% of the model's own support-confidence gap) and adds a warning. Catches obvious substitutions such as "Java was created by Snehith" versus evidence naming James Gosling, without assuming contradiction whenever an entity merely differs or is absent.
- **Numeric/date/percent consistency:** a conservative quantity-unit and year check emits a warning for mismatches (e.g. "released in 1995" vs evidence "released in 1996"); the mismatch is surfaced to the Judge as a secondary signal, not treated as an automatic contradiction.
- **Non-factual content:** non-factual/opinion claims are filtered by the shared decomposer's checkable-content rule before NLI; they are marked `non_factual`, excluded from claim counts, and never become hallucination.

These rules have unit coverage but are not included in the benchmark figures above.

## Limitations

- Recall and F1 are not sufficient for autonomous final decisions. Always send flagged claims to the Verifier; the Judge decides from the per-class claim counts and independent Verifier verdicts.
- `NOT_ENOUGH_INFO` is a first-class class: it means the supplied evidence is insufficient, not that the claim is false, and it is never folded into the contradiction rate.
- RAGTruth is English and RAG-oriented; other languages and domains require separate evaluation.
- Claim/span boundaries are deterministic and can be imperfect for abbreviations or malformed model output.
- Evidence selection degrades to deterministic lexical retrieval when the shared Verifier stack or its models are unavailable (e.g. offline dense encoder).
- The entity guard does not resolve aliases or coreference and may miss non-entity contradictions; the numeric checks are deliberately conservative and can miss complex quantitative reasoning.
- High-stakes medical, legal, financial, and safety uses require domain data and human review.
