# HalluciGuard Detector — Model Card

This card reports **Phase 2 evidence-grounded DeBERTa** only. The optional Phase 1 logit-feature uncertainty head has no trained checkpoint or measured performance yet; its distinct contract and limitations are in [Detector Phase 1](../docs/detector-phase1.md). Do not apply the Phase 2 test metrics below to Phase 1.

## Intended use

Triage factual sentences in an LLM draft against evidence supplied by a retrieval service. Output classes are `SUPPORTED`, `CONTRADICTED`, and `NOT_ENOUGH_INFO`. The detector is not an open-world truth oracle and is not the final HalluciGuard Judge.

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

## Runtime guard

A named-entity mismatch guard raises the verification risk when claim and evidence share a named anchor but contain different additional names. It does **not** change the model's class label or calibrated probabilities and is not itself a contradiction verdict. The response includes a warning whenever this guard fires. This guard has unit coverage but is not included in the historical benchmark figures above.

Structured JSON-like evidence is normalized to bounded, field-labelled text before lexical selection and model inference. The trace separately reports normalization losses, selected snippets, and tokenizer pair lengths when inspectable; it does not reconstruct the exact retained evidence tokens. This Detector selection path is lexical-only. The upstream Verifier may separately run BM25, BGE-m3 dense retrieval, and reranking, with execution/fallback diagnostics in its retrieval trace. The historical benchmark above has not been rerun with this normalization.

## Limitations

- Recall and F1 are not sufficient for autonomous final decisions. Always send flagged claims to the Verifier.
- `NOT_ENOUGH_INFO` means the supplied evidence is insufficient, not that the claim is false.
- RAGTruth is English and RAG-oriented; other languages and domains require separate evaluation.
- Sentence boundaries are deterministic and can be imperfect for abbreviations or malformed model output.
- The entity guard does not resolve aliases or coreference and may miss non-entity contradictions.
- High-stakes medical, legal, financial, and safety uses require domain data and human review.
