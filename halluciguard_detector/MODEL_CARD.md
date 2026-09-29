# HalluciGuard Detector — Model Card

## Intended use

Triage factual claims in an LLM draft against evidence supplied by a retrieval service. The answer is first decomposed into atomic claims (shared Verifier `ClaimDecomposer`); evidence is selected per claim with the shared hybrid retriever and cross-encoder reranker (reranking on the real claim), then each claim is classified as `SUPPORTED`, `CONTRADICTED`, or `NOT_ENOUGH_INFO`. Output classes and per-claim probabilities keep these three classes separate. The detector is not an open-world truth oracle and is not the final HalluciGuard Judge.

## Training

- Base encoder: `microsoft/deberta-v3-xsmall`
- Detector initialization: fresh classification head and fresh fine-tuning run
- Data: RAGTruth human span annotations converted to claim-level examples
- Train: 29,832 examples after majority-class downsampling
- Dev: 10,873 natural-distribution examples, grouped away from train by source ID
- Test: 18,777 examples from the untouched official test split
- Run: 1 epoch, seed 42, batch size 16, maximum length 256, AdamW
- Sequence length: **256**, defined once in `calibration.DEFAULT_MAX_LENGTH` and
  cross-checked against `tokenizer_config.json` by `python -m halluciguard_detector.model_card`

Calibration and thresholds, all fitted on **dev** and applied unchanged to test:

| Parameter | Value | Question it answers |
|---|---:|---|
| Temperature | 0.7586 | how confident the probabilities are |
| `contradiction_threshold` | 0.460 | **Task A** -- is the claim refuted? |
| `verification_risk_threshold` | 0.625 | **Task B** -- does the claim still need checking? |
| `hallucination_threshold` | 0.625 | *deprecated* mirror of Task B; do not use for refutation |

The two thresholds answer different questions and are deliberately different
numbers. `hallucination_threshold` is retained only so existing readers keep
their previous value instead of silently picking up a different one.

## Held-out test results

These metrics cover the learned model only. They do not include the runtime
named-entity, numeric or year guards, and the threshold for each task is the
dev-fitted value above.

Test class distribution: SUPPORTED 17,258 (91.91%), CONTRADICTED 592 (3.15%),
NOT_ENOUGH_INFO 927 (4.94%). Both tasks are heavily imbalanced, which is why
accuracy alone is not reported as a headline.

### Task A -- contradiction ("is the claim refuted?")

Positive class `CONTRADICTED`, score `P(CONTRADICTED)`, positive rate 3.15%.

| Metric | Value |
|---|---:|
| Precision | 0.2449 |
| Recall | 0.5236 |
| F1 | 0.3337 |
| ROC-AUC | 0.8500 |
| PR-AUC | 0.2311 |
| Accuracy | 0.9341 |
| ECE | 0.0513 |
| Brier | 0.0409 |
| False-positive rate | 0.0526 |
| False-negative rate | 0.4764 |

Confusion counts: TN 17,229; FP 956; FN 282; TP 310.

### Task B -- verification needed ("does the claim still need checking?")

Positive class `CONTRADICTED + NOT_ENOUGH_INFO`, score
`P(CONTRADICTED) + P(NOT_ENOUGH_INFO)`, positive rate 8.09%.

| Metric | Value |
|---|---:|
| Precision | 0.3096 |
| Recall | 0.5293 |
| F1 | 0.3907 |
| ROC-AUC | 0.8212 |
| PR-AUC | 0.2886 |
| Accuracy | 0.8664 |
| ECE | 0.1296 |
| Brier | 0.1083 |
| False-positive rate | 0.1039 |
| False-negative rate | 0.4707 |

Confusion counts: TN 15,465; FP 1,793; FN 715; TP 804.

### Three-class argmax

| Metric | Value |
|---|---:|
| Accuracy | 0.8383 |
| Macro-F1 | 0.5072 |
| Multiclass Brier | 0.0704 |
| Multiclass ECE | 0.0864 |

Per-class recall: SUPPORTED 0.8718 (n=17,258), CONTRADICTED 0.5490 (n=592),
NOT_ENOUGH_INFO 0.3991 (n=927).

### Honest reading of these numbers

- **86% and 93% accuracy are not the story.** They are dominated by the 92%
  SUPPORTED majority. Macro-F1 (0.51) and minority recall are the honest
  summaries.
- **The model misses about half of all refutations** (Task A recall 0.52) and
  **60% of all insufficient-evidence cases**. Recall is the weak axis, not
  precision: the model under-fires rather than over-accuses.
- **The model is well-ranked but poorly calibrated for Task B** (ROC-AUC 0.82
  against ECE 0.13). Its ordering is useful; its raw scores are not.
- A curated runtime probe set of 10 hand-written cases scored **3/8 (37.5%)**
  exact-label accuracy. All five errors predicted `SUPPORTED` for claims that
  were contradicted or unverified -- the model fails in the conservative
  direction and never produced a false accusation. Run
  `python -m halluciguard_detector.benchmark` to reproduce; the number is a
  measurement and is deliberately not asserted in the test suite.
- **These metrics are not sufficient for autonomous final decisions.** Flagged
  claims must go to the Verifier, and the Judge decides from claim counts and
  independent Verifier verdicts.

## Runtime guards (secondary signals)

These guards never overwrite the model's probabilities with hard-coded values. They are bounded and may only resolve a **near-tie**: when the model is torn between supported and contradicted below the decision threshold, a confirmed conflict can reorder the decision; a decisive model call is only nudged.

- **Named-entity conflict:** fires only when claim and evidence share a named anchor and each names a different additional named entity. A *direct* contradiction — the same relation with a swapped entity ("Apple acquired Company A" vs "Apple acquired Company B") — reinforces contradiction and adds a warning. A *mere mismatch* with a different relation ("Apple works with Company A" vs "Apple acquired Company B") is only reported as a warning and is never treated as a contradiction. Catches obvious substitutions such as "Java was created by Snehith" versus evidence naming James Gosling, without assuming contradiction whenever an entity merely differs or is absent.
- **Numeric/date/percent consistency:** a same-slot quantity clash ("population is 10 million" vs "12 million") or a conflicting year ("founded in 2010" vs "2012") reinforces contradiction so a topic-similar but numerically-wrong claim is not left supported by soft NLI alone, and always emits a warning for the Judge.
- **Unknown is protected:** when the model's dominant class is already `NOT_ENOUGH_INFO`, no guard touches the probabilities. Genuine uncertainty is never converted into contradiction mass.
- **Non-factual content:** non-factual/opinion claims are filtered by the shared decomposer's checkable-content rule before NLI; they are marked `non_factual`, excluded from claim counts, and never become hallucination.

These rules have unit coverage but are not included in the benchmark figures above.

## Known train/runtime evidence mismatch

Training pairs the classifier with up to six concatenated lexical snippets
(`data.py` joins them), while the runtime pairs it with a **single** reranked
snippet. The same claim can therefore be classified on differently shaped
evidence at training and at inference.

Measured with the real checkpoint over 9 probes and a 12-document corpus
(`python -m halluciguard_detector.evidence_alignment`):

| Evidence strategy | Label agreement with production | Mean abs. verification-risk delta | Max delta |
|---|---:|---:|---:|
| `training_lexical_joined` (what training uses) | 0.6667 | 0.2371 | 0.6090 |
| `lexical_top1` | 0.6667 | 0.1456 | 0.5531 |
| `hybrid_top1` (no rerank) | 0.6667 | 0.1564 | 0.5531 |
| `hybrid_rerank` (production) | 1.0000 | 0.0000 | 0.0000 |

The training-time evidence shape disagrees with production on **one third** of
probes, with a mean verification-risk shift of **0.237** -- large enough to move
a decision across the 0.625 threshold. Aligning training evidence with the
production single-snippet shape is the highest-value follow-up, and is expected
to improve more than any threshold tweak. This is a small probe set, not a
benchmark: it quantifies a mechanism, it does not estimate accuracy.

Note the dense leg of the hybrid retriever was unavailable offline during this
measurement, so the hybrid rows are BM25 + reranker, not true hybrid fusion.

## Output semantics

- `contradicted_probability` / `contradiction_mass` are the only signals about falsehood.
- `verification_risk` = `P(CONTRADICTED) + P(NOT_ENOUGH_INFO)` is an **operational triage score** for "does this claim still need checking", not a probability that the claim is false.
- `hallucination_probability` (claim) and `probability` (answer) are **deprecated aliases** of `verification_risk`, kept for backward compatibility. New consumers should read `verification_risk`.
- The answer-level `HALLUCINATION` label is driven by contradiction only and follows the claim-level `CONTRADICTED` decisions, so an all-`NOT_ENOUGH_INFO` answer is never reported as hallucinated.

## Limitations

- Recall and F1 are not sufficient for autonomous final decisions. Always send flagged claims to the Verifier; the Judge decides from the per-class claim counts and independent Verifier verdicts.
- `NOT_ENOUGH_INFO` is a first-class class: it means the supplied evidence is insufficient, not that the claim is false, and it is never folded into the contradiction rate.
- RAGTruth is English and RAG-oriented; other languages and domains require separate evaluation.
- Claim/span boundaries are deterministic and can be imperfect for abbreviations or malformed model output.
- Evidence selection degrades to deterministic lexical retrieval when the shared Verifier stack or its models are unavailable (e.g. offline dense encoder). This is reported rather than hidden: the claim and answer set `evidence_degraded=true` with `evidence_route` recording which path ran, so a consumer can weight the verdict accordingly.
- The entity guard does not resolve aliases or coreference and may miss non-entity contradictions; the numeric checks are deliberately conservative and can miss complex quantitative reasoning.
- High-stakes medical, legal, financial, and safety uses require domain data and human review.
- **Unmeasured at full scale.** All figures here come from the saved dev/test
  logits that ship in this checkpoint. The raw RAGTruth source is not in this
  repository, so the conversion, the downsampling and the evidence shapes could
  not be re-derived or re-trained here, and no number in this card was
  estimated, extrapolated or copied from documentation.
- **Label-conversion assumption.** RAGTruth annotates *spans*, not the atomic
  claims the runtime decomposer produces, so training examples are a closer but
  not identical match to runtime claims. `stats.json` records the
  sentence-level versus span-level split so the size of that gap stays visible.
  Unrecognised `label_type` values fall back to `NOT_ENOUGH_INFO` and are
  counted as `unknown_label_type:*` so vocabulary drift cannot pass unnoticed.
