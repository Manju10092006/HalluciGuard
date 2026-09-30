# Detector v2 — evidence alignment investigation

**Branch:** `detector-v2-evidence-alignment`
**Base:** `573f778` (merged PR #53)
**Status of conclusions:** every number below is measured from artifacts in this
repository or from reports/ JSON produced by the commands in
[§8 Reproduction](#8-reproduction). Where a result is preliminary it is labelled
as such. No claim in this report is carried over from a previous conversation.

---

## 1. Summary

The question asked was whether the mismatch between the evidence the Detector
is **trained** on and the evidence it is **served** in production explains the
weak contradiction recall (F1 0.3337, precision 0.2449).

**The measurement does not support that explanation.** Three independent
measurements point the same way:

1. Serving the shipped checkpoint the *production-shaped* single snippet
   instead of the joined training blob **improves** contradiction F1 by
   4.3 points rather than degrading it ([§4](#4-experiment-1-shape-sensitivity)).
   Retraining under identical settings replicates the direction from a second
   design: contradiction recall +8.7 points, and **NOT_ENOUGH_INFO F1 goes from
   0.0000 to 0.2703** ([§10](#10-experiment-4--retrained-ab)). The mismatch is
   real — it changes the prediction on 17% of claims — but it is not the cause
   of the contradiction gap, and the arm the released model was trained on
   cannot predict NOT_ENOUGH_INFO at all.
2. On a hand-written probe of clean, unambiguous contradictions, the shipped
   model detects **0 of 5**, assigning P(CONTRADICTED) between 0.022 and 0.121
   against a decision threshold of 0.46 ([§5](#5-experiment-2-hard-negatives)).
   Nothing about evidence shape can explain a failure on evidence that is
   already correct, single, and on-topic.
3. Categorising all 118 contradiction false negatives shows the largest single
   group is **unsegmented structured sources** — rows where the "evidence
   sentence" is a raw JSON record because the source is a Yelp-style document
   the sentence splitter cannot handle. Genuine retrieval misses are 4% of the
   total ([§6](#6-experiment-3-false-negative-analysis)).

Evidence alignment is still worth doing, and the reason is not the one the
question assumed. It is a **NOT_ENOUGH_INFO** fix: the joined training shape
scores 0.0000 on that class, and production-shaped training restores it while
also raising contradiction recall ([§7](#7-what-to-do-next)). Separately, the
single highest-value change is **not** an alignment change at all — it is
handling structured sources in dataset preparation, which is where ~45% of
contradiction false negatives come from.

---

## 2. What the pipeline actually does

Traced, not assumed. `halluciguard_detector/nli_input.py` now encodes this as a
single importable contract so the two sides cannot drift again.

| Property | Value | Source |
| --- | --- | --- |
| Evidence snippets per pair | **1** | `detector.py` scores `snippets[0]` only |
| Pair order | `evidence`, `claim` | `nli_input.prepare_nli_input` |
| `Claim:` / `Evidence:` markers | **none** | `CLAIM_PREFIX`/`EVIDENCE_PREFIX` are `None` |
| Separator | **none** | `contract_spec()["separator"] is None` |
| Truncation | `longest_first` | `TRUNCATION_STRATEGY` |
| Max sequence length | **256** | shipped `calibration.json` |
| Char-level pre-truncation | **none** | only the tokenizer's `max_length` applies |
| Production pool width | 20 | `evidence.DEFAULT_POOL_K` |
| Production rerank depth | 3 | `evidence.DEFAULT_RERANK_TOP` |

`detector.py` and the training collator both call `encode_nli_pair`, so
training, dev, test and runtime share one definition of the input.

### 2.1 The training side (before this work)

`data.py` called `lexical_evidence(span.text, [source_info])` and joined up to
six selected sentences with `" "`. Measured on the prepared training split:

- mean evidence **276.9 tokens**, mean claim **18.1 tokens**
- **41.4%** of pairs exceed the 256-token limit (dev 39.5%, test 35.3%)

With `longest_first`, the evidence side is the longer side, so the tail of the
joined blob — often the sentence carrying the actual conflict — is what gets
cut. This is a real defect and it is now prevented structurally:
`enforce_single_snippet` raises on a multi-snippet list rather than joining.

---

## 3. Baseline

From the shipped artifact `artifacts/detector-best/test_metrics.json`
(18777 samples: 17258 SUPPORTED / 592 CONTRADICTED / 927 NOT_ENOUGH_INFO),
verified by reading the file and matching the numbers quoted in the brief:

| Task | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Contradiction (Task A) | 0.2449 | 0.5236 | **0.3337** | 0.2311 | 0.8500 |
| Verification needed (Task B) | 0.3096 | 0.5293 | **0.3907** | 0.2886 | 0.8212 |

Three-class macro-F1 **0.5072**, accuracy **0.8383**, confusion matrix
(rows = truth, columns = predicted):

```
[[15046,  954, 1258],
 [  211,  325,   56],
 [  427,  130,  370]]
```

Shipped calibration, unchanged by this work: temperature `0.7586129308`,
contradiction threshold `0.46`, verification-risk threshold `0.625`,
max length `256`.

### 3.1 Two caveats about the baseline

**The artifact and the local data disagree on size.** The artifact was built on
18777 samples; the local prepared test split has **17935** rows
(16383 / 628 / 924). Re-derivation therefore cannot be expected to reproduce
the artifact numbers exactly, and `experiments.py baseline` reports the
disagreement rather than resolving it silently. The artifact numbers are quoted
above because they are what was shipped; all *comparisons* in this report are
paired and within-split, so the discrepancy does not affect them.

**Naive subsampling destroys the signal.** Re-scoring the first 600 rows of the
test split yields contradiction F1 **0.0909** on 21 contradictions, against
0.3337 on the full split. Any reduced run that takes a prefix rather than
balancing classes is not measuring the contradiction axis at all. This is why
the diagnostic split below is class-capped.

---

## 4. Experiment 1 — shape sensitivity

**Design.** One checkpoint (`artifacts/detector-best`), identical claims,
identical labels, identical operating point; only the evidence string changes.
No retraining, so the effect is causal and attributable to evidence alone.
450 claims, 150 per class, drawn deterministically from the natural test pool.

| Arm | Contradiction R | Contradiction P | Contradiction F1 | NEI F1 | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| `lexical_joined` (training shape) | 0.1933 | 0.6744 | 0.3005 | 0.4147 | **0.4345** |
| `lexical_top1` (single snippet) | **0.2333** | 0.6481 | **0.3431** | 0.3439 | 0.3968 |

Paired movement:

- Contradiction recall **+0.0400**, F1 **+0.0426**
- NOT_ENOUGH_INFO F1 **−0.0708**
- Macro-F1 **−0.0377**
- Label agreement **82.89%** — 77 of 450 claims change prediction
  (CONTRADICTED 23/150, NOT_ENOUGH_INFO 36/150, SUPPORTED 18/150)
- Absolute shift in P(CONTRADICTED): mean **0.125**, median 0.047, p90 0.358,
  max 0.809; **23.8%** of claims move by more than 0.20

**Interpretation.** Evidence shape is not a no-op — it changes the output on
one claim in six, and it moves probabilities a long way on some of them. But it
does not degrade the contradiction axis. It *trades* contradiction recall
against NOT_ENOUGH_INFO, because dropping the joined blob removes the context
that told the model when information was merely spread across several
sentences. The macro-F1 cost is real and lands entirely on the NEI class.

This falsifies the specific hypothesis that the shape mismatch is what is
holding contradiction recall down.

---

## 5. Experiment 2 — hard negatives

**Design.** 13 hand-written claim/evidence pairs through the real checkpoint
(`halluciguard_detector/hard_negatives.py`). Every case is reported, not just
the failures. Three of the five contradiction cases are deliberately *not*
contradictions, because a probe set containing only contradictions would reward
a model for saying CONTRADICTED to everything.

| Case | Expected | Predicted | P(CONTRADICTED) |
| --- | --- | --- | --- |
| direct_year — "founded in 1995" vs "founded in 2003" | CONTRADICTED | SUPPORTED | 0.029 |
| direct_number — "5 million users" vs "more than 50 million" | CONTRADICTED | SUPPORTED | 0.022 |
| direct_entity — "created by Snehith" vs "created by James Gosling" | CONTRADICTED | SUPPORTED | 0.035 |
| different_relation_same_entities — explicit denial of an acquisition | CONTRADICTED | NOT_ENOUGH_INFO | 0.121 |
| negation — "Safari runs on Windows" vs "macOS and iOS, not Windows" | CONTRADICTED | SUPPORTED | 0.051 |
| supported_exact | SUPPORTED | SUPPORTED | 0.017 |
| supported_paraphrase | SUPPORTED | SUPPORTED | 0.038 |
| different_year_event | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO | 0.209 |
| different_relation | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO | 0.074 |
| temporal_stale | NOT_ENOUGH_INFO | SUPPORTED | 0.150 |
| multi_hop | NOT_ENOUGH_INFO | SUPPORTED | 0.046 |
| retrieval_miss | NOT_ENOUGH_INFO | SUPPORTED | 0.064 |
| retrieval_miss_supported | SUPPORTED | SUPPORTED | 0.030 |

**Contradiction detection rate: 0 / 5.** Accuracy 5 / 13.

Each of the five is a single, on-topic, unambiguous sentence stating exactly the
conflicting fact. The failure cannot be attributed to retrieval, to evidence
length, or to shape. This is the clearest single result in the report: the
weak contradiction recall is primarily a **model capability** limit on
`deberta-v3-xsmall`, not a pipeline-alignment limit.

Caveat, stated because it matters: 13 cases cannot support a performance claim
and this accuracy is not comparable to the RAGTruth metrics. It is a diagnostic
for failure modes.

---

## 6. Experiment 3 — false-negative analysis

**Design.** All 118 contradiction false negatives on the 450-claim class-capped
split, categorised by first-match deterministic rules
(`halluciguard_detector/error_analysis.py`) that reuse the runtime's own
comparability guards. These are triage labels, not verified ground truth.

| Category | Count | Share |
| --- | --- | --- |
| `UNSEGMENTED_SOURCE` | 53 | 44.9% |
| `NLI_MODEL` | 24 | 20.3% |
| `NEGATION` | 15 | 12.7% |
| `RELATION` | 11 | 9.3% |
| `CLAIM_DECOMPOSITION` | 9 | 7.6% |
| `RETRIEVAL` | 5 | 4.2% |
| `TEMPORAL` | 1 | 0.8% |

Contradiction recall on this split: 0.2133 (32 / 150), with 15 false positives.

**`UNSEGMENTED_SOURCE` is the largest group by a wide margin.** RAGTruth's
Yelp-derived sources are JSON documents. Sentence segmentation cannot split
them, so the model is handed a raw record as its "evidence sentence" — the
first false negative analysed literally begins `{"address": "4421 Hollister
Ave", "attributes": {"Ambience": ...` against the claim "free Wi-Fi".

Measured on the **natural** test split, not the diagnostic one:

- **12.0%** of all rows have a structured-record evidence side
- **36.3%** of CONTRADICTED rows do, against 10.5% of SUPPORTED rows
- 22.5% of NOT_ENOUGH_INFO rows do

Contradictions are therefore ~3.5× more likely to land on a source the evidence
pipeline cannot read. This is a dataset-preparation defect, it is concentrated
exactly on the weak class, and it is the highest-value fix available.

**The defect survives the real production path.** 260 claims were generated
through `ClaimEvidenceEngine.select` with the cross-encoder reranker active
(§9). **36.9%** of the resulting production evidence is a raw structured
record. This is not an artefact of the lexical arms.

**`RETRIEVAL` is 4.2%.** Whatever the retriever is doing wrong, it is not what
is costing contradiction recall. This is the sharpest possible refutation of
the original hypothesis.

**A correction worth recording.** The first version of this categoriser
reported 110 / 118 as `RETRIEVAL`. That was wrong twice over: a Jaccard
threshold punished short claims like "Somali-Canadian" for being short, and a
`"key":` pattern count labelled plain prose reviews as unsegmented. The
categoriser was corrected to use a structural record test and non-generic
claim-term coverage, and the two tests that encode those mistakes are kept in
`tests/test_evidence_alignment.py` so they cannot come back.

---

## 7. What to do next

Ordered by measured impact, not by effort.

1. **Handle structured sources in dataset preparation.** ~45% of contradiction
   false negatives and 12% of all rows. Either extract the human-readable
   fields from Yelp-style records before segmentation, or drop them from
   training with an explicit, reported exclusion. Silently feeding a JSON blob
   as an evidence sentence is the largest single defect found.
2. **Attack negation and relation reasoning directly.** 27 false negatives
   (23%) sit in `NEGATION` and `RELATION`, and the hard-negative probe shows
   the model assigns ~0.05–0.12 to explicit denials. This is a training-data
   problem: the split needs hard negatives that a lexical overlap model cannot
   win. The taxonomy in `hard_negatives.py` is the starting point.
3. **Do the evidence alignment, and the reason is NOT_ENOUGH_INFO, not
   contradiction.** Two independent designs now agree that production-shaped
   single-snippet evidence does not hurt contradiction recall and helps it
   (§4, §10). The real cost of the current joined training shape is that
   `lexical_joined` scores **NEI F1 = 0.0000** on the natural test split — it
   never predicts NOT_ENOUGH_INFO — and retraining on production-shaped
   evidence restores it to 0.2703 while raising macro-F1 by 9 points. Since
   verification-needed routing is what the runtime actually asks, this is the
   change to make. Before shipping it, confirm the NEI gain holds on the full
   natural distribution, since §10 shows the two splits disagree on
   contradiction F1.
4. **Fix the observability gap, and do it before any further retrieval
   experiment.** The trace reports `route: "hybrid", degraded: false` on every
   row of a run in which the dense model demonstrably failed to load (§9). In
   the shipped default configuration `BAAI/bge-m3` is not present, so the
   "hybrid" path is not hybrid at all, and production evidence is BM25-only.
   Comparisons that assume otherwise will agree for the wrong reason.
5. **Consider a larger backbone.** With 0/5 on clean contradictions, the
   ceiling for `deberta-v3-xsmall` is close. This is a capacity result, and no
   amount of pipeline alignment will change it.

### 7.1 Explicitly not done

No thresholds were tuned, no random heuristics were added, no model was
replaced, and the detector architecture is unchanged — as instructed. The
`lexical_joined`, `lexical_top1` and `production_top1` arms are **evaluation
and data-generation constructs only**; no arm changes a runtime decision.

---

## 8. Reproduction

```powershell
# 1. the canonical input contract, and the shape-sensitivity result
python -m pytest halluciguard_detector/tests -q
python -m halluciguard_detector.experiments shape-sensitivity `
  --checkpoint artifacts/detector-best `
  --arm lexical_joined=artifacts/detector-v2/lexical_joined `
  --arm lexical_top1=artifacts/detector-v2/lexical_top1 `
  --split metric

# 2. the hard-negative probe
python -m halluciguard_detector.hard_negatives --checkpoint artifacts/detector-best

# 3. false-negative categorisation
python -m halluciguard_detector.error_analysis `
  --checkpoint artifacts/detector-best `
  --data-dir artifacts/detector-v2/lexical_joined --split metric

# 4. re-derive the baseline (full run takes ~85 min on CPU)
python -m halluciguard_detector.experiments baseline --split test
```

Generated reports:

| File | Contents |
| --- | --- |
| `reports/detector_v2_shape_sensitivity_prelim.json` | §4, two lexical arms |
| `reports/detector_v2_hard_negatives.json` | §5, all 13 cases with probabilities |
| `reports/detector_v2_error_analysis.json` | §6, category counts and examples |
| `reports/detector_v2_baseline_subsample.json` | §3.1, 600-row prefix baseline |
| `reports/detector_v2_train_comparison.json` | §10, retrained A/B, both calibrations |
| `artifacts/detector-v2/production_top1/*.jsonl` | §9, 260 rows from the real evidence path |

### 8.1 Regression status

| Suite | Result |
| --- | --- |
| `halluciguard_detector/tests` (13 files) | **186 passed** |
| `halluciguard_judge/tests/test_detector.py` | **10 passed** |
| `halluciguard_judge/tests/test_claim_extractor.py` + `test_verifier_contract.py` | **12 passed** |

PR #53 regression coverage, and the file that carries it:

| Check | Test file |
| --- | --- |
| Decorated RAGTruth labels → class ids | `test_data_conversion.py` |
| No public symbol removed | `test_schemas.py` |
| Annotation overlap in conversion | `test_data_conversion.py` |
| Numeric comparability guard | `test_guard_comparability.py` |
| Date comparability guard | `test_guard_comparability.py` |
| Entity comparability guard | `test_guard_comparability.py` |
| Bare-number extraction | `test_text.py` |
| `extra_special_tokens` loads | `test_detector_semantics.py` |
| Train/serve sequence length agree | `test_nli_input_contract.py`, `test_detector_semantics.py` |
| Judge fails closed when degraded | `halluciguard_judge/tests/test_detector.py` |

---

## 9. Production-path findings

260 claims were generated through the real `ClaimEvidenceEngine.select` path
with the cross-encoder reranker active (199 train + 61 dev), then the run was
stopped to free CPU for the retraining experiment in §10. Those 260 rows were
enough to settle three questions, and each answer contradicts something that
looked obvious before it was measured.

**1. The "hybrid" route is not hybrid in the shipped configuration.** Every one
of the 260 rows records `evidence_route: "hybrid"`,
`evidence_degraded: false`. The generation log for the same run says:

```
ERROR:root:Error loading dense model BAAI/bge-m3: We couldn't connect to
'https://huggingface.co' ... Falling back.
```

`BAAI/bge-m3` is absent from the default cache, so the dense half of the hybrid
retriever never ran and the trace does not say so. Production evidence
collected under the default configuration is BM25-only. Any A/B that compares
"production" against "lexical" while both are effectively lexical will conclude
that retrieval does not matter, which is the same conclusion for the wrong
reason. **Fix the trace before drawing retrieval conclusions.**

**2. Production evidence is longer than the lexical top-1 arm, not shorter.**

| Arm | mean chars | median chars | max chars |
| --- | --- | --- | --- |
| `lexical_joined` | 964.07 | — | — |
| `lexical_top1` | 206.67 | — | — |
| `production_top1` (real path) | **413.90** | **190** | 1394 |

The cross-encoder picks more complete sentences than lexical top-1 does. So the
alignment problem is not simply "production hands the model a short snippet".
That is worth stating because it changes the remedy: the gap between the
training blob and production evidence is a gap in *which sentence* is chosen
and *how much of it survives truncation*, not a simple length difference.

**3. A metadata bug was found in the arm generator itself.**
`evidence_snippet_count` recorded `3` on all 260 rows. The cause: the arm
counted the ranked list returned by `select_evidence` (top-3) while
`Detector.detect` classifies `snippets[0]` only. Every production row therefore
looked like a multi-snippet contract violation when it was in contract. Fixed
in `evidence_shapes.production_top1` to report `snippet_count=1` and keep the
rest under `extra["unused_ranked"]`, with a regression test
(`test_production_arm_reports_the_snippet_the_model_actually_sees`).

## 10. Experiment 4 — retrained A/B

**Design.** `lexical_joined` and `lexical_top1` retrained from
`microsoft/deberta-v3-xsmall` with identical settings — same epochs (3), batch
size (8), learning rate (2e-5), seed (42), label map, max length (256) and the
same dev split. The only difference is the evidence in the rows. This isolates
the effect of *training* on production shape, which §4 cannot test because §4
holds the checkpoint fixed.

Each arm is scored at **its own dev-fitted operating point**, with the
selection procedure held identical (same dev split, same grid, same rule, once
per arm). See [§10.1](#101-a-methodology-error-worth-recording) for why the
first attempt at this comparison was wrong.

**Natural test split** (800 rows: 731 / 28 / 41):

| Arm | Contr F1 | Contr R | Contr P | Verif F1 | **NEI F1** | Macro F1 | Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lexical_joined` | **0.4483** | 0.4643 | 0.4333 | 0.4427 | **0.0000** | 0.3183 | 0.9137 |
| `lexical_top1` | 0.3544 | **0.5000** | 0.2745 | **0.5652** | **0.2703** | **0.4085** | 0.9062 |

**Class-capped diagnostic split** (450 rows: 150 / 150 / 150):

| Arm | Contr F1 | Contr R | Contr P | Verif F1 | **NEI F1** | Macro F1 | Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lexical_joined` | 0.5081 | 0.4200 | **0.6429** | 0.6386 | 0.0132 | 0.1714 | 0.3356 |
| `lexical_top1` | **0.5352** | **0.5067** | 0.5672 | **0.7061** | **0.2559** | **0.2922** | **0.4000** |

### What this says

**Training on production-shaped evidence improves contradiction recall on both
splits** (+3.6 points recall on test, +8.7 on the diagnostic split) and
**improves every other axis on the diagnostic split** — verification F1 +6.8,
NEI F1 +24.3, macro-F1 +12.1. This replicates §4's direction from an
independent design: production-shaped single-snippet evidence is not the cause
of the contradiction-recall problem, and is better for it.

**The contradiction-F1 ordering flips between splits, and the test-split number
should not be trusted.** The test split holds 28 contradictions; 0.4643 vs
0.5000 recall is 13 vs 14 correct examples, so a single flip moves F1 by about
four points. The `lexical_joined` advantage there comes entirely from
precision (0.4333 vs 0.2745), i.e. from predicting contradiction *less often* —
which is the opposite of the objective. This is the concrete reason the
class-capped split exists, and it is also a reason not to over-read the balanced
split, since the balanced split no longer reflects the 91% SUPPORTED prior the
runtime actually sees.

**The clearest finding here is `NEI F1 = 0.0000` for `lexical_joined` on the
test split.** The arm the released model was trained on never predicts
NOT_ENOUGH_INFO at all. Since verification-needed routing is the operational
triage question the runtime asks, the current training shape is not merely
suboptimal, it removes one of three outputs from the model's repertoire. The
production-shaped arm restores it (0.2703). This is a stronger argument for
alignment than the contradiction numbers are, and it is independent of them.

**These retrained arms are not comparable to the released baseline in absolute
terms.** Each was trained on 900 rows; the released model saw far more. Only
arm-A-versus-arm-B is meaningful here.

### 10.1 A methodology error worth recording

The first version of this comparison scored every arm at the released
checkpoint's calibration, on the reasoning that a shared operating point removes
thresholds as a confound. It reported contradiction F1 = **0.0000 for both
arms**, and macro-F1 0.1714 with 0.3356 accuracy for `lexical_joined` on the
balanced split.

Both numbers were artefacts. Temperature and a 0.46 decision threshold are
properties of *one checkpoint's* logit scale. The retrained arms fitted
temperatures of 0.676 and 0.642 and dev-optimal contradiction thresholds of
0.245 and 0.205, so the released 0.46 sits well above where either of them ever
places P(CONTRADICTED), and both score exactly zero. A metric that is
identically zero for every arm is a broken transfer, not a weak model. The
0.3356 accuracy turned out to be genuine but misleading: three-class argmax is
invariant to temperature, so the `lexical_joined` arm genuinely predicts
SUPPORTED for 449 of 450 balanced examples — which is only catastrophic
because the diagnostic split is balanced, and is unremarkable on the 91%
SUPPORTED natural split.

The fix is in `experiments.py`: `format_comparison_table` now reports each arm
at its own operating point and the docstring records the failure so the
cross-checkpoint transfer is not reintroduced. The `shared_calibration` block
is still written to the JSON, labelled, so the mistake stays visible.

### 10.2 Pool-width timing

Killed. CPU-bound and not decision-relevant.

## 11. Known limitations

- The shape-sensitivity and error-analysis splits are **class-capped** (150 per
  class), so their absolute numbers are not comparable to the natural-split
  baseline. Only paired, within-split differences are meaningful.
- The hard-negative set is 13 hand-written cases. It is a failure-mode probe,
  not a benchmark.
- The error-analysis categories are heuristic and first-match. A human reading
  the listed examples will want to overturn some of them; §6 records one such
  overturn that already happened.
- `production_top1` evidence is dense-degraded. A run with `BAAI/bge-m3` present
  measured ~33 s/claim on this CPU-only host, which made a full production arm
  impractical here.
- The full natural-split baseline re-derivation is not included; the artifact
  numbers are quoted and their data-snapshot discrepancy is documented in §3.1.
- The retrained A/B arms were trained on 900 rows each, so §10 compares them to
  each other and to nothing else. It does not show that either arm beats the
  released model.
