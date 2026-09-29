# Detector Audit and Evaluation Report

Scope: `halluciguard_detector` (canonical), `halluciguard_judge` (legacy), and
their integration into `orchestration/` and `services/`.

Every number below was produced by code in this repository against the shipped
checkpoint. Nothing is estimated, extrapolated, or copied from prior
documentation. Where a measurement was not possible, that is stated as a
limitation rather than filled in.

Reproduce with:

```bash
python -m halluciguard_detector.model_card --strict     # metadata + consistency
python -m halluciguard_detector.benchmark               # curated runtime probe set
python -m halluciguard_detector.evidence_alignment      # train/runtime evidence gap
```

---

## 1. Architecture as it actually is

```text
LLM answer
  → atomic claim decomposition        (shared Verifier ClaimDecomposer)
  → hybrid retrieval (pool_k=20)      (shared: BM25 + dense fusion)
  → cross-encoder rerank (top=3)     (shared, reranked on the real claim)
  → DeBERTa-v3-xsmall NLI            (3-class: SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO)
  → Detector triage                   (verification_risk → HIGH/MEDIUM/LOW)
  → Verifier
  → Judge
```

Two scores, two questions, two thresholds:

| Signal | Formula | Question | Threshold |
|---|---|---|---:|
| `contradiction_mass` | `max P(CONTRADICTED)` | is the claim refuted? | 0.460 |
| `verification_risk` | `P(CONTRADICTED) + P(NOT_ENOUGH_INFO)` | does it still need checking? | 0.625 |

`verification_risk` is an operational triage score and must never be read as a
probability of falsehood. `hallucination_threshold` remains only as a deprecated
mirror of the Task B value so existing readers do not silently pick up a
different number.

The near-tie guard resolves `SUPPORTED`-vs-`CONTRADICTED` disagreement and is
driven by the **contradiction** threshold. Reading the verification-risk value
there is what previously let a triage score act as a falsity cut-off.

---

## 2. Bugs found, and what each one cost

### 2.1 Critical — the label-vocabulary rewrite would have destroyed the training set

`data.py` gained an exact-match alias table. The real RAGTruth `label_type`
values are decorated, not bare: `Evident Conflict`, `Subtle Conflict`,
`Evident Baseless Info`, `Subtle Baseless Info`, `Non Factual`. An exact-match
table misses all of them, and every annotation falls into the conservative
`NOT_ENOUGH_INFO` fallback.

Impact had it shipped: 100% of `CONTRADICTED` examples silently relabelled, no
crash, no warning, a corrupted label distribution, and a model that could never
learn to refute. Caught because the pre-existing test
`test_annotation_mapping` used the real vocabulary.

Fixed with longest-marker substring matching over an explicit marker table, plus
separator normalisation so `Non Factual` and `non_factual` resolve identically.
`"non_hallucinated"` contains `"hallucinat"`, so longest-match-wins keeps the
longer `SUPPORTED` marker rather than flipping it to `NOT_ENOUGH_INFO`.

### 2.2 Critical — `prepare_ragtruth` raised `NameError` after writing the data

`stats.json` dereferenced the removed `_ALIAS_LOOKUP`. The preparation step
wrote every data file and *then* failed, so the expensive work was done and lost.

Fixed, and now covered by an end-to-end test that also asserts a real
`Evident Conflict` span survives to disk as `CONTRADICTED`.

### 2.3 High — coverage summed overlapping annotations

`_coverage` added per-annotation overlap lengths. Two annotations over the same
half of a sentence produced `coverage = 1.0`, so a half-annotated sentence was
force-labelled as if a human had marked every word. Now computed as a **union**
of intervals.

### 2.4 High — a bare year disagreement was reported as a contradiction

"Tesla was founded in 2003" vs "Tesla went public in 2018" raised a year
conflict. Both statements are true; they describe different events. This
manufactured contradiction mass in exactly the direction the system must be
conservative in.

Fixed by requiring a shared relation before any numeric or year clash counts.
The gate is permissive where it should be: a genuine same-predicate clash
("Released in 1995" / "released in 1996") is still detected.

### 2.5 High — the comparability gate was defeated by entity misdetection

`shared_relation` excluded the *union* of detected entities from both sides. The
entity regex flags any capitalised word, so a sentence-initial "Released" was
read as a proper noun and the shared predicate "released" was erased from both
sides — hiding the real 1995-vs-1996 conflict that 2.4 had just enabled.

Fixed: a capitalised word counts as a proper noun only if it never occurs in
lowercase anywhere in the two texts. "Released" is a verb; "Company A" /
"Company B" stay excluded.

### 2.6 High — unitless numbers were compared through a placeholder unit

`quantities()` gave numbers with no unit the pseudo-unit `"unit"`, so **any** two
differing bare integers in a comparable sentence were flagged — identifiers
(`record id 12345` vs `98765`) and plain counts included. This directly
contradicted the function's own docstring.

Fixed by requiring a real unit. Years remain handled by the separate year rule,
where 1995-vs-1996 genuinely is a clash.

### 2.7 High — the shipped checkpoint could not be loaded

`tokenizer_config.json` carried a list-form `extra_special_tokens`, raising
`AttributeError: 'list' object has no attribute 'keys'` under the installed
`transformers`. Any consumer loading `artifacts/detector-best` from a clean
checkout hit this.

Fixed by removing the redundant field and setting `model_max_length=256`.
Verified by loading the real checkpoint and running the benchmark.

### 2.8 Medium — train/serve sequence-length skew

Code defaulted to 384; the artifact and model card used 256. Silent skew between
training, evaluation and serving.

Fixed with `calibration.DEFAULT_MAX_LENGTH = 256` as the single source, imported
by `training.py` and `detector.py`, and cross-checked against
`tokenizer_config.json` by `model_card.py --strict`.

### 2.9 Medium — legacy `halluciguard_judge` degraded mode did not fail closed

`_ensure_classifier_loaded()` returned `load()`'s success, but `load()` succeeds
even when it falls back to raw base weights. The **first** request therefore
reported a healthy model while every later request correctly reported degraded.
A fail-closed guarantee enforced only from the second request is not a guarantee.

Fixed to report `_is_finetuned`. The package's own
`test_degraded_mode_is_fail_closed` was failing at HEAD before this change.

### 2.10 Resolved in the preceding change

Class-mapping normalisation, `None` refutation mass, grounded forwarding,
canonical field preference, unknown-only verification guard, and `pool_k`
slicing were fixed in `431d385` (PR #52) and are included here.

---

## 3. Measured evaluation

Source: saved dev/test logits in `artifacts/detector-best/*.npz`, temperature
0.7586, thresholds fitted on **dev** and applied unchanged to test.

Test distribution: SUPPORTED 17,258 (91.91%), CONTRADICTED 592 (3.15%),
NOT_ENOUGH_INFO 927 (4.94%).

| | Task A (contradiction) | Task B (verification needed) |
|---|---:|---:|
| Score | `P(CONTRADICTED)` | `P(C)+P(NEI)` |
| Positive rate | 3.15% | 8.09% |
| Precision | 0.2449 | 0.3096 |
| Recall | 0.5236 | 0.5293 |
| F1 | 0.3337 | 0.3907 |
| ROC-AUC | 0.8500 | 0.8212 |
| PR-AUC | 0.2311 | 0.2886 |
| Accuracy | 0.9341 | 0.8664 |
| ECE | 0.0513 | 0.1296 |
| Brier | 0.0409 | 0.1083 |
| FPR / FNR | 0.0526 / 0.4764 | 0.1039 / 0.4707 |
| TN / FP / FN / TP | 17229 / 956 / 282 / 310 | 15465 / 1793 / 715 / 804 |

Three-class argmax: accuracy 0.8383, macro-F1 0.5072, multiclass Brier 0.0704,
ECE 0.0864. Per-class recall: SUPPORTED 0.8718, CONTRADICTED 0.5490,
NOT_ENOUGH_INFO 0.3991.

**Reading these honestly.** 86–93% accuracy is not the story; it is dominated by
the 92% SUPPORTED majority. Macro-F1 (0.51) and minority recall are the honest
summaries. The model misses roughly half of all refutations and 60% of all
insufficient-evidence cases — **recall is the weak axis**. It is well ranked
(ROC-AUC 0.82–0.85) but poorly calibrated for Task B (ECE 0.13), so its ordering
is useful and its raw scores are not.

Task B reproduces the historical figures exactly (F1 0.3907, ROC-AUC 0.8212,
PR-AUC 0.2886), which confirms the refactor did not alter the measured
behaviour — it only stopped the number being mislabelled as "hallucination".

### Runtime probe set

10 curated cases, real checkpoint: **3/8 (37.5%)** exact-label accuracy.
All five errors predicted `SUPPORTED` for claims that were contradicted or
unverified. The model fails in the conservative direction and produced **no**
false accusation, and all 8 semantic invariants held.

Invariants are asserted in the test suite; the accuracy figure deliberately is
not. A wrong model answer is a measurement; a wrong invariant is a code defect.

### Train/runtime evidence mismatch

| Strategy | Agreement with production | Mean abs. risk delta | Max delta |
|---|---:|---:|---:|
| `training_lexical_joined` (training) | 0.6667 | 0.2371 | 0.6090 |
| `lexical_top1` | 0.6667 | 0.1456 | 0.5531 |
| `hybrid_top1` (no rerank) | 0.6667 | 0.1564 | 0.5531 |
| `hybrid_rerank` (production) | 1.0000 | 0.0000 | 0.0000 |

Training feeds the classifier up to six concatenated lexical snippets; runtime
feeds it one reranked snippet. The two disagree on a third of probes with a mean
verification-risk shift of 0.237 — enough to cross the 0.625 threshold.

**Aligning training evidence with the production shape is the highest-value
follow-up**, ahead of any threshold tuning. The dense leg was unavailable
offline, so these hybrid rows are BM25 + reranker, not true fusion. This is a
9-probe mechanism check, not an accuracy estimate.

---

## 4. Test results

| Suite | Result |
|---|---|
| `halluciguard_detector/tests` | **131 passed** |
| `halluciguard_judge/tests` | **19 passed** (1 pre-existing failure fixed) |
| `orchestration/tests` | 177 passed, 2 failed |
| `tests/` | 10 passed |
| Combined | **354 passed**, 2 failed |

The 2 orchestration failures are **pre-existing and environmental** — both also
fail on base commit `5fb5008` with the Detector changes stashed, and both depend
on the offline `BAAI/bge-m3` dense encoder:

- `test_verifier_stabilization.py::test_single_claim_end_to_end_verification`
- `test_verifier_v1_stabilization.py::test_1_paris_capital_of_france`

`agents/corrector_agent/test_api.py` and `test_orchestrator.py` fail to collect
(`No module named 'app'`), also pre-existing and unrelated.

`ruff check --select F,E9` on the changed packages: clean.

---

## 5. Limitations — what was not measured

- **The raw RAGTruth source is not in this repository.** The conversion,
  downsampling and evidence shapes could not be re-derived or re-trained here.
  All metrics come from the saved logits in the checkpoint. No model was
  retrained as part of this work.
- **RAGTruth annotates spans, not runtime atomic claims.** Training examples
  are a closer but not identical match. `stats.json` records the
  sentence-level versus span-level split so the gap stays visible.
- **The dense retrieval leg was unavailable offline** (`BAAI/bge-m3`), so the
  shared hybrid retriever ran as BM25 + reranker. True hybrid fusion is
  unmeasured.
- **The runtime probe set is 10 hand-written cases.** It is a semantic
  regression harness, not a benchmark, and must not be read as an accuracy
  estimate.
- **The evidence-alignment experiment is 9 probes over a 12-document corpus.**
  It quantifies a mechanism; it does not estimate accuracy at scale.
- **Unknown `label_type` values fall back to `NOT_ENOUGH_INFO`.** This is
  conservative and counted, but a dataset revision could still shift the label
  mix if it introduced new vocabulary.
- **Recall remains the weak axis** (0.52 contradiction, 0.40
  insufficient-evidence). Thresholds were fitted for F1; operating the
  Detector as an autonomous decision-maker is not supported by these numbers.
- No new model calls were added, and the shared retrieval components were not
  duplicated.

---

## 6. Recommended next steps, in order

1. **Align training evidence with the production single-snippet shape.** Largest
   measured effect: 0.237 mean risk shift, a third of probes mislabelled.
2. **Attack recall.** Threshold tuning cannot fix F1 0.39 when minority recall
   is 0.52/0.40; this needs class weighting, resampling, or more
   `NOT_ENOUGH_INFO` training signal.
3. **Re-derive the label mix** once the raw dataset is available, and check
   `unknown_label_type:*` counters for vocabulary drift.
4. **Run the evidence experiment with the dense leg available** to complete the
   hybrid-vs-reranker comparison that was blocked offline.
5. **Resolve the 2 environmental orchestration failures** by providing the dense
   encoder in CI.
