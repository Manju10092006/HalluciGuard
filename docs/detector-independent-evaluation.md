# Independent Detector evaluation workflow

This workflow never enables fast-path acceptance or establishes production readiness.
It evaluates a supplied frozen checkpoint on supplied prepared claim/evidence rows;
it does not call n8n, retrieve evidence, train, select a checkpoint or tune thresholds.
Dataset locks do not retroactively establish the training history of a checkpoint.

## Input contract and data roles

Each JSONL row must contain nonempty string `id`, `source_id`, `claim`, `evidence`,
and exactly one human-supplied label and matching integer ID:
`SUPPORTED=0`, `CONTRADICTED=1`, `NOT_ENOUGH_INFO=2`. The tool rejects aliases,
invalid labels and missing provenance. Supply `response_id`; for existing RAGTruth
converter IDs only, the documented `response:span-start-span-end` representation
(`response:12-48`, optionally `#region`) permits deriving the original response ID.
Explicit `document_id` is preferred; otherwise document grouping uses `source_id`.
Identifier values remain case-sensitive and cannot be padded with whitespace.

Roles are `training`, `development` and `locked_test`. Only development is allowed
for temperature, thresholds, model/hyperparameter selection. The locked split is
not rebalanced, oversampled, relabelled or deduplicated. Exact row identities,
source/response identities, label counts and every file are pinned before metrics.
No examples are silently removed. Exclusion lists/counts are explicit (currently
empty; contaminated supplied splits are rejected, rather than cleaned invisibly).

## Exact and near contamination audits

Text normalization v1 is Unicode NFKC, then casefold, then collapse all Unicode
whitespace to single spaces. Punctuation, negation and numbers remain. The Unicode
database version is recorded and verified. Exact audits cover row IDs, claims,
evidence, claim+evidence pairs, source IDs, document IDs, response IDs and composite
source+response IDs, within and across all three roles. Repeated response/source
IDs within a split can represent multiple claims; they are reported, not treated
as independent examples. Duplicate row IDs are always invalid lock identities.

Near duplicates require **both** normalized claim and evidence character-5-gram
set Jaccard similarities to be at least 0.85 (configurable before locking).
Strings shorter than five characters have a single full-string shingle. An
inverted claim-shingle index enumerates all potentially qualifying pairs; none
with positive claim similarity is omitted. Exact pairs are counted separately.
No random sampling, approximate LSH, top-K cut or silent removal is used. The
candidate budget defaults to 10,000,000; exhaustion fails the audit as incomplete.
Increase `--max-candidates` explicitly if practical. Counts, thresholds, pair IDs
and similarities are reported. This lexical method is not a semantic-paraphrase
or translated-content audit; its zero overlaps would not prove semantic isolation.

`build` groups connected components of repeated source/document/response IDs,
exact content and measured near duplicates, then deterministically assigns whole
components by SHA256 of seed and sorted member IDs. Default component fractions
are 80/10/10; these do not promise the same row/class fractions. At least three
independent components are required; insufficient data blocks rather than forces
a split. Existing split assignments can instead be audited and frozen with `lock`.
An official held-out split must be supplied through `lock`, not reshuffled using
`build`. Neither command authorizes fitting or implies an already trained model
never saw the new final split.

```sh
python -m halluciguard_detector.independent_evaluation audit \
  --train TRAIN.jsonl --dev DEV.jsonl --test TEST.jsonl
python -m halluciguard_detector.independent_evaluation lock \
  --train TRAIN.jsonl --dev DEV.jsonl --test TEST.jsonl \
  --dataset-name NAME --dataset-version VERSION --git-sha SOURCE_SHA \
  --created-at FIXED_ISO_TIMESTAMP --exposure-record EXPOSURE.json \
  --output halluciguard_detector/data/independent-NAME-VERSION
```

Use ignored local data directories. Do not commit restricted/private inputs or
large weights. The tool returns the **external** SHA256 of `gate-manifest.json`;
record it independently and pass it explicitly on verification/evaluation. A
self-reported manifest hash does not provide mutation protection. Split manifests
contain sorted row/source/response IDs, label counts, exclusions, seed, timestamp,
git identity, normalization/Unicode version, source artifact hashes and audit
summary. Reuse the creation timestamp and identical source paths/version inputs
for byte-identical regeneration. The bundle cannot be overwritten in place.

## Historical checkpoint exposure

An exposure declaration must link the *actual historical* training and calibration
row artifacts to both the frozen checkpoint and the frozen calibration file:

```json
{
  "checkpoint_sha256": "ACTUAL_WEIGHT_BYTES_SHA256",
  "calibration_sha256": "ACTUAL_CALIBRATION_BYTES_SHA256",
  "coverage": "complete",
  "attested_by": "RESPONSIBLE_PROVENANCE_OWNER",
  "training_artifacts": [{"path": "HISTORICAL_TRAIN.jsonl", "sha256": "FILE_SHA256"}],
  "development_artifacts": [{"path": "HISTORICAL_DEV.jsonl", "sha256": "FILE_SHA256"}]
}
```

The gate hashes the declaration at lock time, rechecks every referenced historical
file, and audits all exact/group/near overlaps against locked test. Historical
train/dev contamination also blocks independent calibration. Missing or incomplete
membership blocks evaluation, including when a freshly generated split looks clean.
Hashes cannot independently prove completeness of a supplied declaration; that
claim needs genuine historical records and owner attestation. Never fabricate it.
Account for LF/CRLF calibration bytes explicitly; the values must remain frozen.

```sh
python -m halluciguard_detector.independent_evaluation verify \
  --bundle BUNDLE --manifest-sha256 EXTERNALLY_RECORDED_SHA256
python -m halluciguard_detector.independent_evaluation evaluate \
  --bundle BUNDLE --manifest-sha256 EXTERNALLY_RECORDED_SHA256 \
  --checkpoint artifacts/detector-best --output LOCAL_FRESH_OUTPUT \
  --bootstrap-seed 20261003 --bootstrap-replicates 2000
```

The runner checks all locks and exposure before reserving output, writes a hashed
evaluation plan **before** inference, loads only local checkpoint/tokenizer files,
uses the existing canonical NLI input/pair/truncation contract, and rechecks locks
and protected checkpoint files after inference. Failed runs may leave a plan or
partial local predictions, but never publish `evaluation.json` as a success.
Do not feed private prediction files back into the repo. CLI blockers exit 2 with
`INDEPENDENT_EVALUATION_BLOCKED`, no fabricated metric values.

## Calibration protection

`labels_for_role(bundle, pinned_hash, role)` verifies the bundle and returns
read-only role-bound labels. Locked markers survive normal slicing. Standard
`fit_temperature` and `best_threshold` reject locked/training markers even if an
explicit development declaration is supplied. Naked arrays are rejected unless
the caller explicitly declares `data_role="development"`; existing offline
training does so only after verifying managed files and row-role markers. Copying
or renaming managed locked rows into `dev.jsonl` is rejected before model loading.
Training code was not executed or the checkpoint/calibration changed in this task.

This is a safeguard against accidental normal-tool use, not an access-control
boundary against a researcher deliberately stripping metadata, relabelling data
and falsely declaring development provenance. Restrict final labels in the actual
research process as well. Legacy unmanifested numerical archives lack row/source
provenance and cannot establish an independent gate through these tools.

## Metrics and confidence intervals

`model_quality` reports argmax three-class accuracy/confusion/P/R/F1, support,
macro F1 and micro F1. Missing denominators are null; macro F1 includes all three
classes with zero for undefined class F1. Probabilistic diagnostics are top-label
ECE over ten equal-width bins and multiclass Brier (mean sum over classes,
range 0–2), alongside calibrated probabilities and raw logits/classes per row.

`operational_policy` separately reports saved contradiction-threshold P/R/F1,
FP/FN counts and hypothetical risk-based bypass: eligibility is
`P(CONTRADICTED)+P(NEI) < saved_risk_threshold`. It reports coverage, contradicted
and all non-supported eligible counts, conditional false accepts within eligible
rows, false accepts among all non-supported rows, supported rows routed for
verification and overall routing/abstention fraction. These are model-score
diagnostics, not the complete production routing/Judge policy or truth guarantees.
Fast-path acceptance stays disabled irrespective of these values.

The default 95% percentile bootstrap uses seed 20261003 and 2,000 source-cluster
replicates: sample source groups uniformly with replacement, retain every claim
from each sampled source (including repeated sampled groups), and recompute
metrics. This respects within-source dependence, unlike row IID resampling.
Intervals report valid/undefined replicate counts; absent denominators are never
replaced with zero. At least two source clusters are required. Few source groups,
across-source dependence and incomplete provenance limit interpretation. The
bootstrap does not estimate model-training uncertainty or cure leakage.

Unit tests use small synthetic fixtures solely to verify this framework. Injected
predictors are refused for real bundles; synthetic runs always return
`SYNTHETIC_FRAMEWORK_TEST_ONLY`, never establish independent model performance.
