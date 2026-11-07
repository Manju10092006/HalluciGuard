# Reliability investigation, 2026-11-06

The frozen entry state is recorded in `results/research_20261106_r1/baseline/inventory.json`. This branch retains the historical policy by default. The reviewed manuscript, source archive, supplement, original figures, checkpoints and result files remain preserved.

## ReVerifier diagnosis before implementation

The four standalone valid probes and eight valid mixed candidates are not missing claims. Archived claim IDs and texts reconcile, coverage is complete, status is completed, and their failures occur before the topicality veto. The archived default graph verifies the repaired capital but returns unverified for the location. Its raw trace records one retrieved, reranked and NLI-selected passage, then zero decision-grade passages. The standalone probes use the prior experimental `nli_temporal` configuration. Its capital failures must not be attributed to the baseline structured configuration.

Fresh CPU inference on the unchanged Asterwyn passage produces entailment 0.973643 for the capital and neutral 0.981536 for the explicitly repeated location. `RelationVerifier.extract_triples` (original lines 220 onward) does not cover `lies in`, so `VerificationPipeline._select_decision_grade_evidence` (original lines 351 to 503) receives no structured rescue and no pair score above its decision threshold. `_reverifier_node` (original lines 1218 to 1456) correctly withholds a candidate containing an unresolved report. Removing its all-verified requirement would conceal the upstream failure.

Hypothesis for an opt-in candidate: recognize a bounded, declarative `X lies in Y` construction, preserve polarity and temporal restrictions, and let unresolved extraction defer to neural evidence rather than treating parser failure as evidence irrelevance. A match is evidence-conditioned. A new triple alone is never support. Test valid candidates alongside residual and newly introduced false assertions. Preserve and compare the original selector and scores. The diagnostic is a fixed-passage component experiment, not a new hosted graph run or a human evaluation.

## Judge diagnosis before implementation

The five archived false targets share query anchors with verified alternatives. `_claim_leaves_core_gap` removes already grounded anchors and can therefore treat a competing answer as peripheral. The default policy is retained for reproducibility. An opt-in `slot_aware` policy will treat an unresolved alternative for the same subject and relation as a grounding gap, without declaring the alternative false. Cofounders may have multiple true values, but every released competing value still needs evidence. `strict_all_verified` will require all reported claims to be verified. Frozen-state replay cannot predict the outcome of additional retrieval retries.

## Memory diagnosis before implementation

`MemoryAgent.store_fact` receives evidence URLs and excerpts but saves only domain metadata on graph source nodes, a count in SQLite, and generic metadata in vectors (original lines 181 to 252). This discards provenance before readback. The graph exception path also omits the canonical `memory_result` even though the agent outcome reports failure. An additive provenance record and canonical failure result can be tested with isolated storage. These changes do not establish a benefit for memory-assisted generation.

## Measured outcomes, 7 November 2026

The original gate accepted 0/8 valid mixed repair controls and withheld all 16 invalid controls. The opt-in soft evidence policy accepted 8/8 valid controls and still withheld all 16 invalid controls. All 24 candidates had complete report coverage in each mode. Mean gate-only time was 11.34 ms versus 14.29 ms; this excludes inference, retrieval and the repair provider. Predictions, fresh raw neural scores, revised texts and the actual gate outputs are in `repair_diagnosis` and `repairs` under the new run. The same known diagnostic cards informed the change. This is not a held-out repair estimate.

On 100 previously inspected reference targets, structured/no-scoped/soft/NLI-temporal agreement was 68/82/85/83. Soft helped 24 and harmed seven relative to structured. The held-out ANLI R2 sample was selected before inference and all four modes agreed with 38/120 labels. Soft retained 17 false supports and increased false contradictions from 22 to 23. There is no held-out aggregate gain; soft remains optional. The experiment does not promote a new default.

Across all 180 Judge states the default reproduced the historical action. At the final state for each of 100 requests, both alternate policies withheld the same five known false targets and retained the same 48 target-passing releases. They tied on this cohort. They do not establish correct whole-answer release or the outcomes of their additional retry requests. The proposed slot rule checks competing assertions rather than declaring every alternative value false.

The Detector audit reproduced the historical metrics and dataset repetition counts. Its removal changed zero actions in 180 Judge states. Binary sentence flags now use the calibrated risk threshold rather than the three-class argmax. This is a contract correction; no new cost or latency benefit is measured.

Corrector literal preservation now rejects deletion of a designated preserved string from a repaired sentence. Memory tests recover supplied evidence and execution metadata from all three primary stores, propagate vector save failure and emit canonical failure at the graph boundary. Cross-store rollback, duplicate provenance merging, and memory-assisted generation remain unresolved.

The approved Groq baseline pilot stopped after three provider failures, following 42 requests. It completed 12 of 80 planned records and failed three of the 15 attempted records. Detailed provider error codes were omitted in the executed export, so the precise remote failure causes are unavailable. Failure objects misleadingly inherited default provider metadata from `BaseLLMService._failed`; configured routing allowed only Groq. A later runner export change retains the attempt trail and error code. The exact executed runner was reconstructed and hash-checked in `external_baselines/executed_producer.py`. No post-stop retry or provider fallback was attempted.

The broad regression run passed 601 tests and the final targeted run passed 11, with overlapping cases. Regression counts are software evidence, not factual accuracy. Source line numbers for the initial diagnosis refer to the immutable entry snapshot; current symbols and locations are listed in `docs/CODE_MAP.md`.
