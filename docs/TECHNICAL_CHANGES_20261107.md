# Validated implementation changes

| Area | Change | Evidence | Remaining boundary |
| --- | --- | --- | --- |
| Verifier | Selectable original, no-scoped, temporal and soft policies. Soft defers inconclusive structure to NLI and recognizes a bounded literal location construction. | 100-target development replay, 24-candidate repair replay, 120-example fresh ANLI R2 transfer and regression controls. | No held-out aggregate agreement gain; optional only. |
| Judge | Historical default, strict all-verified and answer-slot-aware policies. Explicit alias map, selected predicates, polarity/hedge handling and temporal scope. | All 180 historical default actions reproduced; both alternatives remove five known false targets among 53 releases. | Bounded parser, upstream false verification, extraction omissions and subsequent retry outcomes remain. |
| ReVerifier | Raw verification diagnostics retained; gate unchanged. | Actual gate invoked over frozen NLI; complete coverage 24/24 per policy. | Same infrastructure can repeat errors; fallback coverage is not semantic completeness. |
| Detector | Per-sentence verification flags use binary risk threshold. | Boundary cases where argmax and combined binary risk disagree. Historical metrics reproduced. | No demonstrated routing/cost benefit in the examined normal path. |
| Corrector | Literal preserved-claim check after authorized sentence splicing. | Supported clause deletion inside a target sentence is rejected. | Semantic preservation and new factual assertions require independent verification. |
| Memory | Structured provenance in graph, vectors and SQLite; vector save errors surface; canonical graph failure result. | Isolated store readback, reopen of all three persisted stores and forced failures in regression suites. | Non-atomic stores and direct API admission remain; no recall-generation evaluation. |
| Baselines | Nine named interface routes, bounded callbacks, three clearly approximate literature adapters. | Four adapter tests and 42-request authorized pilot. | Pilot incomplete after preset provider-failure stop. |
| Annotation | Blinded CSVs, guidelines, paired agreement and adjudication tooling. | 309 blank items and tests for blank, missing and duplicate IDs. | Human judgments pending. |

No trained checkpoints, calibration files, historical datasets or prior results were overwritten. User-owned initial changes remain outside the new commit. The original branch entry is recorded in the baseline inventory. Test XMLs and experiment configurations identify the actual working-source hashes at execution; a later code commit does not retroactively change those identities.
