# Detector, evidence and Memory integration

## Provenance and scope

Base: origin/main at f8900ca66721971b515674a10e2a17d8d8437cc0.
Reviewed source checkpoints:

- agents/detector-five-defects: 814d94f9aebb07e9e42147531b85dccbe868562e.
- agents/complete-project-audit-and-improvement: 0919b4aa31c89b7ff099a0c4a8ceb5f446ba6eb0.
- agents/detector-evidence-repair-20261002: fff500b4643d45bcff39e05769e659e5e79ded63.

Current main already carries compatible versions of the earlier Detector and
retrieval safeguards. This integration applies reviewed evidence/Memory repairs
and narrow regression fixes, not a wholesale merge of the obsolete HaluEval
architecture. Original worktrees and independent experiments remain intact.

## Retained and integrated safeguards

- Retain production RAGTruth/DeBERTa, per-instance lazy model initialization and
  locking, calibration configuration, grouped dataset preparation, label/span
  validation, and existing output-overwrite protection.
- Preserve all supported evidence wrapper collections, bounded normalization,
  source identifiers, and separate normalization/selection/tokenization metadata.
- Report sparse/dense contributions, errors, fusion, reranker execution, selected
  passage hashes, NLI submissions and observed preprocessing. Cached execution
  proof is historical, not a fresh model invocation.
- Distinguish founding place from headquarters, defer temporal or incomplete
  relationships to NLI, and avoid excluding unmentioned co-founders or merging
  distinct names. Preserve raw NLI scores and existing scoring thresholds.
- Preserve Enum evidence labels across canonical handoffs. The Judge's narrow
  decision-grade gate requires an actual matching citation label rather than
  neutral-only score dominance; its broader policy is not redesigned.
- Update linked Memory claims and fact confidence alongside vector/cache state.
  Attempt all stores and independently verify durable state. Report partial
  failures explicitly without restoring stale verified values.

## Verification-first release policy

Phase1Service already declines fast_path mode. Graph and direct-service routing
now honor that same disabled release gate, including when environment flags are
opted in. A grounded DeBERTa risk is not a Phase 1 uncertainty certificate.
Missing status and string-valued execution flags cannot imply successful
inference. Detector failure still permits the Verifier to run on a valid draft.

The direct LLM/Detector/Verifier slice has no post-retrieval Detector stage. In
certification mode it reports certification_completed=false and an explicit
reason rather than certifying initial evidence-free triage. Certification helper
failure cannot silently disable the requested mode or block retrieval.

## Exclusions and limitations

No retraining, threshold tuning, model replacement, new LLM calls, deployment,
frontend changes, model weights, datasets, live databases, or backup files are
included. The rejected pilot-training/MiniCheck utilities and experimental
training options remain on their original source branch.

Regression tests establish engineering contracts, not precision, recall,
calibration, false-accept rate, or production readiness. Dense/hybrid failure-path
tests use controlled backends. Optional token observations explicitly remain
unavailable when tokenizer interfaces do not expose preprocessing. Memory's
per-agent lock and per-file atomic replacement do not provide a cross-process,
multi-store transaction. Broader live-domain correction validation and a
reproducible held-out detection benchmark remain release work.
