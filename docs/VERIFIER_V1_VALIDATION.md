# HalluciGuard Verifier V1 Validation

## Scope and preserved architecture

This validation covers only `agents/verifier_agent/`, its dedicated diagnostics,
and verifier tests. The existing chain remains unchanged: domain routing, source
adapter retrieval, hybrid retrieval, reranking/relevance, NLI, evidence scoring,
and verdict aggregation. Detector, Judge, Corrector, Memory, Frontend,
LangGraph orchestration, deployment, and Docker were not modified.

## Root causes and repairs

| Symptom | Cause | Repair |
| --- | --- | --- |
| Paris could become `CONFLICTED` | Tangential Wikipedia pages from the same provider reached NLI and were counted as independent contradictory evidence. | Apply a deterministic relevance and per-source diversity gate before NLI. |
| Myth-discussion text could appear supportive | Literal NLI can entail a quoted/described claim without asserting it. | Reject support when the complete claim is presented as folklore, fiction, belief, allegation, or similar non-assertive context. |
| Live adapter contract tests failed | Wikipedia and PMC request order changed without matching the documented contract/tests. | Restore REST-first Wikipedia retrieval and summary-first PMC metadata retrieval, with fallbacks/extraction retained. |
| Verifier tests used old verdict enum values | Tests asserted removed labels rather than the public V1 contract. | Update assertions to `CONTRADICTED` and `UNVERIFIED`. |
| Diagnostic CLI failed on Windows evidence text | Console output used the legacy code page. | Configure standard streams to UTF-8 with replacement for unencodable output. |

## Evidence and verdict behavior

- `VERIFIED`: decision-grade relevant evidence supports the claim.
- `CONTRADICTED`: decision-grade relevant evidence contradicts it.
- `UNVERIFIED`: no decision-grade evidence, neutral evidence, or degraded NLI.
- `CONFLICTED`: relevant evidence from distinct sources supports both sides.

The relevance gate uses normalized cross-encoder scores, accepting scores of at
least `0.40`. This is the existing evidence scorer's relevance boundary; it is
now applied before NLI. Passages below it never receive an NLI decision and
therefore have no scoring, conflict, or citation impact. Multiple passages from
the same provider are treated as one source for a single subclaim, retaining the
highest-reranked candidate while preserving conflict across independent sources.

## Routing, sources, cache, and degraded state

- General requests use the `general` adapter and Wikipedia REST search, with
  MediaWiki Action API fallback.
- Healthcare uses PubMed, PubMed Central, OpenFDA, and ClinicalTrials.gov; the
  response reports attempted, succeeded, and failed sources.
- `VERIFIER_CACHE_ENABLED=false` bypasses cache reads and writes; `true` retains
  existing SQLite caching.
- Degraded NLI results are excluded from decision-grade evidence and produce no
  factual support or contradiction.
- If an adapter fails or returns no usable evidence, response stage status and
  source fields make the degraded/insufficient state observable.

## Test results

| Command | Result |
| --- | --- |
| `python -m compileall agents\\verifier_agent` | Passed. |
| `cd agents/verifier_agent; python -m pytest -q` (baseline) | 67 passed, 6 failed: legacy verdict assertions and stale adapter-order tests. |
| `cd agents/verifier_agent; python -m pytest -q` (after repair) | 75 passed, 8 warnings. |
| `python -m pytest -q orchestration/tests/test_verifier_v1_stabilization.py` | 8 passed, 9 warnings. |
| `cd agents/verifier_agent; python -m pytest -q tests/test_final_stage_hardening.py tests/test_official_integrations.py` | 14 passed. |

## Live diagnostic results

| Claim | Result |
| --- | --- |
| Paris is the capital of France. | `VERIFIED` |
| The Eiffel Tower is in London. | `CONTRADICTED` |
| The Moon is made of green cheese. | `UNVERIFIED` |
| Aspirin is used to treat mild pain. | `UNVERIFIED` |
| Xyzabc123 is the capital of France. | `UNVERIFIED` |

Warnings are from upstream Torch/Transformers deprecations and the existing
Detector Pydantic configuration, not Verifier failures. `rank_bm25` is not
installed in this environment; sparse retrieval reports the condition and the
existing hybrid retriever falls back deterministically rather than reporting
BM25 success.

## Remaining limitations

Live source content and model scores can change. The generic non-assertive
context policy is deliberately conservative: it prevents quoted myth/fiction
discussion from verifying a claim, but may return `UNVERIFIED` when no direct
contradictory passage is available. Per-source diversity prevents a single
provider's search variants from creating a conflict; independent source
corroboration is still supported.

The corrected healthcare query retains treatment context, but the live official
sources in this run did not yield a decision-grade aspirin/pain passage after
reranking. The pipeline correctly reports `UNVERIFIED` instead of fabricating
evidence; the required Aspirin `VERIFIED` acceptance probe therefore remains
unsatisfied.

## Status

**V1 NOT YET STABILIZED** because the live Aspirin acceptance probe remains
unverified.
