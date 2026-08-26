# STEP 5 IMPLEMENTATION REPORT

HalluciGuard Corrector Agent — candidate validation and bounded retry.

> **Execution honesty.** Every test count in this report was produced under
> *test-only stdlib shims* (`_pydantic_shim`, `_pytest_shim`), because this
> environment has no `pytest`, no `pydantic`, and `pip` is blocked (verified: all
> three fail). The shims are strictly narrower than the real libraries, so a green
> run is meaningful evidence but **not** a certification. A real `pytest` +
> `pydantic` v2 run remains outstanding and is a Step 6 prerequisite. These
> numbers must never be reported as "pytest passed".

---

## Files created

Production (Step 5 core):

- `agents/corrector_agent/corrector/failure_codes.py` (73 lines) — the 12-code
  validation vocabulary, plus `RETRYABLE_CODES` and `TERMINAL_CODES` sets.
- `agents/corrector_agent/corrector/text_features.py` (637) — pure extraction
  primitives: numbers (digit + word form), dates, quantities, entities, content
  tokens, abstention detection, scaffolding/control-char detection, similarity and
  edit-distance. No I/O, no model, no side effects.
- `agents/corrector_agent/corrector/validators.py` (953) — the ten independent
  gate functions as pure predicates over `(candidate, original, evidence)`, plus
  abstention detection and the echo ladder.
- `agents/corrector_agent/corrector/validation.py` (733) — the 12-gate pipeline
  (`validate_candidate`), the fixed `GATE_ORDER`/`SHORT_CIRCUIT_GATES`, context
  builders from Step 2–4 output, and the optional `EntailmentScorer` seam
  (`NullEntailmentScorer`, `EntailmentJudgement`).
- `agents/corrector_agent/corrector/retry.py` (828) — bounded retry: classification
  (`classify_retry`), append-only retry-prompt construction (`build_retry_prompt`),
  the resolve loop (`resolve_target`, `resolve_generation`), and the fixed
  `RETRY_CONSTRAINTS` vocabulary.

Tests (Step 5):

- `agents/corrector_agent/tests/test_validation.py` (766) — gates + pipeline,
  including the mandatory recorded echo-failure test.
- `agents/corrector_agent/tests/test_retry.py` (563) — policy, bounds, retry-prompt
  construction, and security.
- `agents/corrector_agent/tests/test_step5_fuzz.py` (349) — seeded property/fuzz
  tests and AST-level scope guards.

Test-only harness (not production code; installed only when the real packages are
absent):

- `agents/corrector_agent/tools/_pydantic_shim.py`, `_pytest_shim.py`,
  `run_tests_stdlib.py`, `_pure_loader.py`, and two standalone verifiers
  (`verify_text_features_stdlib.py`, `verify_validators_stdlib.py`).

## Files modified (all additive)

- `agents/corrector_agent/corrector/contracts.py` — appended the Step 5 internal
  contracts (`ValidationFailure`, `CandidateValidation`, `TargetDecision`,
  `CandidateAttempt`, `TargetResolution`, `ValidationOutcome`). No Step 1–4 model
  was altered.
- `agents/corrector_agent/corrector/config.py` — added Step 5 thresholds
  (`evidence_alignment_threshold`, `minimal_edit_min_similarity`,
  `max_expansion_ratio`, `require_entailment_check`, `enforce_single_sentence`,
  `max_prompt_tokens`) and their environment-variable parsing.
- `agents/corrector_agent/corrector/__init__.py` — a `STEP 5 STATUS` docstring note
  and re-exports of `validate_candidate`, `resolve_generation`, `resolve_target`,
  `EntailmentScorer`, `NullEntailmentScorer`. **`correct()` is unchanged** and
  remains the Step 1 passthrough (verified: `status=completed`,
  `corrected_text == original_response`, `validation_status=unvalidated`).
- `agents/corrector_agent/tests/conftest.py` — added the Step 5 fixtures
  (`step5`, `step5_case`, `recorded_echo_case`) built from real Step 1–4 code, and
  aliased the recorded missing-person strings onto the kit.

## Files untouched (verified)

`orchestration/schemas.py` and every other protected component were checksum-verified
byte-for-byte against a baseline taken before Step 5: **422 / 422 files OK, 0 failed,
0 missing.** The baseline covers `orchestration/schemas.py`, `agents/corrector_agent/app/`,
`Corrector_agent-reference/`, `agents/verifier_agent/`, `agents/detector_agent/`,
`agents/memory_agent/`, and `agents/corrector_agent/training/` (Step 4 benchmark
infrastructure and training artifacts). The Step 1–4 corrector modules
(`adapter.py`, `targeting.py`, `evidence.py`, `prompt_builder.py`, `model_client.py`,
`output_parser.py`, `sentence_utils.py`) were not modified.

## Architecture implemented

A candidate flows through a single, fixed pipeline and then, if it fails
retryably, a bounded retry loop:

```
GenerationOutcome (Step 4)
   → build_contexts(plan, bundles)              # authorized ids, evidence pairs
   → for each target: resolve_target
        → validate_candidate  (12 gates, conjunctive)
        → classify_retry      (retry only if every failure is retryable)
        → build_retry_prompt  (append-only feedback DATA + fixed mandate)
        → re-validate …       (bounded by total_attempts ≤ ATTEMPT_HARD_CAP)
   → ValidationOutcome (per-target resolutions + warnings)
```

The pipeline consumes only *internal* contracts. The canonical
`orchestration/schemas.py` boundary is reached exclusively through the existing
Step 1 adapter; the validation/retry core imports **only** the standard library
and sibling `corrector` modules (AST-verified). The `EntailmentScorer` seam is an
optional, injected `Protocol`: absent, it contributes nothing; present, it can only
*add* a failure, never overturn a deterministic gate.

## Validation gates

`GATE_ORDER` (fixed): `structure, authorization, target_identity,
original_preservation, numbers, dates, entities, unsupported_addition,
evidence_grounding, contradiction, minimal_edit, entailment`.
`SHORT_CIRCUIT_GATES` = the first three; a failure there stops the pipeline before
any semantic gate runs (so an unauthorized target is never measured against
evidence). Acceptance is **conjunctive**: a candidate passes only when *every* gate
is clear, i.e. zero failure codes. The safety-critical `evidence_grounding` gate is
terminal when no supporting evidence exists (no retry can conjure evidence) and
retryable when evidence exists but alignment is low.

The echo gate (gate 10, contradiction) implements the escalating ladder that the
recorded failure requires: exact match → whitespace/case → punctuation → claim-scoped
retained disputed values. This is why simply adding a period, changing case, or
swapping `48 hours` → `forty-eight hours` / `48 h` are all still rejected.

## Retry behavior (POLICY 2)

`max_retries = 2`, 3 total attempts, additionally hardened:

- **Bound enforced in code.** `total_attempts` clamps to `[1, ATTEMPT_HARD_CAP]`
  (=4); `max_retries=10_000` still yields 4, `max_retries=-5` yields 1. No config
  or env var can widen it.
- **Never an identical prompt / identical candidate.** Prompt and candidate
  fingerprints (SHA-256 of content) are tracked; a repeated prompt or a repeated
  candidate stops the loop and routes to UNRESOLVED — the correct response under
  greedy decoding, where an identical prompt provably returns an identical output.
- **Repeat-offense refusal.** A failure code seen on a prior attempt is not retried
  again (`REPEATED_FAILURE_CODE`).
- **Failure-specific fixed constraints.** Every retry carries constraints drawn
  only from the constant `RETRY_CONSTRAINTS` table; no applicable constraint means
  no retry (`NO_CONSTRAINT_AVAILABLE`).
- **No truncation.** A prompt over budget refuses (`BUDGET_EXCEEDED`) rather than
  being cut — the reference implementation's character-truncation could amputate
  the output mandate.
- Confirmed end-to-end: three disjoint retryable failures run the full 3-attempt
  budget and end UNRESOLVED with the original preserved; a retryable failure
  followed by a good candidate is ACCEPTED on attempt 2.

## Abstention behavior (POLICY 1)

A legitimate abstention (e.g. *"Current evidence is insufficient to support this
claim."*) is recognized (`is_abstention=True`), is **not** treated as accepted, is
**not** treated as unsupported/hallucinated content, and bypasses minimal-edit and
contradiction rejection. It is never retried (retrying is how a fabrication gets
produced) and its target routes to **UNRESOLVED** with the original preserved. No
new supervisor-level status was introduced — `REJECTED` and `UNRESOLVED` are
diagnostic distinctions that both mean "original preserved, no accepted text". A
smuggled abstention that also asserts an ungrounded value is *not* exempt.

## Security behavior

- **DATA vs INSTRUCTION separation.** Retry feedback is placed in a hashed
  `HG-DATA-<hex>` fence; the instruction half is assembled only from constant
  strings plus the plan's own `sentence_id`. No model output, evidence text, or
  derived reason can enter the instruction region.
- **Injection-shaped candidates are described, never quoted** — verified: an
  "ignore previous instructions…" candidate is withheld from the feedback DATA.
- **Runtime guards, not just convention:** `_instruction_leak` refuses if any
  substantial DATA line appears in the mandate; a fence-tag collision in model
  output refuses (`FENCE_COLLISION`) rather than repairing.
- **Fail-closed.** A gate that raises is caught and recorded as a terminal
  `VALIDATION_ERROR` (retryable=False) with no traceback or candidate text leaked
  into the detail — verified by monkeypatching a gate to raise. `require_entailment_check`
  with no scorer also fails closed. The pipeline never raises on adversarial input
  (empty, control chars, 5 000 chars, nested fences, injection, emoji).
- **No fabrication / no silent fallback.** `accepted_text` is non-empty only for an
  accepted candidate; an unavailable model performs no generation at all and every
  target keeps its original sentence.

## Tests executed

Run via `python3 agents/corrector_agent/tools/run_tests_stdlib.py` (corrector) and
the same shim path for the orchestration contract test.

| Suite | Tests | Scope |
|---|---:|---|
| test_adapter.py | 23 | Step 1–4 regression |
| test_targeting.py | 53 | Step 1–4 regression |
| test_evidence.py | 61 | Step 1–4 regression |
| test_prompt_model.py | 126 | Step 1–4 regression |
| test_step4a_contract_alignment.py | 9 | Step 1–4 regression |
| **test_validation.py** | **69** | **Step 5 — gates + pipeline + echo** |
| **test_retry.py** | **36** | **Step 5 — policy, bounds, prompt, security** |
| **test_step5_fuzz.py** | **44** | **Step 5 — property/fuzz + scope guards** |
| orchestration/test_canonical_contracts.py | 11 | Canonical boundary regression |

## Test results

- **Existing Corrector suite (Step 1–4 regression): 272 / 272 passed** (23 + 53 + 61
  + 126 + 9), matching the known baseline — Step 5 introduced no regression.
- **New Step 5 suite: 149 / 149 passed** (69 + 36 + 44).
- **Corrector total: 421 / 421 passed, 0 failed, 0 skipped.**
- Mandatory echo-failure test **passes** (`test_recorded_echo_is_rejected_with_evidence_contradiction`
  plus five non-verbatim evasions, and the correct rewrite passes).
- Abstention→UNRESOLVED **verified** (`test_abstention_resolves_unresolved_and_flags_abstained`).
- Retry bounds + identical-candidate termination **verified**
  (`test_three_disjoint_failures_exhaust_to_unresolved`,
  `test_identical_candidate_twice_stops_early`,
  `test_retry_never_exceeds_the_hard_cap_even_if_config_lies`).
- Fail-closed **verified** (`test_a_gate_that_raises_becomes_validation_error`).

*(All counts shim-executed; see the execution-honesty note above.)*

## Regression results

- `orchestration/tests/test_canonical_contracts.py`: **11 / 11 passed** under the
  shim — the canonical contract surface is intact.
- Protected-file checksums: **422 / 422 OK** against the pre-Step-5 baseline,
  including `orchestration/schemas.py`.
- The remaining orchestration suites (`test_graph_*`, `test_llm_detector_*`,
  `test_verifier_*`, `test_runtime_validation`, `test_base_llm_service`) exercise
  the LangGraph / Detector / Verifier / LLM-service machinery that Step 5 does not
  touch and were intentionally not run: they require heavy runtime dependencies
  (torch/transformers/langgraph) unavailable here and are outside the Step 5 change
  surface.

## Known limitations

- **Shim-only execution.** No real `pytest`/`pydantic` run has occurred. The
  pydantic shim supports the subset `orchestration/schemas.py` uses (types,
  `ge`/`le`, required, enums, unions, lists/tuples/dicts) but not custom validators
  (`schemas.py` has none) — so the canonical test is faithful here, but this is not
  a substitute for real pydantic v2 validation semantics.
- **Retry is out-of-distribution for the fine-tuned model.** Zero of the 426 train
  + 10 edge-case records contain retry/feedback prompts. The retry prompt is kept
  as close to the training shape as possible, but an out-of-distribution retry can
  only ever yield an UNRESOLVED target, never a worse *accepted* one.
- **Entailment seam is unused by default.** `require_entailment_check=False` and no
  scorer is wired in; the seam is exercised by tests with stub scorers only.
- **`correct()` is still the Step 1 passthrough.** Validation and retry are not yet
  connected to the supervisor-facing `CorrectionResult` (that is Step 6).

## Remaining issues

- None blocking within the Step 5 scope. Open items are the outstanding real
  `pytest` + `pydantic` run and the Step 6 wiring, both listed below.

## Step 6 prerequisites

1. Run the full suite under **real `pytest` + `pydantic` v2** and reconcile any
   divergence from the shim numbers before relying on them.
2. Wire `resolve_generation`'s `ValidationOutcome` back to the canonical
   `CorrectionResult` in `correct()` — mapping ACCEPTED → corrected text, and both
   REJECTED and UNRESOLVED → original preserved — using the existing adapter
   builders (no new supervisor-level contract).
3. Provide a real `CandidateSource` (the Step 4 `ModelClient.generate_for_prompt`
   already satisfies the Protocol structurally) so retries perform real generation.
4. Decide whether to enable the `EntailmentScorer` seam in production and, if so,
   inject a real scorer without importing `agents.verifier_agent.*` directly.
5. Reverification (the separate Step 6 concern) is deliberately **not** implemented
   here.

---

*Production readiness is not claimed. Passing tests under narrow shims is evidence
of correctness against the specified behaviors, not a certification for deployment.*
