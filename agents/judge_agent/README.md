# ⚖️ HalluciGuard Judge Agent

> **Workflow / policy decision layer of the HalluciGuard trust pipeline.**

The Judge answers a different question from the Detector and Verifier:

> **"Given the Verifier's factual findings, system health, domain policy and risk, what should HalluciGuard do next?"**

It is the **decision/governance layer** — not a search engine and not a
hallucination detector. It does **not** perform its own fact-checking, NLI
inference, or keyword refutation. The Verifier is the factual authority; the
Judge arbitrates workflow.

---

## ✅ Canonical implementation

The **active** Judge is a single, auditable module:

```text
agents/judge_agent/judge_agent.py   → class JudgeAgent.evaluate(...)
```

It is wired into the production graph at `orchestration/graph.py::_judge_node`.
Everything else in this directory (`decision_intelligence.py`,
`workflow_orchestrator.py`, `evidence_governance.py`, `coverage_analyzer.py`,
`nli_engine.py`, …) is **legacy / not on the active path** and is retained only
for reference. Those modules are **deprecated**; do not import them into the
graph. There is no `ESCALATE_HUMAN` decision — routing to human review is a
graph-level *route*, not a Judge decision.

---

## 🧭 Decision model

Canonical decisions (`orchestration.schemas.JudgeDecision`):

| Decision       | Meaning                                                        |
|----------------|----------------------------------------------------------------|
| `ACCEPT`       | Sufficient **verified** basis to release the draft verbatim.   |
| `CORRECT`      | Fixable contradiction(s) → hand a `CorrectionRequest` to the Corrector. |
| `VERIFY_AGAIN` | Grounding gap, retries remain → request another verification pass. |
| `REJECT`       | Post-correction re-verification failed after the retry budget. |
| `ABSTAIN`      | Insufficient evidence / degraded pipeline → withhold (fail-closed). |

### Hard invariants

- **UNVERIFIED ≠ VERIFIED.** Absence of a contradiction is *not* proof of truth.
- **detector-risk ≠ factual-verdict.** A high Detector probability is a triage
  prior only. Zero-evidence + high Detector → `ABSTAIN`, never `REJECT`.
- **claim-count ≠ decision.** ACCEPT is gated by *criticality*, not by whether
  verified claims outnumber unverified ones.
- **re-verification failure ≠ success.** ACCEPT after correction requires
  `passed AND remaining_contradictions == 0`.
- **ACCEPT = sufficient verified basis for release** — never a rubber stamp.

### Criticality gate (deterministic, query-aware)

Each claim is classified **CORE** (its salient terms overlap the user query) or
**PERIPHERAL** (incidental detail the user did not ask about). Classification is
pure string logic — no model, no network — and **fail-closed**: a degenerate or
empty query makes every claim CORE.

> **The pipeline never ACCEPTs while a CORE claim is UNVERIFIED / CONFLICTED, or
> while any claim is CONTRADICTED.** Only a PERIPHERAL unverified fragment
> (e.g. a decomposition artifact) may be tolerated, and only in
> `MODERATE`/`RELAXED` domains.

### Precedence (first match wins)

1. Post-correction re-verification present → `_evaluate_reverification` (gate above).
2. Invalid / failed Verifier payload → `ABSTAIN`.
3. Any contradiction → `CORRECT` (correct-first; strict domains gated by re-verification).
4. Zero claims → `VERIFY_AGAIN` (retries left) else `ABSTAIN`.
5. All claims verified → `ACCEPT`.
6. Verified CORE + only PERIPHERAL unverified, MODERATE/RELAXED → `ACCEPT`.
7. Any remaining unverified/conflicted (incl. any CORE gap) → `VERIFY_AGAIN`
   (retries left) else `ABSTAIN`.

### `decision_basis` reason codes

Every `JudgeResult` carries a stable, machine-readable `decision_basis` naming
the rule that fired (`CONTRADICTION_PRESENT_CORRECT`, `ALL_CLAIMS_VERIFIED_ACCEPT`,
`PERIPHERAL_UNVERIFIED_TOLERATED_ACCEPT`, `CORE_UNVERIFIED_RETRY`,
`CORE_UNVERIFIED_ABSTAIN`, `NO_EVIDENCE_ABSTAIN`, `REVERIFICATION_PASSED_ACCEPT`,
`REVERIFICATION_FAILED_REJECT`, …). It is for metrics/audit, never a factual
claim about the response. An `ACCEPT` emitted with a core grounding gap is logged
as `false_accept_suspect` (by construction the tree cannot produce it — the log
makes any regression loud).

---

## 🧪 Tests

```text
agents/judge_agent/tests/test_judge_defects.py
```

Regression + invariant fixtures: Lamborghini all-unverified → `ABSTAIN`
(never `ACCEPT`); Russia count-tie → not auto-`ACCEPT`; Vietnam re-verification
`passed=False` → not `ACCEPT`; empty verifier + high detector → `ABSTAIN`
(never `REJECT`); verified-core + peripheral-unverified → `ACCEPT`; contradiction
→ `CORRECT`; all-verified → `ACCEPT`.

```bash
python -m pytest agents/judge_agent/tests -q
```

---

## 🚧 Status (honest)

- **Deterministic decision model:** ✅ implemented, tested, wired into the graph.
- **Criticality gate + `decision_basis` observability:** ✅ implemented.
- **Optional LLM-judge layer for borderline cases:** ❌ not implemented. The
  current arbitration is fully deterministic; there is no LLM in the Judge path.
- **Legacy `decision_intelligence.py` subsystem:** 🟡 deprecated, not deleted.
- **70-case decision benchmark:** ❌ not built. Coverage is the fixture set above.

---

## 🔗 Related

- [Root HalluciGuard README](../../README.md)
- [Verifier Agent](../verifier_agent/README.md)
- [Corrector Agent](../corrector_agent/README.md)
- [LangGraph Orchestration](../../orchestration/README.md)
