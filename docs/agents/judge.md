# Judge Agent

`agents/judge_agent/judge_agent.py` converts Detector risk, Verifier claim reports, domain policy, retry state and optional ReVerifier state into a workflow decision.

| Decision | Meaning |
|---|---|
| `ACCEPT` | Release is permitted because core claims are grounded under policy |
| `CORRECT` | One or more contradicted claims receive a scoped `CorrectionRequest` |
| `VERIFY_AGAIN` | Evidence is incomplete/conflicted and retry budget remains |
| `REJECT` | Correction failed or critical unresolved risk cannot be released |
| `ABSTAIN` | The system cannot establish a safe result; route to human review |

The Judge treats Detector output as triage, never evidence. It does not retrieve sources, invent facts, or rewrite content. A `CORRECT` result contains explicit claims to correct, claims to preserve, trusted evidence, contradictory evidence and instructions. Reverification has its own precedence: passed corrections may be accepted; remaining contradictions may retry; exhausted repair is rejected; failed reverification abstains.
