# ⚖️ Judge Agent

## Status: 🟡 Awaiting Implementation

## Role
The Judge Agent is the **decision-maker** in the HalluciGuard pipeline. It receives evidence from the Verifier Agent and makes final accept/reject/flag decisions.

## Architecture Position
```
... → Verifier Agent → Evidence Reports → [JUDGE AGENT] → Final Verdicts → Corrector Agent
```

## Responsibilities
1. **Threshold-Based Decision** — Apply domain-specific risk thresholds
   - Healthcare: High bar (trust_score > 0.8 to accept)
   - General: Lower bar (trust_score > 0.5 to accept)
2. **Multi-Evidence Reasoning** — Weigh conflicting evidence
3. **Confidence Calibration** — Ensure decision confidence matches evidence strength
4. **Verdict Categories**:
   - `ACCEPT` — Claim is well-supported by evidence
   - `REJECT` — Claim contradicts authoritative sources
   - `FLAG_FOR_REVIEW` — Insufficient or conflicting evidence
5. **Audit Trail** — Generate detailed reasoning for each decision

## Input Schema
See `agents/verifier_agent/schemas/models.py` → `VerifierOutputV2`

## Getting Started
1. Review the Verifier Agent's output contract
2. Define domain risk profiles
3. Implement your agent in this directory
4. Create a PR to the `judge-agent` branch

## Tech Stack Suggestions
- **Pydantic** for schema validation
- **FastAPI** for HTTP interface (port 8003)
- Decision trees or rule-based engines

## Contact
Assigned to: [Your Name]
