# ✏️ Corrector Agent

## Status: 🟡 Awaiting Implementation

## Role
The Corrector Agent is the **fact-repair engine** in the HalluciGuard pipeline. It takes rejected claims and rewrites them using verified evidence.

## Architecture Position
```
... → Judge Agent → Rejected Claims → [CORRECTOR AGENT] → Corrected Text → Memory Agent
```

## Responsibilities
1. **Evidence-Based Rewriting** — Replace hallucinated claims with verified facts
2. **Citation Injection** — Add inline source references
3. **Tone Preservation** — Maintain the original writing style
4. **Change Logging** — Track what was changed and why
5. **Minimal Intervention** — Only modify claims that were rejected; preserve accepted content

## Getting Started
1. Review the Judge Agent's output contract
2. Access the Verifier Agent's evidence cache
3. Implement your agent in this directory
4. Create a PR to the `corrector-agent` branch

## Tech Stack Suggestions
- **LLM API** (OpenAI/Gemini) for text rewriting
- **difflib** for change tracking
- **FastAPI** for HTTP interface (port 8004)

## Contact
Assigned to: [Your Name]
