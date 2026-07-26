# 🔍 Detector Agent

## Status: 🟡 Awaiting Implementation

## Role
The Detector Agent is the **first stage** in the HalluciGuard pipeline. It receives raw LLM output and identifies which claims are suspicious and need verification.

## Architecture Position
```
LLM Output → [DETECTOR AGENT] → Suspicious Claims → Verifier Agent → Judge Agent → ...
```

## Responsibilities
1. **Claim Extraction** — Parse LLM text into individual, atomic claims
2. **Suspicion Scoring** — Rate each claim's likelihood of being hallucinated using:
   - Perplexity analysis
   - Named entity density
   - Hedging language detection ("might", "could", "approximately")
   - Statistical claim detection
   - Citation absence detection
3. **Domain Classification** — Route each claim to the correct domain for the Verifier Agent
4. **Output Contract** — Generate `SuspiciousClaim` objects matching the schema in `agents/verifier_agent/schemas/models.py`

## Output Schema
```python
class SuspiciousClaim(BaseModel):
    claim_id: str
    text: str
    # Additional fields your implementation should add:
    # domain: str
    # suspicion_score: float (0.0 - 1.0)
    # detection_method: str
```

## Getting Started
1. Review the Verifier Agent's input contract: `agents/verifier_agent/schemas/models.py`
2. Study the architecture diagrams in the project root
3. Implement your agent in this directory
4. Create a PR to the `detector-agent` branch

## Tech Stack Suggestions
- **spaCy** for NLP/NER
- **transformers** for perplexity scoring
- **FastAPI** for the agent's HTTP interface (port 8001)

## Contact
Assigned to: [Your Name]
