# HalluciGuard - Detector Agent

The **Detector Agent** is the entry point agent in the [HalluciGuard](https://github.com/halluciguard) multi-agent framework for hallucination mitigation in Large Language Models (LLMs).

## Responsibilities

### What it DOES:
- Receives a `user_query` and an `llm_response`.
- Estimates a `confidence_score` (0.0 to 1.0) and `hallucination_probability` (0.0 to 1.0).
- Categorizes risk into a `risk_level` (`LOW`, `MEDIUM`, `HIGH`).
- Recommends a `next_action` (`Accept` or `Verify`).

### What it DOES NOT do:
- Verify facts against external sources.
- Retrieve external evidence or documents.
- Correct or rewrite LLM responses.
- Make final content moderation decisions.

*(Those responsibilities belong to downstream verification, retrieval, and correction agents).*

---

## Directory Structure

```
detector_agent/
│
├── __init__.py        # Package exports (DetectorAgent, models, config)
├── detector.py        # Core DetectorAgent class logic
├── models.py          # Pydantic V2 data models & Enums (Input, Output, RiskLevel, NextAction)
├── config.py          # Configurable threshold settings (pydantic-settings)
├── app.py             # FastAPI web application providing POST /detect
├── requirements.txt   # Module dependencies
└── README.md          # Technical documentation
```

---

## Installation & Setup

Requires **Python 3.11+**.

```bash
# Navigate to project directory
cd detector_agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Option 1: Direct Python Module Usage

```python
from detector_agent import DetectorAgent

agent = DetectorAgent()

query = "What is the capital of France?"
response = "The capital of France is Paris."

result = agent.detect(user_query=query, llm_response=response)

print(result.model_dump_json(indent=2))
```

**Output:**
```json
{
  "confidence_score": 0.95,
  "hallucination_probability": 0.05,
  "risk_level": "LOW",
  "next_action": "Accept"
}
```

---

### Option 2: Running the FastAPI Web API

Start the server using `uvicorn`:

```bash
uvicorn detector_agent.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

#### Example Request (`POST /detect`)

```bash
curl -X POST "http://localhost:8000/detect" \
     -H "Content-Type: application/json" \
     -d '{
           "user_query": "Who walked on the moon first?",
           "llm_response": "Neil Armstrong was the first person to walk on the Moon in 1969."
         }'
```

#### Response Payload

```json
{
  "confidence_score": 0.95,
  "hallucination_probability": 0.05,
  "risk_level": "LOW",
  "next_action": "Accept"
}
```

---

## Extension Guide for Phase 2 Detection Signals

In Phase 1, `DetectorAgent` returns configured baseline scores.

Phase 2 will integrate detection signals into `DetectorAgent`:
- **Token Probability**: Log-probability analysis of generated tokens.
- **Predictive Entropy**: Token distribution uncertainty estimation.
- **Semantic Similarity**: Semantic drift against prompt embeddings.
- **Self-Consistency**: Agreement scoring across sampled candidate responses.

Private extension methods are predefined in `detector.py`:
- `_compute_token_probability()`
- `_compute_entropy()`
- `_compute_semantic_similarity()`
- `_compute_self_consistency()`

When implementing Phase 2 features, implement these helper methods or import specialized sub-modules and integrate their output in `_aggregate_scores()`.

---

## License

Part of the **HalluciGuard** open-source ecosystem.
