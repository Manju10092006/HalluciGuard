# HalluciGuard Detector Agent - Interface & Integration Specification

**Version**: 2.0.0 (Production & Research Integration Release)  
**Author**: Principal Software Architect & AI Systems Engineer  

---

## Executive Overview

This document provides the definitive **Interface and Integration Specification** for the **HalluciGuard Detector Agent**. It specifies how external system components—specifically the **Verifier Agent**, **Judge Agent**, **Corrector Agent**, and **Memory Agent**—integrate with the Detector Agent without needing to inspect internal implementation details.

---

## Section 1 — Public API Specification

### Primary Entry Point: `DetectorAgent.detect()`

```python
from detector_agent.detector import DetectorAgent
from detector_agent.models import DetectionResult

agent = DetectorAgent()
result: DetectionResult = agent.detect(user_query: str, llm_response: str)
```

#### Inputs
- `user_query` (`str`): The user's input prompt or question. Must be a non-empty string.
- `llm_response` (`str`): The LLM candidate response text to evaluate. Must be a non-empty string.

#### Return Value
Returns a Pydantic `DetectionResult` instance containing:
- `confidence_score` (`float`): Overall confidence in $[0.0, 1.0]$. Higher values indicate factual reliability.
- `hallucination_probability` (`float`): $1.0 - \text{confidence\_score}$.
- `risk_level` (`RiskLevel`): Enum value (`RiskLevel.LOW`, `RiskLevel.MEDIUM`, `RiskLevel.HIGH`).
- `next_action` (`NextAction`): Enum value (`NextAction.ACCEPT`, `NextAction.VERIFY`).
- `metrics` (`SignalMetricsDetail`): Breakdown of Token Probability, Predictive Entropy, Semantic Similarity, and Self-Consistency.

---

## Section 2 — Request/Response Contracts

### HTTP REST Contract (`POST /detect`)

#### Request Schema
```json
{
  "user_query": "What is the capital of France?",
  "llm_response": "The capital of France is Paris."
}
```

#### Response Schema
```json
{
  "confidence_score": 0.9998,
  "hallucination_probability": 0.0002,
  "risk_level": "LOW",
  "next_action": "ACCEPT",
  "metrics": {
    "token_probability": {
      "avg_token_prob": 0.9995,
      "min_token_prob": 0.9850,
      "low_prob_tokens": []
    },
    "entropy": {
      "avg_entropy": 0.0042,
      "normalized_entropy": 0.0005
    },
    "semantic_similarity": {
      "similarity_score": 0.9982
    },
    "self_consistency": null
  },
  "metadata": {
    "query_category": "factual_question",
    "sc_executed": false
  }
}
```

---

## Section 3 — Integration Scenarios for External Agents

### 1. Verifier Agent Integration
The **Verifier Agent** uses `next_action` to decide whether to trigger costly web retrieval or external fact-checking:
```python
if detection.next_action == NextAction.VERIFY:
    trigger_web_search_verification(user_query)
```

### 2. Judge Agent Integration
The **Judge Agent** compares multiple candidate answers and picks the response with the maximum `confidence_score`.

### 3. Corrector Agent Integration
The **Corrector Agent** inspects `risk_level == RiskLevel.HIGH` and low-probability token spans to perform targeted rewrites.

### 4. Memory Agent Integration
The **Memory Agent** indexes `LOW` risk responses into vector storage and routes `HIGH` risk hallucinations into a blocklist.

---

## Section 4 — Error Handling & Edge Cases

| Exception / Edge Case | System Behavior | Mitigation / Fallback |
| :--- | :--- | :--- |
| `Empty/Null Query` | Raises `ValueError` | Input validation in API layer returning HTTP 400 Bad Request. |
| `Model Loading Failure` | Raises `RuntimeError` | `ModelManager` retries loading or falls back to CPU. |
| `GPU Out Of Memory` | CUDA Exception caught | Automatically falls back to CPU inference. |
| `Self-Consistency Gating Skip` | SC metric set to `null` | Weight redistributed to single-pass signals dynamically. |

---

## Section 5 — Performance Characteristics & Resource Requirements

- **Latency**:
  - Single-Pass Detection (SC Skipped): ~45ms - 80ms per query.
  - Multi-Pass Detection (SC Triggered): ~350ms - 600ms per query.
- **Memory Footprint**:
  - PyTorch Causal LM (`Qwen2.5-0.5B`): ~1.2 GB VRAM / RAM.
  - SentenceTransformer (`all-MiniLM-L6-v2`): ~120 MB RAM.

---

## Section 6 — Visual System Diagrams

See included files:
- [mermaid_sequence.mmd](file:///c:/Users/prane/Downloads/Halluciguard/detector_agent/mermaid_sequence.mmd)
- [mermaid_component.mmd](file:///c:/Users/prane/Downloads/Halluciguard/detector_agent/mermaid_component.mmd)

---

## Section 7 — Versioning & Backward Compatibility Policy

- **API Versioning**: `v2.0.0`
- **Backward Compatibility Guarantee**: All return fields (`confidence_score`, `hallucination_probability`, `risk_level`, `next_action`) will maintain stable JSON types and value ranges across future minor versions.
