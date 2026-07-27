# HalluciGuard - Research-Grade Benchmark Dataset Infrastructure

This package (`detector_agent/datasets`) provides a modular, extensible, research-grade dataset loading infrastructure for evaluating hallucination detection across multi-agent pipelines.

---

## Architectural Principles & Design Rationale

### 1. Separate `context` Storage
- **Why**: Prompts (`query`) and supporting background knowledge/documents (`context`) are kept strictly separated rather than concatenated into a single string.
- **Agent Benefit**: 
  - **Detector Agent**: Evaluates intrinsic query-response uncertainty and semantic alignment without prompt bloat.
  - **Verifier Agent**: Directly compares `context` (retrieved knowledge or source documents) against `response` to detect factual ungroundedness or contradictions.
  - **Correction Agent**: Uses isolated `context` to ground response rewriting.

### 2. Pair Preservation (`pair_id`)
- **Why**: Benchmark datasets like HaluEval and TruthfulQA contain paired factual (`LOW` risk) vs hallucinated (`HIGH` risk) responses for identical prompts/documents. Sharing a unique `pair_id` (e.g. `QA_000001`, `SUM_000184`) preserves this pairing.
- **Agent & Experiment Benefit**:
  - **Pairwise Ranking Evaluation**: Enables computing Delta Confidence $\Delta S = S_{\text{correct}} - S_{\text{hallucinated}}$ for counterfactual sensitivity testing.
  - **Judge Agent**: Evaluates relative preference and ranking accuracy between paired generations.

### 3. Preservation of `source_dataset` & `source_model`
- **Why**: Preserving provenance metadata (`source_dataset="HaluEval"`, `source_model="ChatGPT"`) tracks dataset source and generating LLM architecture.
- **Research Benefit**:
  - Enables cross-model hallucination profile analysis (comparing hallucination rates of ChatGPT vs. Open-Source Llama-3).
  - Allows filtering and benchmarking per dataset domain (`HaluEval` vs `TruthfulQA`).

### 4. Original Labels & Spans Retention
- **Why**: Original raw labels (`metadata["original_label"]`) and fine-grained character/token spans (`metadata["hallucination_spans"]`) are preserved without loss of dataset fidelity.
- **Future Agent Benefit**:
  - **Verifier & Judge Agents**: Perform token-level and phrase-level hallucination span attribution, highlighting exact text segments where hallucinations occur.

---

## Dataclass Schema (`BenchmarkExample`)

```python
@dataclass
class BenchmarkExample:
    query: str
    response: str
    expected_risk: str                 # "LOW", "MEDIUM", or "HIGH"
    category: str
    context: Optional[str] = None      # Isolated supporting knowledge or source document
    source_dataset: str = "HaluEval"   # Provenance dataset ("HaluEval", "TruthfulQA")
    source_model: Optional[str] = None # Generating model ("ChatGPT", "gpt-3.5-turbo")
    pair_id: Optional[str] = None      # Shared pairing ID (e.g. "QA_000001")
    metadata: Optional[dict] = None    # Preserves original_label, hallucination_spans, etc.
```

---

## Usage Example

```python
from detector_agent.datasets import HaluEvalLoader, TruthfulQALoader

# Load HaluEval QA samples
halueval_loader = HaluEvalLoader(subset="qa")
qa_examples = halueval_loader.load_dataset(limit=10)

# Load TruthfulQA Generation samples
truthfulqa_loader = TruthfulQALoader(subset="generation")
tqa_examples = truthfulqa_loader.load_dataset(limit=10)
```
