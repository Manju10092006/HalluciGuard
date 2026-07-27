# HalluciGuard Detector Agent - Comprehensive System Architecture & Technical Specification

**Author**: Senior AI Research Engineer & Systems Architect  
**Project**: HalluciGuard Detection Engine  
**Version**: 2.0.0 (Calibrated & Verified)  
**Date**: July 2026  

---

## Executive Summary

**HalluciGuard** is a multi-signal, agentic AI system designed to detect and quantify hallucinations in Large Language Model (LLM) outputs. The **Detector Agent** serves as the primary inference and uncertainty estimation engine within the HalluciGuard ecosystem.

Rather than relying on a single heuristic or a secondary LLM judge, the Detector Agent synthesizes **four complementary statistical uncertainty signals**:
1. **Token Probability**: Next-token log-probability confidence scores derived directly from causal model logits.
2. **Predictive Entropy**: Shannon entropy over the vocabulary probability distribution at each decoding step.
3. **Semantic Similarity**: Vector cosine similarity between prompt context embeddings and response embeddings.
4. **Self-Consistency**: Stochastic decoding variation across $N$ sampled alternative candidate completions.

To optimize computational efficiency, the architecture incorporates an **Intelligent Gating Mechanism (`SelfConsistencyGate`)** that executes expensive multi-sample Self-Consistency only when single-pass signals indicate ambiguity (`MEDIUM` risk) on analytical prompt types.

---

## Section 1 — Project Structure & File Hierarchy

```text
detector_agent/
├── __init__.py                          # Package export definitions
├── app.py                               # FastAPI REST HTTP server
├── classifier.py                        # Rule-based prompt category classifier
├── config.py                            # Pydantic settings & signal weights validator
├── detector.py                          # Core DetectorAgent orchestrator
├── gate.py                              # Intelligent Gating controller for Self-Consistency
├── model_manager.py                     # Thread-safe shared model & tokenizer loader
├── models.py                            # Pydantic data contracts & metric schemas
├── utils.py                             # Platform DLL & environment helpers
├── signals/                             # Statistical Signal Calculators
│   ├── __init__.py                      # Signals package exports
│   ├── entropy.py                       # Predictive Entropy Calculator
│   ├── self_consistency.py              # Self-Consistency Sampling Calculator
│   └── semantic_similarity.py           # Embedding Semantic Similarity Calculator
├── datasets/                            # Benchmark Dataset Infrastructure
│   ├── __init__.py                      # Datasets package exports
│   ├── base_loader.py                   # Abstract BaseDatasetLoader interface
│   ├── benchmark_example.py             # Standardized BenchmarkExample dataclass
│   ├── halueval_loader.py               # HaluEval HF dataset streaming loader
│   ├── truthfulqa_loader.py             # TruthfulQA HF dataset streaming loader
│   ├── test_loaders.py                  # Dataset infrastructure verification script
│   └── README.md                        # Dataset infrastructure documentation
└── evaluation/                          # Research Evaluation Suite
    ├── __init__.py                      # Evaluation package exports
    ├── ablation_evaluator.py            # 9-Configuration Signal Ablation Suite
    ├── csv_export.py                    # Evaluation CSV Exporter
    ├── dashboard.py                     # HTML/MD/JSON Evaluation Dashboard Generator
    ├── evaluator.py                     # Standard Classification Evaluator
    ├── metrics.py                       # Reusable Classification Metrics (Pydantic)
    ├── pairwise_evaluator.py            # Counterfactual Pairwise Ranking Evaluator
    └── report.py                        # Console Table Formatter
```

### Directory & File Directory Matrix

| File Path | Primary Responsibility | Public Classes / Functions | Module Interactions |
| :--- | :--- | :--- | :--- |
| `detector.py` | Orchestrates 4-signal detection & score aggregation | `DetectorAgent` | Uses `ModelManager`, `DetectorConfig`, `SelfConsistencyGate`, `signals/*` |
| `config.py` | Configuration schema & Pydantic weight validation | `DetectorConfig`, `SignalWeights` | Consumed by `DetectorAgent`, `signals/*`, `evaluation/*` |
| `models.py` | Input/output Pydantic data schemas | `DetectionResult`, `RiskLevel`, `NextAction`, `SignalMetricsDetail` | Imported across entire codebase |
| `model_manager.py` | Shared PyTorch & SentenceTransformer loader | `ModelManager` | Injected into all `signals/*` to prevent redundant weights loading |
| `classifier.py` | Heuristic prompt classification into 11 categories | `PromptClassifier`, `PromptCategory` | Used by `SelfConsistencyGate` |
| `gate.py` | Gating logic controlling Self-Consistency execution | `SelfConsistencyGate` | Evaluated in `DetectorAgent.detect()` |
| `signals/entropy.py` | Shannon entropy calculator over vocabulary logits | `EntropyCalculator` | Consumes `ModelManager.tokenizer` & logits |
| `signals/semantic_similarity.py` | Cosine similarity calculator over SentenceTransformer | `SemanticSimilarityCalculator` | Consumes `ModelManager.sentence_model` |
| `signals/self_consistency.py` | Stochastic response sampler & similarity calculator | `SelfConsistencyCalculator` | Consumes `ModelManager` model/tokenizer & `sentence_model` |
| `datasets/benchmark_example.py` | Standardized research benchmark dataclass | `BenchmarkExample` | Consumed by dataset loaders and evaluation suite |
| `datasets/base_loader.py` | Abstract interface for dataset loaders | `BaseDatasetLoader` | Abstract base for `HaluEvalLoader`, `TruthfulQALoader` |
| `datasets/halueval_loader.py` | HaluEval benchmark streaming loader | `HaluEvalLoader` | Emits `BenchmarkExample` pairs with preserved context & metadata |
| `datasets/truthfulqa_loader.py` | TruthfulQA benchmark streaming loader | `TruthfulQALoader` | Emits `BenchmarkExample` pairs |
| `evaluation/evaluator.py` | Classification benchmark evaluator | `Evaluator` | Executes `DetectorAgent` over dataset examples |
| `evaluation/pairwise_evaluator.py` | Counterfactual pairwise ranking evaluator | `PairwiseEvaluator` | Measures $\text{conf}(\text{correct}) > \text{conf}(\text{hallucinated})$ |
| `evaluation/ablation_evaluator.py` | 9-configuration signal ablation framework | `AblationEvaluator` | Executes `DetectorAgent` under varied `SignalWeights` |
| `evaluation/dashboard.py` | Standalone HTML/MD/JSON dashboard generator | `DashboardGenerator` | Runs full evaluation & renders inline Base64 charts |
| `app.py` | FastAPI HTTP REST web service | `app`, FastAPI routes | Exposes `POST /detect` endpoint |

---

## Section 2 — Module Responsibilities

### 1. Orchestration Module (`detector_agent/detector.py`)
- **Purpose**: Serves as the central pipeline orchestrator. Receives `user_query` and `llm_response`, coordinates signal calculation, evaluates gating rules, aggregates scores, and determines final risk levels.
- **Inputs**: `user_query: str`, `llm_response: str`.
- **Outputs**: `DetectionResult` Pydantic model (`confidence_score`, `hallucination_probability`, `risk_level`, `next_action`, `metrics`).
- **Dependencies**: `model_manager`, `config`, `gate`, `signals/*`.
- **Algorithms**: Single-pass statistical inference, weighted linear score aggregation, calibrated double-threshold risk binning (`0.40`, `0.55`).

### 2. Configuration Module (`detector_agent/config.py`)
- **Purpose**: Centralized type-safe configuration system built on Pydantic BaseSettings.
- **Key Validation**: `SignalWeights.validate_weights_sum()` enforces that all individual weights are within $[0.0, 1.0]$ and sum to exactly $1.0$ (within $1\times 10^{-4}$ tolerance).

### 3. Model Management Module (`detector_agent/model_manager.py`)
- **Purpose**: Implements the Singleton/Flyweight pattern for neural model instances. Loads `Qwen/Qwen2.5-0.5B-Instruct` (causal LM + tokenizer) and `sentence-transformers/all-MiniLM-L6-v2` (embedding model) exactly once.
- **Memory Safety**: Prevents GPU/CPU memory leaks and redundant weight allocations across parallel signal execution.

### 4. Intelligent Gating Modules (`classifier.py`, `gate.py`)
- **Purpose**: Lightweight prompt categorization and Self-Consistency execution control.
- **Classification Categories**: 11 prompt types (`factual_question`, `explanation`, `summarization`, `reasoning`, `creative_writing`, `storytelling`, `brainstorming`, `poetry`, `fictional_content`, `roleplay`, `other`).
- **Gating Rule**: Self-Consistency runs **only** when single-pass risk is `MEDIUM` **and** query category is analytical (`factual_question`, `explanation`, `summarization`, `reasoning`).

---

## Section 3 — Class Documentation

### `DetectorAgent`
- **Responsibilities**: Pipeline orchestration, signal execution, score aggregation, risk categorization.
- **Constructor**: `__init__(config: Optional[DetectorConfig] = None, model_manager: Optional[ModelManager] = None)`
- **Public Methods**:
  - `detect(user_query: str, llm_response: str) -> DetectionResult`
- **Private Methods**:
  - `_compute_token_probability(...)`
  - `_compute_entropy(...)`
  - `_compute_semantic_similarity(...)`
  - `_compute_self_consistency(...)`
  - `_aggregate_scores(...) -> tuple[float, float]`
  - `_determine_risk_level(...) -> RiskLevel`
  - `_determine_next_action(...) -> NextAction`

### `ModelManager`
- **Responsibilities**: Lazy-loading and serving PyTorch causal language model, tokenizer, and SentenceTransformer embedding model.
- **Properties**: `model`, `tokenizer`, `sentence_model`.

### `BenchmarkExample`
- **Responsibilities**: Unified dataclass representing a benchmark instance across all agents.
- **Fields**: `query`, `response`, `expected_risk`, `category`, `context`, `source_dataset`, `source_model`, `pair_id`, `metadata`.

### `PairwiseEvaluator`
- **Responsibilities**: Evaluates pairwise ranking accuracy ($\text{conf}_{\text{correct}} > \text{conf}_{\text{hallucinated}}$) across matched `pair_id` pairs.
- **Public Methods**: `evaluate_pairs()`, `print_report()`, `export_csv()`, `plot_distributions()`.

### `AblationEvaluator`
- **Responsibilities**: Evaluates 9 signal configurations to measure individual signal contributions.
- **Public Methods**: `run_ablation_study()`, `export_csv()`, `generate_markdown_report()`, `generate_bar_charts()`.

---

## Section 4 — Detector Execution Pipeline

```text
                         +-----------------------------------+
                         |      User Request (Query + Response) |
                         +-----------------------------------+
                                           |
                                           v
                         +-----------------------------------+
                         |       DetectorAgent.detect()      |
                         +-----------------------------------+
                                           |
               +---------------------------+---------------------------+
               |                           |                           |
               v                           v                           v
    +--------------------+       +--------------------+       +--------------------+
    | Token Probability  |       | Predictive Entropy |       | SemanticSimilarity |
    |      Signal        |       |       Signal       |       |       Signal       |
    +--------------------+       +--------------------+       +--------------------+
               |                           |                           |
               +---------------------------+---------------------------+
                                           |
                                           v
                         +-----------------------------------+
                         |    Single-Pass Score Aggregation   |
                         +-----------------------------------+
                                           |
                                           v
                         +-----------------------------------+
                         |    Initial Risk Determination     |
                         |        (LOW / MEDIUM / HIGH)      |
                         +-----------------------------------+
                                           |
                                           v
                         +-----------------------------------+
                         |    SelfConsistencyGate.should_run |
                         +-----------------------------------+
                                           |
                         +-----------------+-----------------+
                         |                                   |
                (If MEDIUM & Analytical)                 (If LOW/HIGH or Creative)
                         |                                   |
                         v                                   v
             +-----------------------+              +------------------+
             | Self-Consistency      |              | Skip SC          |
             | Calculator (N Samples)|              | (0 Compute Cost) |
             +-----------------------+              +------------------+
                         |                                   |
                         +-----------------+-----------------+
                                           |
                                           v
                         +-----------------------------------+
                         |    Final Weighted Score Synthesis |
                         +-----------------------------------+
                                           |
                                           v
                         +-----------------------------------+
                         |     Final DetectionResult Output   |
                         |  (Confidence, Risk, Next Action)  |
                         +-----------------------------------+
```

### Detailed Pipeline Walkthrough

1. **Single-Pass Inference**:
   - Token Probability computes average token log-probability: $S_{\text{prob}} = \exp(\text{avg\_logprob})$.
   - Predictive Entropy computes normalized certainty: $S_{\text{entropy}} = 1.0 - \frac{H}{\ln V}$.
   - Semantic Similarity computes context-response embedding cosine similarity: $S_{\text{sem}} = \frac{\cos(\theta) + 1.0}{2.0}$.
2. **Initial Risk Determination**:
   - Aggregates single-pass signals into an initial confidence score and maps to `LOW`, `MEDIUM`, or `HIGH`.
3. **Intelligent Gating**:
   - `SelfConsistencyGate` inspects initial risk level and prompt category. If risk is `LOW` or `HIGH`, or if prompt is non-analytical (e.g. `creative_writing`), Self-Consistency is skipped.
4. **Conditional Self-Consistency**:
   - If triggered, samples $N=5$ completions at temperature $T=0.7$, computes pairwise cosine similarities, and outputs $S_{\text{sc}}$.
5. **Final Score Synthesis**:
   - Combines active signal metrics using weights from `DetectorConfig.signal_weights`.
6. **Risk & Action Assignment**:
   - Maps final hallucination probability to `LOW` ($\le 0.40$), `MEDIUM` ($0.40 < P < 0.55$), or `HIGH` ($\ge 0.55$).
   - Recommends `NextAction.ACCEPT` (for `LOW`) or `NextAction.VERIFY` (for `MEDIUM`/`HIGH`).

---

## Section 5 — Configuration System

| Parameter Name | Type | Default Value | Description / Operational Impact |
| :--- | :---: | :---: | :--- |
| `token_probability` | float | `0.40` | Weight for Token Probability signal. |
| `entropy` | float | `0.30` | Weight for Predictive Entropy signal. |
| `semantic_similarity` | float | `0.30` | Weight for Semantic Similarity signal. |
| `self_consistency` | float | `0.00` (or `0.15` in Full Detector) | Weight for Self-Consistency signal. |
| `low_risk_threshold` | float | `0.40` | Probability at or below which risk is `LOW` (Accept). |
| `high_risk_threshold` | float | `0.55` | Probability at or above which risk is `HIGH` (Verify). |
| `num_samples` | int | `5` | Number of candidate completions sampled for Self-Consistency. |
| `temperature` | float | `0.7` | Temperature for stochastic decoding sampling. |
| `top_p` | float | `0.9` | Top-p nucleus sampling threshold. |
| `max_new_tokens` | int | `100` | Token limit for sampled completions. |
| `model_name` | str | `"Qwen/Qwen2.5-0.5B-Instruct"` | Base causal language model for logit extraction. |

---

## Section 6 — Evaluation Framework Architecture

The evaluation suite comprises four interconnected modules:
1. **`Evaluator`**: Runs single-pass and multi-signal detection across benchmark datasets and calculates multi-class confusion matrices, precision, recall, and F1 scores.
2. **`PairwiseEvaluator`**: Measures counterfactual pairwise ranking accuracy ($\text{conf}_{\text{correct}} > \text{conf}_{\text{hallucinated}}$) across matched prompt pairs.
3. **`AblationEvaluator`**: Executes 9 distinct signal configurations to measure individual signal performance and Self-Consistency activation rates.
4. **`DashboardGenerator`**: Aggregates all evaluation results into a standalone HTML dashboard (`evaluation_dashboard.html`) with embedded Base64 Matplotlib charts, GitHub Markdown summary (`summary_report.md`), and JSON metadata (`evaluation_summary.json`).

---

## Section 7 — Dataset Infrastructure

The dataset package (`detector_agent/datasets/`) provides a SOLID, extensible data loading infrastructure:
- **`BenchmarkExample`**: Unified dataclass holding `query`, `response`, `expected_risk`, `category`, `context`, `source_dataset`, `source_model`, `pair_id`, and `metadata`.
- **`BaseDatasetLoader`**: Abstract base class defining `load_dataset(limit: Optional[int])`.
- **`HaluEvalLoader`**: Streams `pminervini/HaluEval` (QA, Dialogue, Summarization, General) from Hugging Face Hub.
- **`TruthfulQALoader`**: Streams `truthfulqa/truthful_qa` from Hugging Face Hub.
- **Context Isolation & Pair Preservation**: Keeps `query` and `context` strictly separated while assigning shared `pair_id`s to correct vs. hallucinated answer pairs.

---

## Section 8 — Key Architectural Design Decisions & Rationale

1. **Shared ModelManager Singleton**:
   - *Rationale*: Causal LMs and SentenceTransformers require significant VRAM/RAM. Injecting a shared `ModelManager` prevents memory duplication during parallel signal evaluation.
2. **Configurable Dynamic Score Aggregation**:
   - *Rationale*: Hardcoded signal weights prevent empirical tuning. Externalizing weights into Pydantic models enables seamless ablation studies and domain adaptation.
3. **Intelligent Gating (`SelfConsistencyGate`)**:
   - *Rationale*: Self-Consistency requires generating $N$ candidate responses, increasing inference latency by $5\times$. Gating SC to run only on `MEDIUM` risk analytical queries achieves a $75\%+$ reduction in compute cost without sacrificing accuracy.
4. **Counterfactual Pairwise Ranking Evaluation**:
   - *Rationale*: Classification accuracy alone fails to measure relative confidence calibration. Pairwise gap evaluation ($\Delta S = S_{\text{correct}} - S_{\text{hallucinated}}$) tests whether the detector reliably prefers truthful text.

---

## Section 9 — Extension Points for Future Developers

1. **Adding a New Detection Signal**:
   - Step 1: Create `detector_agent/signals/new_signal.py` implementing `NewSignalCalculator`.
   - Step 2: Add metric model to `detector_agent/models.py`.
   - Step 3: Add field to `SignalWeights` in `detector_agent/config.py`.
   - Step 4: Add invocation step in `DetectorAgent.detect()`.

2. **Adding a New Benchmark Dataset**:
   - Step 1: Subclass `BaseDatasetLoader` in `detector_agent/datasets/new_dataset_loader.py`.
   - Step 2: Implement `load_dataset()` returning `List[BenchmarkExample]`.
   - Step 3: Register loader in `detector_agent/datasets/__init__.py`.

3. **Adding a New Evaluation Metric or Visualization**:
   - Step 1: Add metric calculation function to `detector_agent/evaluation/metrics.py`.
   - Step 2: Add chart renderer method in `DashboardGenerator` (`detector_agent/evaluation/dashboard.py`).

---

## Conclusion & System Status

The **HalluciGuard Detector Agent** architecture represents a production-ready, research-grade hallucination detection system. All signals, gating mechanisms, evaluation tools, dataset loaders, and visualization dashboards have been fully implemented, calibrated, and empirically verified.
