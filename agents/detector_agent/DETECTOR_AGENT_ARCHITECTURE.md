# HalluciGuard Detector Agent — FAANG-Level System Architecture & Technical Specification

---

## 1. Executive Summary & Architectural Role

The **Detector Agent** serves as the **Stage 1 Early Warning & Gating Engine** of the HalluciGuard platform. It receives a user prompt ($Q$) and an LLM-generated output ($R$), and evaluates the likelihood of hallucination using internal neural logit geometry, sequence entropy, semantic alignment, and conditional stochastic self-consistency.

Rather than running expensive external verification on every request, the Detector Agent operates as a **high-throughput, compute-adaptive filter**. If a response is evaluated as `LOW` risk, it is immediately marked as `Accept` (bypassing downstream retrieval). If the risk is `MEDIUM` or `HIGH`, it triggers the `Verify` action, handing off structured suspicious claim payloads to the **Verifier Agent**.

```
+----------------------------------------------------------------------------------------------------+
|                                    HALLUCIGUARD DETECTOR AGENT                                     |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [User Prompt Q + LLM Response R]                                                                  |
|               |                                                                                    |
|               v                                                                                    |
|  +----------------------------------------------------------------------------------------------+  |
|  | STEP 1: SINGLE-PASS STATISTICAL SIGNALS                                                       |  |
|  |  - Token Log Probability (avg_logprob, min_logprob, low_confidence_ratio)                      |  |
|  |  - Predictive Token Entropy (H(X) normalized by max theoretical log|V|)                         |  |
|  |  - Semantic Similarity (Prompt-Response Embedding Cosine Distance)                             |  |
|  +----------------------------------------------------------------------------------------------+  |
|               |                                                                                    |
|               v                                                                                    |
|  +----------------------------------------------------------------------------------------------+  |
|  | STEP 2: INITIAL SCORE AGGREGATION & SINGLE-PASS RISK CLASSIFICATION                           |  |
|  |  Confidence = w1*Prob + w2*(1-Entropy) + w3*SemanticSim  -->  Single-Pass Risk (LOW/MED/HIGH)  |  |
|  +----------------------------------------------------------------------------------------------+  |
|               |                                                                                    |
|               v                                                                                    |
|  +----------------------------------------------------------------------------------------------+  |
|  | STEP 3: INTELLIGENT GATING ENGINE (SelfConsistencyGate)                                      |  |
|  |  Condition A: Is Single-Pass Risk == MEDIUM?                                                 |  |
|  |  Condition B: Is Query Intent Analytical (Factual / Explanation / Summarization / Reasoning)?  |  |
|  +----------------------------------------------------------------------------------------------+  |
|         |                                                |                                         |
|    [NO] (Skip Compute)                              [YES] (Execute Sampling)                       |
|         |                                                |                                         |
|         v                                                v                                         |
|         |                                 +--------------------------------------------+           |
|         |                                 | STEP 4: SELF-CONSISTENCY SAMPLING (N=5)    |           |
|         |                                 |  - Stochastic Sampling (T=0.7, top_p=0.9)  |           |
|         |                                 |  - Pairwise Cosine Similarity Matrix       |           |
|         |                                 +--------------------------------------------+           |
|         |                                                |                                         |
|         +-----------------------+------------------------+                                         |
|                                 |                                                                  |
|                                 v                                                                  |
|  +----------------------------------------------------------------------------------------------+  |
|  | STEP 5 & 6: FINAL SCORE SYNTHESIS & PIPELINE DECISION                                        |  |
|  |  Final Hallucination Probability  -->  Risk Level (LOW / MEDIUM / HIGH)                         |  |
|  |  Next Action: [ACCEPT] (if LOW)  or  [VERIFY] (if MEDIUM/HIGH -> Send to Verifier Agent)     |  |
|  +----------------------------------------------------------------------------------------------+  |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Technical Signal Suite Breakdown

The Detector Agent calculates four distinct statistical and semantic signals:

### 2.1 Token Probability Signal (`_compute_token_probability`)
Extracts internal model logit confidence across generated tokens:
- **Log Probability Extraction**: For each generated token $t_i$:
  $$\log P(t_i \mid t_{<i}, Q) = \text{log\_softmax}(\mathbf{z}_i)[t_i]$$
- **Metrics**:
  - `avg_logprob`: Mean token log probability across sequence.
  - `min_logprob`: Worst-case confidence dip in sequence.
  - `low_confidence_ratio`: Proportion of tokens with $\log P < -2.0$ ($P < 0.135$).

### 2.2 Predictive Token Entropy Signal (`EntropyCalculator`)
Measures model uncertainty from the vocabulary logit distribution at each generation step:
- **Shannon Entropy**:
  $$H(t_i) = -\sum_{v \in V} P(v) \log P(v)$$
- **Theoretical Normalization**:
  $$\text{NormalizedEntropy} = \frac{\bar{H}}{\log(|V|)}$$
  where $|V|$ is the model vocabulary size (e.g. 151,646 for Qwen2.5).

### 2.3 Semantic Similarity Signal (`SemanticSimilarityCalculator`)
Measures prompt-response semantic coherence using dense embeddings:
- **Embedding Projection**: Encodes $Q$ and $R$ using `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-m3`.
- **Cosine Distance**:
  $$\text{SemanticSim}(Q, R) = \frac{\mathbf{e}_Q \cdot \mathbf{e}_R}{\|\mathbf{e}_Q\|_2 \|\mathbf{e}_R\|_2}$$

### 2.4 Self-Consistency Signal (`SelfConsistencyCalculator`)
Evaluates semantic consensus across $N=5$ stochastically sampled candidate generations ($T=0.7, \text{top\_p}=0.9$):
- **Pairwise Cosine Matrix**: Encodes $N$ generated candidate outputs into normalized vectors $\{\mathbf{v}_1, \dots, \mathbf{v}_N\}$:
  $$S_{i,j} = \mathbf{v}_i \cdot \mathbf{v}_j$$
- **Consensus Score**:
  $$\text{ConsistencyScore} = \frac{\bar{S} + 1.0}{2.0} \in [0.0, 1.0]$$

---

## 3. Intelligent Gating Architecture (`SelfConsistencyGate`)

### The Problem with Brute-Force Self-Consistency
Running multi-sample stochastic decoding ($N=5$) for every query increases inference latency by **$500\%$** and GPU compute costs dramatically.

### The HalluciGuard Solution: Dual-Condition Gating
The `SelfConsistencyGate` evaluates two strict conditions before allowing Self-Consistency to execute:

```python
# Condition A: Single-pass risk must be MEDIUM
Condition_A = (Initial_Risk_Level == RiskLevel.MEDIUM)

# Condition B: Prompt category must be Analytical
Condition_B = Query_Category in {
    QueryCategory.FACTUAL_QUESTION,
    QueryCategory.EXPLANATION,
    QueryCategory.SUMMARIZATION,
    QueryCategory.REASONING
}

Should_Execute_Self_Consistency = Condition_A and Condition_B
```

#### Why This Is Best-in-Class:
- **LOW Risk Queries** (High token confidence, low entropy): Skipped $\rightarrow$ Accepted in $< 50\text{ms}$.
- **HIGH Risk Queries** (Clear hallucination indicators): Skipped $\rightarrow$ Directly routed to Verifier Agent for external fact retrieval.
- **Creative / Open-Ended Prompts** (`roleplay`, `storytelling`, `poetry`, `brainstorming`): Skipped $\rightarrow$ High variation is expected in creative content, rendering self-consistency meaningless.
- **MEDIUM Risk Analytical Queries**: Executed $\rightarrow$ Compute is expended **only** on ambiguous factual queries where additional sampling resolves uncertainty.

---

## 4. Input & Output Contract Specification

### Input Contract (`DetectionInput`)
```json
{
  "user_query": "What is the recommended first-line treatment for Type 2 Diabetes Mellitus?",
  "llm_response": "Metformin is recommended as first-line therapy for type 2 diabetes."
}
```

### Output Contract (`DetectionResult`)
```json
{
  "confidence_score": 0.8842,
  "hallucination_probability": 0.1158,
  "risk_level": "LOW",
  "next_action": "Accept",
  "metrics": {
    "token_probability": {
      "avg_logprob": -0.14231,
      "min_logprob": -1.20451,
      "low_confidence_ratio": 0.0
    },
    "entropy": {
      "average_entropy": 0.4215,
      "maximum_entropy": 2.1402,
      "normalized_entropy": 0.0354
    },
    "semantic_similarity": {
      "raw_similarity": 0.8124,
      "normalized_similarity": 0.8124
    },
    "self_consistency": null
  }
}
```

---

## 5. Architectural Evaluation: Normal vs. Best-in-Class Verification

| Feature | Standard Baseline Verification | HalluciGuard Detector Agent | FAANG Technical Advantage |
|---------|--------------------------------|-----------------------------|---------------------------|
| **Detection Method** | Single regex heuristic or fixed perplexity check. | **Hybrid Multi-Signal Fusion** (Logprobs + Predictive Entropy + Semantic Embeddings + Self-Consistency). | Evaluates both internal model confidence geometry and semantic textual alignment. |
| **Compute Efficiency** | Runs 5-10 sample generations on *every* request. | **Intelligent Dual-Condition Gate** (`SelfConsistencyGate`). | Cuts GPU compute overhead by $\sim 80\%$ by reserving sampling for ambiguous `MEDIUM` risk analytical queries. |
| **Intent Awareness** | Treats all prompts identically (code, poem, medical fact). | **11-Category Regex Intent Classifier** (`QueryCategory`). | Prevents false hallucination alarms on non-factual creative writing or roleplay prompts. |
| **Score Aggregation** | Fixed hardcoded formula. | **Dynamic Pydantic Weighted Aggregator** (`SignalWeights`). | Enforces strict weight normalization ($\sum w = 1.0$) with runtime re-weighting when signals are enabled/disabled. |
| **Downstream Handoff** | Binary true/false output. | **Pipeline Action Directive** (`Accept` vs. `Verify`). | Seamless integration with Verifier Agent for automated claim extraction and multi-source API verification. |

---

## 6. FAANG Production Recommendations for Further Evolution

1. **Quantized Model Serving for Detector**:
   - Serve the baseline logit model (`Qwen2.5-0.5B-Instruct` or `Llama-3.2-1B`) in **vLLM** or **TensorRT-LLM** with FP8/INT4 quantization to extract logprobs in $< 15\text{ms}$.

2. **Learnable Signal Weight Optimization**:
   - Replace manual weights ($0.40$ logprob, $0.30$ entropy, $0.30$ semantic similarity) with a trained **Logistic Regression / Calibrated XGBoost Meta-Learner** fit on benchmark datasets (FEVER, HaluEval, TruthfulQA).

3. **Span-Level Hallucination Attribution**:
   - Extend the token probability metrics from sequence-level averages down to **token-level span highlighting**, identifying exact hallucinated phrases within long LLM paragraphs.
