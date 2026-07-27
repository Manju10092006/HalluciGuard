# HalluciGuard Detector Agent - API Reference Manual

**Version**: 2.0.0  
**Target Audience**: Developers, Integrators, Research Engineers  

---

## 1. Class: `DetectorAgent`

The primary interface for hallucination detection.

### Constructor Signature
```python
def __init__(
    config: Optional[DetectorConfig] = None,
    model_manager: Optional[ModelManager] = None
) -> None
```
- **Parameters**:
  - `config`: Custom `DetectorConfig` settings instance. If `None`, defaults to standard weights and thresholds.
  - `model_manager`: Pre-initialized `ModelManager` instance. If `None`, instantiates a shared singleton loader.

---

### Method: `detect`

Executes multi-signal hallucination detection on a user prompt and candidate response.

```python
def detect(
    user_query: str,
    llm_response: str
) -> DetectionResult
```

- **Parameters**:
  - `user_query` (`str`): The original prompt or question provided by the user. Must be a non-empty string.
  - `llm_response` (`str`): The candidate LLM output string to evaluate. Must be a non-empty string.

- **Returns**: `DetectionResult` Pydantic data model.

- **Exceptions**:
  - `ValueError`: Raised if `user_query` or `llm_response` are empty or blank strings.
  - `RuntimeError`: Raised if neural model inference fails or weights are corrupted.

#### Usage Example
```python
from detector_agent.detector import DetectorAgent

agent = DetectorAgent()
result = agent.detect(
    user_query="What is the capital of Australia?",
    llm_response="The capital of Australia is Canberra."
)

print(f"Confidence: {result.confidence_score:.4f}")
print(f"Risk Level: {result.risk_level.value}")
```

---

## 2. Pydantic Data Models (`detector_agent.models`)

### `DetectionResult`
```python
class DetectionResult(BaseModel):
    confidence_score: float           # [0.0, 1.0] Higher means more trustworthy
    hallucination_probability: float  # [0.0, 1.0] (1.0 - confidence_score)
    risk_level: RiskLevel             # RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH
    next_action: NextAction           # NextAction.ACCEPT, NextAction.VERIFY
    metrics: SignalMetricsDetail      # Detailed per-signal statistical breakdowns
    metadata: Dict[str, Any]          # Optional execution metadata & query categories
```

### Enums
- `RiskLevel`: `"LOW"`, `"MEDIUM"`, `"HIGH"`
- `NextAction`: `"ACCEPT"`, `"VERIFY"`

---

## 3. Class: `DetectorConfig`

Configurable runtime settings and signal weights.

```python
class DetectorConfig(BaseSettings):
    signal_weights: SignalWeights = Field(default_factory=SignalWeights)
    low_risk_threshold: float = 0.40
    high_risk_threshold: float = 0.55
```

### `SignalWeights`
```python
class SignalWeights(BaseModel):
    token_probability: float = 0.35
    entropy: float = 0.25
    semantic_similarity: float = 0.25
    self_consistency: float = 0.15
```
*Note*: `SignalWeights` validates that all individual weights are $\ge 0.0$ and sum to $1.0$.

---

## 4. Class: `ModelManager`

Thread-safe model loader managing PyTorch model instances.

```python
class ModelManager:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None
    
    @property
    def model(self) -> AutoModelForCausalLM
    
    @property
    def tokenizer(self) -> AutoTokenizer
    
    @property
    def sentence_model(self) -> SentenceTransformer
```
