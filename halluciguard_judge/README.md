# halluciguard_judge — New Standalone Hallucination Detector

> **Status**: Standalone. Does NOT touch `agents/detector_agent/`. Will be integrated only after passing benchmarks.

---

## Architecture

```
LLM Response
     |
     v
Claim Extractor        (claim_extractor.py)
     |
[Claim, Claim, ...]    Each = atomic factual statement
     |
     v
DeBERTa-v3-base        (classifier.py)
Fine-tuned Classifier
     |
     v
   ┌─────────────────┐
   |                 |
LOW / HIGH       MEDIUM (uncertain)
   |                 |
   |                 v
   |           LLM Judge          (llm_judge.py)
   |           (optional,         Two-stage Datadog approach:
   |            OpenRouter)       CoT reasoning -> JSON verdict
   |                 |
   └────────┬────────┘
            |
            v
      DetectorOutput
            |
            v
        VERIFIER
```

**Key design rules (from spec):**
- Claim extraction runs **FIRST** (before classifier)
- Classifier is primary — **NOT NLI**
- LLM judge is **optional, uncertain cases only**
- No web search / Tavily / Wikipedia here — that is Verifier's job
- `hallucination_probability` = triage signal, NOT factual verdict

---

## Quick Start

```python
from halluciguard_judge import JudgeDetector

detector = JudgeDetector()

result = detector.detect(
    user_query="Who created Java?",
    llm_response="Java was created by Dennis Ritchie in 1972. "
                 "It was developed at Sun Microsystems."
)

print(result.overall_risk)    # HIGH
print(result.routing)         # VERIFY

for claim in result.claims:
    print(f"{claim.claim_id}: P(H)={claim.hallucination_probability:.3f} [{claim.risk_level}]")
    print(f"  -> {claim.text}")
```

**Example output (with trained model):**
```
C001: P(H)=0.94 [HIGH]  -> Java was created by Dennis Ritchie in 1972.
C002: P(H)=0.12 [LOW]   -> It was developed at Sun Microsystems.
Overall risk: HIGH
Routing: VERIFY
```

---

## Claim Output Contract (from spec)

```json
{
  "claims": [
    {
      "claim_id": "C001",
      "text": "Java was created by Dennis Ritchie.",
      "hallucination_probability": 0.94,
      "risk_level": "HIGH"
    },
    {
      "claim_id": "C002",
      "text": "Java was developed at Sun Microsystems.",
      "hallucination_probability": 0.08,
      "risk_level": "LOW"
    }
  ],
  "overall_risk": "HIGH",
  "routing": "VERIFY"
}
```

---

## Training

### Step 1: Download training data

**HaluBench:**
```bash
# https://huggingface.co/datasets/PatronusAI/HaluBench
python -c "
from datasets import load_dataset
ds = load_dataset('PatronusAI/HaluBench')
import json
# Convert to halluciguard format
samples = []
for item in ds['test']:
    samples.append({
        'query': item.get('question', ''),
        'claim': item.get('answer', ''),
        'label': 1 if item.get('answer_type') == 'hallucinated' else 0,
        'source': 'halubench'
    })
with open('halluciguard_judge/datasets/training/halubench.json', 'w') as f:
    json.dump(samples, f, indent=2)
"
```

**RAGTruth:**
```bash
# https://github.com/ParticleMedia/RAGTruth
# Download source.jsonl and convert
```

**Custom HalluciGuard data** (most important eventually):
- Format: `[{"query": "...", "claim": "...", "label": 0|1}]`
- Put in: `halluciguard_judge/datasets/training/custom_halluciguard.json`

### Step 2: Run training

```bash
python -m halluciguard_judge.trainer \
    --data_dir halluciguard_judge/datasets/training \
    --output_dir halluciguard_judge/checkpoints/deberta-v3-hallucination \
    --epochs 5 \
    --batch_size 16 \
    --max_length 512
```

Training will:
1. Load all `halueval*.json`, `ragtruth*.json`, `custom*.json` from `data_dir`
2. Apply class-weighted loss (handles imbalance)
3. Early stopping on validation F1
4. Temperature calibration on held-out set
5. Save checkpoint to `output_dir`

### Step 3: Configure checkpoint path

```bash
# .env or environment variable
JUDGE_CLASSIFIER_MODEL=halluciguard_judge/checkpoints/deberta-v3-hallucination
```

---

## Evaluation (Compare vs Old Detector)

```bash
python -m halluciguard_judge.evaluator \
    --dataset halluciguard_judge/datasets/sample_halueval.json \
    --dataset_type halueval \
    --output eval_results.json \
    --compare_old
```

**Success criterion** (from spec):
> F1 >= 0.78 on production-style test set without constant ~0.9 output

---

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `JUDGE_CLASSIFIER_MODEL` | `halluciguard_judge/checkpoints/...` | Path to fine-tuned checkpoint |
| `JUDGE_LOW_THRESHOLD` | `0.30` | P(H) <= this → LOW risk |
| `JUDGE_HIGH_THRESHOLD` | `0.60` | P(H) >= this → HIGH risk |
| `JUDGE_LLM_ENABLED` | `true` | Enable LLM judge for uncertain cases |
| `JUDGE_LLM_MODEL` | `qwen/qwen3-14b` | OpenRouter model for judge |
| `JUDGE_MAX_CLAIMS` | `15` | Max claims per response |
| `ALWAYS_VERIFY` | `false` | Always route to Verifier |

---

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Public API |
| `models.py` | Pydantic schemas (pipeline state contract) |
| `config.py` | Environment-based configuration |
| `claim_extractor.py` | Atomic claim extraction (FIRST in pipeline) |
| `classifier.py` | DeBERTa-v3-base classifier (PRIMARY detector) |
| `llm_judge.py` | Optional two-stage LLM judge (uncertain cases) |
| `prompts.py` | LLM judge prompt templates |
| `detector.py` | Main orchestrator (JudgeDetector) |
| `trainer.py` | Fine-tuning script |
| `evaluator.py` | Benchmark evaluator |
| `datasets/` | Sample training + test data |
| `tests/` | Unit tests |

---

## Running Tests

```bash
# Claim extractor tests (no model needed — fast)
pytest halluciguard_judge/tests/test_claim_extractor.py -v

# Detector pipeline tests (loads DeBERTa, ~60s first run)
pytest halluciguard_judge/tests/test_detector.py -v

# All
pytest halluciguard_judge/tests/ -v
```

---

## Degraded Mode

If the fine-tuned checkpoint is not found:
- Returns `P(H) = 0.5` for all claims
- Routes ALL responses to `VERIFY` (**fail-closed**)
- `result.degraded = True`
- `result.status = "degraded"`

This means the system is safe even without a trained model.

---

## Integration (After Benchmark Passes)

Only connect to HalluciGuard **after**:
1. `F1 >= 0.78` on test set
2. No constant-output behavior (old problem: always ~0.9)
3. Per-claim probabilities are meaningful (not all same value)

Then update `agents/detector_agent/detector.py` to use `JudgeDetector`.
