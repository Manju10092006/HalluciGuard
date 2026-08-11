# HalluciGuard Corrector Agent

The **Corrector Agent** is a critical component of the HalluciGuard ecosystem. Its role is to take a hallucinated, unsupported, or contradictory LLM output (as flagged by the Detector and Verifier agents) and rewrite it so that it strictly adheres to the provided grounding evidence without losing the user's original intent.

## Architecture & Migration

Originally prototyped as an Android application, this agent has been fully migrated to a robust Python backend architecture. 

It uses a multi-stage pipeline:
1. **Detector & Verifier Integration:** Receives claims and evidence analysis.
2. **Judge Agent:** Evaluates the severity and type of hallucination.
3. **Planner Agent:** Creates a step-by-step correction plan based on the evidence.
4. **Model Client:** Uses a fine-tuned LoRA adapter on `Qwen/Qwen2.5-1.5B-Instruct` to generate the final corrected response.
5. **Orchestrator:** A deterministic stitching pipeline that merges the original text with the generated corrections, guaranteeing no fallback to hallucinated text.

## Model Fine-Tuning

The Qwen2.5-1.5B model was fine-tuned using LoRA (Low-Rank Adaptation) on a curated dataset of 738 hallucination correction pairs. 
- **Framework:** Hugging Face `transformers`, `peft`, `trl`
- **Optimization:** BitsAndBytes 4-bit quantization, Gradient Checkpointing
- **Hardware Profile:** Optimized to train and run inference on consumer GPUs (e.g., RTX 3060/4060 with 8GB-12GB VRAM).

## Empirical Benchmark Performance

The agent is evaluated using a comprehensive benchmark harness (`benchmark_engine.py`).
- **Detector Accuracy:** 100%
- **Hallucination Elimination Rate:** ~83.33%
- **Claim Preservation:** 100%

*Full benchmark reports, confusion matrices, and ROC curves can be found in the generated markdown and CSV artifacts within this directory.*

## Setup and Usage

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support

### Installation
```bash
pip install -r requirements.txt
```

### Running the Orchestrator
You can run the main orchestrator script directly:
```bash
python app/main.py
```

### Running Benchmarks
To run the evaluation harness on the validation dataset:
```bash
python benchmark_engine.py
```
