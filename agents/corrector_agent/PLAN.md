# Corrector Agent — Web Port & Local Qwen 1.5B Fine-Tune Plan

## 1. Current State Analysis
The `corrector_agent` is currently implemented as an Android/Kotlin application. It features a pipeline consisting of an Orchestrator, Planner, PromptBuilder, Model Client, Merger, and Judge. 
- **Model Backend**: It currently relies on API-based external cloud models (Gemini/HuggingFace Router).
- **Judge Agent**: Uses naive, heuristic word-overlap ratios for verification and artificially injects a fake rejection on the first attempt for demonstration purposes.
- **Memory Agent**: It is currently missing a genuine memory implementation; it merely logs a string claiming to "sync" to the memory agent instead of actually persisting hallucination events.

## 2. Goals / Non-Goals
**Goals**:
- Upgrade the corrector agent's intelligence, verification authenticity, and memory persistence.
- Replace the Android/Kotlin architecture with a FastAPI-based pure Python backend.
- Replace external API calls with a locally hosted, fine-tuned 1.5B parameter model (Qwen2.5-1.5B-Instruct).
- Implement real model-based verification for the judge and genuine SQLite-backed logging for the memory agent.

**Non-Goals**:
- We will **NOT** modify any other agent's code, interface contract, or input/output format (e.g., `detector_agent`, `judge_agent`, etc.). All other agent folders remain strictly read-only.
- We will not alter the corrector agent's external JSON payload structure; it must seamlessly integrate with any existing pipeline interfaces.

## 3. Dataset Integration Plan
**Source**: Three dataset files in `C:\Users\PENDYAL GAURAV\Documents\SDC-II\Datasets` containing ~600 records.
**Data Processing**: 
1. Load, validate, and deduplicate records across the three JSONL files.
2. Split the ~600 records into `train` (80%), `validation` (10%), and `held-out test` (10%) splits.
**Schema Mapping**:
- `query` & `original_response`: Used as inputs for the Planner and PromptBuilder.
- `judge_decision`, `hallucination_type`, & `severity`: Used as signals for the planner to guide correction strategy.
- `verified_evidence`: Inserted into the prompt as grounding truth to constrain the rewrite.
- `judge_reason`: Injected into the PromptBuilder as additional grounding signal/context for *why* the original response failed.
- `support_score`, `contradiction_score`, `confidence_score`: Evaluated as trust/routing signals to dictate handling intensity.
**Target (Label) Generation**: 
To produce the ground-truth "corrected response" for training, we will port the existing `LocalOnDeviceClient.kt` logic. This script will programmatically synthesize the target by preserving accurate claims verbatim, rewriting contradicted claims using `verified_evidence`, and adding disclaimers for unsupported claims. An optional LLM-assisted paraphrasing pass may be applied to make the synthetic labels more fluid and natural.

## 4. Model & Fine-Tuning Plan
**Model Choice**: `Qwen/Qwen2.5-1.5B-Instruct`. It is highly capable for its small 1.5B footprint, natively instruction-tuned, and aligns perfectly with our need to follow complex "preserve/rewrite/disclaimer" rules locally.
**Fine-Tuning (LoRA/QLoRA)**:
- **Setup**: QLoRA (4-bit quantization) with `peft` and `trl` to run on consumer-grade hardware.
- **Parameters**: Rank (r=16), 3-5 epochs, batch size 4 (with gradient accumulation).
- **Time**: Fine-tuning on ~600 records will take under 1 hour on a standard consumer GPU (e.g., RTX 3060/T4).
**Local Serving**:
- The LoRA weights will be merged into the base model.
- We will optionally convert the model to GGUF format and serve it using Ollama or `llama.cpp` to provide a robust, Python/CUDA-free local HTTP endpoint (`http://localhost:11434`). This seamlessly replaces the previous external API calls without changing external pipeline interactions.

## 5. Corrector Agent Improvements
- **Model-Based Verification**: The `judge` component will discard the naive heuristic word-overlap and hardcoded fake rejections. It will be replaced by a localized NLI/entailment check (or a second pass from the fine-tuned model) to genuinely verify if the corrected text aligns with the evidence.
- **Genuine Memory Agent**: We will implement a real SQLite-backed memory table `(id, timestamp, claim_text, status, evidence_used, correction_applied, domain)` instead of standard output logs, enabling historical hallucination analytics.
- **Prompt Refinement**: The `PromptBuilder` will be tightened to explicitly leverage the dataset's `judge_reason` and `hallucination_type` fields, providing the local Qwen model with sharper signals for correction.

## 6. File-by-File Change List
*(All changes confined strictly to `agents/corrector_agent/`)*

- `app/` (existing Kotlin Android code) → **[DELETE]**: Entire tree removed in favor of the Python backend.
- `app/main.py` → **[ADD]**: FastAPI entrypoint and endpoint definitions (`/correct`).
- `app/models.py` → **[ADD]**: Pydantic models matching existing `Models.kt` schema.
- `app/planner.py` → **[ADD]**: Pure Python port of `CorrectionPlanner.kt`.
- `app/prompt_builder.py` → **[ADD]**: Pure Python port of `PromptBuilder.kt`, enhanced with `judge_reason`.
- `app/merger.py` → **[ADD]**: Pure Python port of `ResponseMerger.kt` and diff computation.
- `app/judge.py` → **[ADD]**: Real entailment-based verification replacing `JudgeVerificationEngine.kt`.
- `app/orchestrator.py` → **[ADD]**: Port of `CorrectorOrchestrator.kt` with retry loop logic.
- `app/model_client.py` → **[ADD]**: Qwen client to interface with local Ollama/Transformers (replaces Gemini/HF clients).
- `app/memory_agent.py` → **[ADD]**: SQLite persistence logic for hallucination logging.
- `training/build_dataset.py` → **[ADD]**: Script to ingest the ~600 records, map schema, and synthesize target labels.
- `training/train_lora.py` → **[ADD]**: QLoRA training script.
- `requirements.txt` → **[ADD]**: Python package dependencies.
- `frontend/` (HTML/JS/React) → **[ADD]**: Minimal web interface for testing the `/correct` endpoint.
- `build.gradle.kts`, `settings.gradle.kts`, `gradle.properties`, `gradle/` → **[DELETE]**: No longer needed.
- `benchmark_engine.py`, `benchmark_blind_engine.py`, `build_full_dataset.py` → **[MODIFY/DELETE]**: Clean up or adapt to the new Python application structure as needed.

## 7. Evaluation Plan
- **Test Set**: Use the 10% held-out test split from the 600 records. Ensure absolutely no ground-truth leakage occurs in the planner, prompt builder, or judge steps.
- **Metrics**: 
  - *Correction Rate*: % of hallucinations successfully rewritten to match `verified_evidence`.
  - *Preservation Rate*: % of accurate claims left untouched.
  - *Judge Accuracy*: False positive / False negative rates of the newly implemented model-based `judge.py`.
- **Before/After Analysis**: Compare the fine-tuned Qwen 1.5B's success metrics against a baseline run (using the raw, untuned model or the previous logic logs).

## 8. Risks / Open Questions
1. **Hardware Assumptions**: Do we have enough local VRAM available to run the merged Qwen 1.5B in fp16 natively, or is the GGUF/Ollama quantization path strictly required for the final deployment?
2. **Schema Enhancements**: Are we allowed to append extra fields to the final JSON returned to the client (e.g., adding `corrector_confidence`), or must it identically match the old Android spec without any new keys?
3. **Training Label Generation**: If the `LocalOnDeviceClient.kt` programmatic label generation proves too robotic, should we definitely invoke a cloud API (like Gemini/Claude) once *just* to generate higher-quality training labels, prior to local training?
