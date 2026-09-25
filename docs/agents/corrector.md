# Corrector Agent

The Corrector in `agents/corrector_agent/corrector/` performs evidence-bound, Judge-authorized repair. It is conditional and does not run for accepted answers.

## Safety sequence

1. Normalize the canonical `CorrectionRequest`.
2. Locate only authorized claims in the original response.
3. Lock preserved claims and all non-target spans.
4. Bind each target to its own supporting and contradictory evidence.
5. Build a bounded prompt for the configured local or hosted model.
6. Validate structure, authorization, identity, preservation, numbers, dates, entities, additions, grounding, contradiction, minimal edit and optional entailment.
7. Retry within a fixed budget or return an explicit failure.
8. Splice accepted sentence repairs back without regenerating the entire answer.

Provider routing uses the same hosted Groq/Gemini/OpenRouter service when configured. A local Qwen LoRA path also exists, but unavailable weights never silently fall back unless the operator explicitly enables that behavior.

Model echo, no-op matching, ambiguous targets, unsupported additions and provider failures are categorized. The Corrector cannot authorize its own edits and cannot make the final release decision.
