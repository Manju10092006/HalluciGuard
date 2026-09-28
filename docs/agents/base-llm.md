# Base LLM

The Base LLM converts a user query into the candidate answer that HalluciGuard evaluates. `services/base_llm_service.py` implements one auditable multi-provider request path; `services/llm_providers.py` declares provider configuration.

## Provider order

The default order is Groq → Gemini → OpenRouter. A provider without a key is skipped. Each provider receives its own bounded retry budget; exhaustion advances to the next configured provider. The result records the serving provider, model, latency, finish reason and a secret-free attempt trail.

Configured defaults are Groq `openai/gpt-oss-120b`, Gemini `gemini-3.6-flash`, and OpenRouter `qwen/qwen-2.5-7b-instruct`. Environment variables may override them.

The Base LLM is a generator only. Its answer is never trusted merely because it is fluent, confident, or produced by a particular provider.
