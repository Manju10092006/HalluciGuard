# Hosted LLM providers

`services/llm_providers.py` and `services/base_llm_service.py` implement the shared hosted generation layer used by Base LLM, Claim Analyzer and hosted Corrector paths.

| Order | Provider | Default model | Credential |
|---:|---|---|---|
| 1 | Groq | `openai/gpt-oss-120b` | `GROQ_API_KEY` |
| 2 | Gemini OpenAI-compatible endpoint | `gemini-3.6-flash` | `GEMINI_API_KEY` |
| 3 | OpenRouter | `qwen/qwen-2.5-7b-instruct` | `OPENROUTER_API_KEY` |

The order can be replaced with `HALLUCIGUARD_LLM_PROVIDER_ORDER`. Models, base URLs, timeouts and token budgets are environment-configurable. Providers without keys are skipped. Retryable transport/server failures use bounded backoff; exhaustion advances to the next provider. Malformed or empty assistant content is failure, not success.

OpenRouter has specific handling for unavailable models and credit-limited responses. Provider attempts are logged without credentials. Secrets are never serialized into response metadata.

Corrector can pin provider-specific correction models using `HG_CORRECTOR_GROQ_MODEL`, `HG_CORRECTOR_GEMINI_MODEL`, and `HG_CORRECTOR_OPENROUTER_MODEL`.
