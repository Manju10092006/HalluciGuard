# Shared backend services

`services/` contains reusable stages that are not standalone agent packages.

| Module | Role |
|---|---|
| `base_llm_service.py` | Hosted chat completion, bounded retries, error taxonomy and provider-attempt trace |
| `llm_providers.py` | Groq → Gemini → OpenRouter declarations and environment-driven provider order |
| `claim_analyzer.py` | LLM-assisted factual-claim extraction with deterministic fallback |
| `n8n_retrieval_client.py` | Authenticated n8n webhook client and response normalization |
| `evidence_ranker.py` | Evidence relevance ordering before verification decisions |
| `character_regenerator.py` | Character/text regeneration helper |
| `llm_detector_service.py` | Compatibility service for the earlier LLM/Detector slice |
| `llm_detector_verifier_service.py` | Compatibility vertical-slice service |

## Provider routing

The default hosted generation order is Groq, Gemini, then OpenRouter. Providers without credentials are skipped; retry exhaustion advances to the next provider. The successful provider and model are recorded without logging credentials.

Defaults declared in `llm_providers.py` are Groq `openai/gpt-oss-120b`, Gemini `gemini-3.6-flash`, and OpenRouter `qwen/qwen-2.5-7b-instruct`. Deployments can override all three through environment variables.

## Claim Analyzer

The analyzer receives the Base LLM response, user query and domain. It returns preserved claim IDs, factual claim text, retrieval queries, discarded spans, analyzer source and fallback reason. It classifies checkability only; it never assigns truth. Malformed or unavailable hosted analysis activates a deterministic fallback and records `source=fallback`.

## n8n boundary

The n8n client sends individual or batched claim payloads and normalizes workflow responses into typed passages. It does not prune evidence or declare verdicts. Timeouts, non-200 responses and malformed JSON return controlled failures for the Python Verifier to handle.
