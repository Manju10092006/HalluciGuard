# Verifier Domain Intelligence

HalluciGuard Verifier now uses `config/domain_intelligence.yaml` as the single production registry for domain-specific model routing, authoritative APIs, knowledge bases, evidence ranking, confidence calibration, notebooks, and benchmarks.

## Runtime Flow

1. `DomainValidator` canonicalizes detector domains against the 30 supported domains.
2. `ModelRouter` loads the matching domain profile and returns a `ModelRoutingDecision`.
3. `VerificationPipeline` uses that decision for adapter selection, dense retrieval model, reranker model, and NLI model.
4. `ModelManager` lazily loads models by ID and uses CPU/GPU automatically. Model downloads are disabled by default so CI and missing-model environments fall back quickly; set `ALLOW_MODEL_DOWNLOADS=true` to warm or benchmark real model weights.
5. `AdapterRegistry` registers each required domain independently. Domains without a dedicated adapter use `DomainProxyAdapter` over the strongest existing authoritative adapter while preserving their own APIs, source credibility, and model profile.

## Supported Domains

The registry covers healthcare, medicine, pharmacy, biology, genetics, chemistry, physics, mathematics, astronomy, space science, climate and environment, agriculture, food and nutrition, artificial intelligence, machine learning, computer science, cybersecurity, programming, data science, finance, economics, business, law, government and public policy, history, geography, education, psychology, sociology, and philosophy.

## Research Notebooks

Run this command after editing the registry:

```powershell
python research\generate_research_notebooks.py
```

The generator creates executable notebooks under `research/`. Each notebook imports `models.research_companion`, which reads production registry and router modules. Notebooks must not duplicate inference logic.

## Validation

Run the verifier suite from `agents/verifier_agent`:

```powershell
python -m pytest -q
```

The suite validates domain coverage, alias canonicalization, adapter registration, notebook existence, notebook production imports, representative notebook execution, API contract smoke tests, and existing adapter/health/claim behavior.
