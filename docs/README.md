# HalluciGuard documentation

This folder is organized into three layers: **current guides** (the authoritative
description of the working system), **detailed agent guides**, and **reports**
(dated, historical, provenance material).

## Current guides

- [Architecture](architecture.md)
- [Project structure: frontend, backend, src and APIs](project-structure.md)
- [Agent contracts](agents.md)
- [Run all seven agents and inspect Memory](seven-agent-validation.md)
- [Detector](detector.md)
- [API reference](api.md)
- [Retrieval and n8n](retrieval.md)
- [Verification semantics](verification.md)
- [Hosted LLM providers](llm-providers.md)
- [Training and experiments](experiments.md)
- [Research and model provenance](research.md)
- [Deployment](DEPLOYMENT.md)
- [Operations runbook](runbook.md)
- [Decision log](decision-log.md)

## Detailed agent guides

Per-agent input/output contracts and behavior live in [`docs/agents/`](agents/):

- [Base LLM](agents/base-llm.md)
- [Claim Analyzer](agents/claim-analyzer.md)
- [Verifier](agents/verifier.md)
- [Judge](agents/judge.md)
- [Corrector](agents/corrector.md)
- [ReVerifier](agents/reverifier.md)
- [Memory](agents/memory.md)

## Reports (historical & provenance)

Versioned baseline audits, change logs, validation runs, and engineering
reports live under [`docs/reports/`](reports/). See the
[reports index](reports/README.md).

The [October 4 execution repair review](reports/execution-repair-review-20261004.md) lists the source changes, actual validation scope and known remaining issues. It distinguishes a working console/repair demonstration from measured hallucination-detection accuracy.

> Dates and scope matter: a historical document can describe earlier models,
> verdict names, or pipeline stages and must **not** override the current
> architecture above.
