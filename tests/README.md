# Testing guide

HalluciGuard tests are colocated with packages plus a small root regression suite.

| Location | Coverage |
|---|---|
| `halluciguard_detector/tests/` | text segmentation, schemas and Detector adapter behavior |
| `agents/verifier_agent/tests/` | retrieval, reranking, NLI, evidence semantics, domains and regressions |
| `agents/judge_agent/tests/` | Judge defects and Verifier contract normalization |
| `agents/corrector_agent/tests/` | targeting, evidence binding, retry, validation, fuzz and integration |
| `agents/memory_agent/tests/` | cache, vector/graph persistence, trust, patterns and APIs |
| `orchestration/tests/` | graph routing, contracts, failures, Detector bridge and integrations |
| `services/tests/` | provider router, character regeneration and Corrector failure mapping |
| `tests/` | cross-package correction topicality and n8n repair regressions |

## Focused commands

```powershell
python -m pytest -q halluciguard_detector/tests
python -m pytest -q orchestration/tests
python -m pytest -q agents/verifier_agent/tests
python -m pytest -q agents/judge_agent/tests
python -m pytest -q agents/corrector_agent/tests
python -m pytest -q agents/memory_agent/tests
python -m pytest -q services/tests tests
```

Use focused suites during development and expand in CI. Model-runtime and live E2E tests can require large downloads, network access, provider credentials, n8n, and more time than hermetic unit tests. Mocked tests must not be described as proof that a real model or external provider executed.

The trained Detector integration finalization ran 44 focused orchestration and contract tests successfully. See `docs/experiments.md` for the distinction between automated and live results.
