# HalluciGuard agent packages

HalluciGuard uses a Base LLM followed by seven trust stages. Not every stage is a directory under `agents/`: generation and claim analysis are shared services, the trained Detector is the standalone `halluciguard_detector` package, and the remaining core agents live here.

| Stage | Implementation | Detailed guide |
|---|---|---|
| Base LLM | `services/base_llm_service.py` | [`docs/agents/base-llm.md`](../docs/agents/base-llm.md) |
| Claim Analyzer | `services/claim_analyzer.py` | [`docs/agents/claim-analyzer.md`](../docs/agents/claim-analyzer.md) |
| Detector | `halluciguard_detector/` + `orchestration/detector_bridge.py` | [`halluciguard_detector/README.md`](../halluciguard_detector/README.md) |
| Verifier | `agents/verifier_agent/` | [`docs/agents/verifier.md`](../docs/agents/verifier.md) |
| Judge | `agents/judge_agent/` | [`docs/agents/judge.md`](../docs/agents/judge.md) |
| Corrector | `agents/corrector_agent/` | [`docs/agents/corrector.md`](../docs/agents/corrector.md) |
| ReVerifier | `orchestration/graph.py` using the Verifier pipeline | [`docs/agents/reverifier.md`](../docs/agents/reverifier.md) |
| Memory | `agents/memory_agent/` | [`docs/agents/memory.md`](../docs/agents/memory.md) |

Canonical handoffs are defined in `orchestration/schemas.py`. Agents must not silently replace failed model execution with a successful-looking result. The graph records completion, failure, degradation and skipped conditional stages explicitly.

```text
agents/
├── corrector_agent/
│   ├── corrector/       # targeting, evidence binding, generation and validation
│   ├── tests/           # unit, fuzz and integration coverage
│   └── training/        # dataset and training utilities
├── judge_agent/         # criticality, policy, consensus and decisions
├── memory_agent/
│   ├── api/             # optional standalone FastAPI surface
│   ├── cache/           # verification cache
│   ├── knowledge_graph/ # entity/relation persistence
│   ├── vector_store/    # semantic recall
│   ├── trust/           # source trust state
│   └── patterns/        # learned verification patterns
└── verifier_agent/
    ├── adapters/        # domain/source retrieval adapters
    ├── claims/          # decomposition and relation extraction
    ├── rerankers/       # BGE cross-encoder
    ├── nli/             # DeBERTa NLI
    ├── scorers/         # evidence and conflict scoring
    ├── api/             # pipeline and optional API
    └── tests/           # model, semantics and regression coverage
```

See [`docs/agents.md`](../docs/agents.md) for the complete input/output matrix.
