# HalluciGuard Documentation Status

This document records the distinction between **implemented**, **validated**, **integrated**, and **planned** work.

## ✅ Implemented / strongly validated

- Detector Agent implementation and HaluEval-based classifier integration.
- Verifier Agent nine-stage retrieval/ranking/NLI/evidence pipeline.
- Local DeBERTa NLI validation for contradiction, entailment and neutral cases.
- NLI degraded-state hardening so failed inference is not treated as real evidence.
- Memory Agent implementation with knowledge graph, vector memory, cache, pattern learning and source trust.
- LangGraph orchestration architecture with shared state, Supervisor routing, bounded retries and structured trace/bus concepts.
- Base LLM service abstraction and OpenRouter integration code under active development.

## 🟡 Implemented but still under independent validation

- Judge Agent.
- Corrector Agent.
- Base LLM live provider execution.
- Full LangGraph real-model execution.
- Frontend-to-backend adapter.

## 🔜 Planned product milestones

1. Validate the OpenRouter model path with a real server-side key.
2. Finish the real active LangGraph backend E2E.
3. Connect the existing Next.js frontend through its VerificationService/HalluciGuardAdapter architecture.
4. Containerize the Python backend.
5. Deploy frontend on Vercel and backend on Render.
6. Validate real browser E2E.
7. Independently validate Judge.
8. Independently validate Corrector.
9. Reintroduce Judge and Corrector into the active five-agent graph only after their contracts and runtime behavior pass real tests.

## Documentation rule

The repository should never describe an agent as “production ready” merely because its files exist or a mocked test passes. Runtime/model availability and real end-to-end behavior are part of the definition of a working component.
