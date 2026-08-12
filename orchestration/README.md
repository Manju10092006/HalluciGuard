# HalluciGuard LangGraph Verification Engine

This package is the **stateful supervisor/orchestration layer** for the five existing HalluciGuard agents. It does not replace their internal implementations.

## Agent topology

```text
User / LLM draft
      |
      v
+-------------+
|   Detector  |  HaluEval DistilBERT risk gate
+------+------+ 
       | LOW/MEDIUM                  HIGH
       |                              |
       v                              v
    ACCEPT                       +-----------+
       |                         | Verifier  |
       |                         +-----+-----+
       |                               |
       |                               v
       |                         +-----------+
       |                         |   Judge   |
       |                         +-----+-----+
       |                               |
       |              +----------------+----------------+
       |              |                |                |
       |          VERIFY_AGAIN      CORRECT        terminal
       |              |                |                |
       |              +----> Verifier  v                v
       |                         Corrector            Finish
       |                              |                |
       +------------------------------+----------------+
                                      |
                                      v
                                +-----------+
                                |  Memory   |
                                | KG+Vector |
                                +-----+-----+
                                      |
                                      v
                                  FINAL ANSWER
```

## Responsibilities

- **LangGraph:** shared state, conditional routing, retries, bounded loops, checkpoints/traceability when enabled.
- **Detector:** HaluEval-based hallucination-risk gate. LOW/MEDIUM bypass verification; HIGH enters the Verifier.
- **Verifier:** existing 9-stage retrieval, reranking and NLI pipeline. It supplies evidence, source metadata and claim-level verdicts.
- **Judge:** existing governance/decision intelligence engine. It decides ACCEPT, CORRECT, REJECT, ABSTAIN, VERIFY_AGAIN or human escalation.
- **Corrector:** existing evidence-grounded correction pipeline. It runs only when the Judge requests correction.
- **Memory:** existing NetworkX knowledge graph + FAISS vector store + cache/pattern/source-trust persistence. It records verified execution facts after the terminal path.

## State contract

`orchestration.state.HalluciGuardState` is the inter-agent contract. Each node writes only its own state section (`detector`, `verifier`, `judge`, `corrector`, `memory`) plus routing/trace fields. This avoids uncontrolled agent-to-agent text coupling.

## Install

Install the orchestration dependencies and then each agent's existing requirements:

```bash
pip install -r orchestration/requirements.txt
pip install -r agents/detector_agent/requirements.txt
pip install -r agents/verifier_agent/requirements.txt
pip install -r agents/memory_agent/requirements.txt
# install the existing Judge/Corrector dependencies from their project manifests
```

## Run

```bash
uvicorn orchestration.api:app --host 127.0.0.1 --port 8010 --reload
```

Then:

```bash
curl -X POST http://127.0.0.1:8010/verify \
  -H "Content-Type: application/json" \
  -d '{
    "user_query":"What is the capital of France?",
    "llm_response":"The capital of France is Tokyo.",
    "domain":"general"
  }'
```

The response contains the detector result, verifier evidence, Judge decision, optional correction result, Memory commit result, and the complete graph trace.

## Important design rule

Do **not** duplicate detector/verifier/judge/corrector/memory business logic in the graph. The graph is the supervisor. Existing agents remain independently testable and reusable.

## Current graph behavior

1. Detector runs first.
2. LOW/MEDIUM -> accept draft -> Memory -> END.
3. HIGH -> Verifier -> Judge.
4. Judge `VERIFY_AGAIN` -> Verifier, bounded by `HALLUCIGUARD_MAX_VERIFICATION_RETRIES` (default 2).
5. Judge `CORRECT` -> Corrector -> Memory -> END.
6. Judge terminal decisions -> Finish -> Memory -> END.

This preserves the architecture: **LangGraph controls the workflow; LangChain/LangChain-Core supplies model/tool primitives where used; agents remain specialists.**
