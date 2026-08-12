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

- **LangGraph:** shared state, conditional routing, retries, bounded loops and execution trace.
- **Detector:** HaluEval-based hallucination-risk gate. LOW/MEDIUM follow the configured fast path; HIGH enters the Verifier.
- **Verifier:** the existing 9-stage retrieval, reranking and NLI pipeline. LangGraph invokes the real `VerificationPipeline`, not a mock.
- **Judge:** the existing governance/decision intelligence engine. It decides ACCEPT, CORRECT, REJECT, ABSTAIN, VERIFY_AGAIN or human escalation.
- **Corrector:** the existing evidence-grounded correction pipeline. It runs only when the Judge requests correction.
- **Memory:** the existing NetworkX knowledge graph + FAISS vector store + cache/pattern/source-trust persistence. It records the terminal verification facts.

## State contract

`orchestration.state.HalluciGuardState` is the inter-agent communication contract. The state contains request/draft data, Detector output, normalized claim/evidence data, Verifier output, Judge decision, Corrector output, Memory output, retry count and trace information. Agents do not call each other directly.

## Real integration details

The graph deliberately adapts the existing agent contracts instead of rewriting them:

1. The Detector is instantiated from `agents.detector_agent.detector.DetectorAgent` and its actual HaluEval model result controls routing.
2. The Verifier is loaded from the existing `agents/verifier_agent` package and executes `VerificationPipeline.verify(...)`. Its `ClaimReport`/`EvidenceItem` output is transformed into the Judge's existing `claim_evidence_pairs` contract.
3. The Judge is the existing `DecisionIntelligenceEngine`; LangGraph supplies Detector output, real Verifier evidence, retry state and memory context.
4. The Corrector receives a real `JudgeVerificationPayload` constructed from the actual Verifier evidence and Judge reasoning.
5. Memory receives the actual verified claims/evidence and writes through the existing MemoryAgent, including its knowledge graph and vector store.

No fake evidence, fake Judge decision, fake correction, or duplicate business logic is introduced by the orchestration layer.

## Install

Install orchestration dependencies and then the existing agent dependencies:

```bash
pip install -r orchestration/requirements.txt
pip install -r agents/detector_agent/requirements.txt
pip install -r agents/verifier_agent/requirements.txt
pip install -r agents/memory_agent/requirements.txt
```

Use the existing Judge/Corrector manifests or environment setup for those agents where applicable.

## Run the Verification Engine

```bash
uvicorn orchestration.api:app --host 127.0.0.1 --port 8010
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

The response contains the real Detector result, Verifier evidence, Judge decision, optional Corrector result, Memory result and graph trace.

## Real end-to-end validation

After installing all agent dependencies and making the trained Detector artifacts available, run:

```bash
python -m orchestration.scripts.verify_e2e
```

This intentionally executes real graph nodes and prints the observed path. A successful run should show a trace containing the actual nodes reached, for example:

```text
 detector -> verifier -> judge -> corrector -> memory
```

or, for a terminal Judge decision without correction:

```text
 detector -> verifier -> judge -> finish -> memory
```

A LOW/MEDIUM Detector fast-path request is expected to show:

```text
 detector -> accept -> memory
```

The E2E script is therefore different from the graph-contract pytest suite: pytest proves the graph topology and routing functions compile; `verify_e2e.py` proves the real installed five-agent runtime can execute.

## Bounded verification loop

`HALLUCIGUARD_MAX_VERIFICATION_RETRIES` controls the maximum number of re-verification cycles and defaults to `2`. Once the limit is reached, another `VERIFY_AGAIN` decision is converted to the terminal path instead of creating an infinite loop.

## Important architecture rule

Do **not** duplicate detector/verifier/judge/corrector/memory business logic in LangGraph. LangGraph is the supervisor. Existing agents remain independently testable and reusable.

RabbitMQ, Redis, Neo4j, Kubernetes and Grafana should only be introduced when the corresponding production requirement is actually implemented; they are not fake dependencies of this graph.

**LangGraph controls workflow. LangChain/LangChain-Core supplies model/tool primitives where used. The five agents remain specialists.**
