# Seven-agent backend demo

The executable remains at repository root as `demo_7_agents.py` so imports and user commands stay stable. This directory contains its documentation only.

## Run your own query

```powershell
python demo_7_agents.py "Who founded Microsoft?"
```

Interactive and built-in demonstrations:

```powershell
python demo_7_agents.py --interactive
python demo_7_agents.py --demo 1
python demo_7_agents.py --demo 2
python demo_7_agents.py --demo 3
python demo_7_agents.py --demo all
```

Append the raw structured graph state:

```powershell
python demo_7_agents.py "What is the capital of India?" --raw
```

## Output sections

1. **Base LLM** — provider, model, latency and generated candidate.
2. **Claim Analyzer** — factual claims, discarded text and retrieval queries.
3. **Detector** — the latest grounded model output when evidence exists, including load/inference/calibration proof.
4. **Retrieval + Verifier** — documents, evidence passages, NLI signals and per-claim verdicts.
5. **Judge** — decision, reason, basis, confidence and correction authorization.
6. **Corrector** — corrected text and validation, or truthful `NOT_REQUIRED`.
7. **ReVerifier** — independent post-correction result, or truthful `NOT_REQUIRED`.
8. **Memory** — persistence status, stored/duplicate/failed counts and backend availability.
9. **Final result** — response, terminal status and complete agent trace.

The script accepts only a user query. It does not accept a prewritten draft, fabricate evidence, or claim that a conditional agent ran when it did not.
