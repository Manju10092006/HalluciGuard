# Claim Analyzer

`services/claim_analyzer.py` receives the complete candidate answer, original query and domain. It separates independently checkable factual propositions from opinions, transitions, instructions, questions, refusals and other non-factual spans.

Each factual output preserves a claim ID, claim text, original sentence, claim type and retrieval queries. The Analyzer does not label claims true or false.

The preferred path uses a hosted LLM with a strict JSON contract. Invalid JSON, unavailable credentials or provider errors activate a deterministic fallback. The result identifies whether `llm` or `fallback` produced it and records the fallback reason. Unicode refusal patterns are filtered so a refusal is not accidentally sent to retrieval as a factual claim.

Example:

```text
Answer: "Microsoft was founded by Bill Gates and Paul Allen in 1975."
Claims:
  c1 Microsoft was founded by Bill Gates and Paul Allen.
  c2 Microsoft was founded in 1975.
```
