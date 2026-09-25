# ReVerifier

The ReVerifier is a graph stage implemented in `orchestration/graph.py`. It independently decomposes the corrected candidate and runs the Verifier pipeline again instead of trusting Corrector output.

It checks remaining contradicted/conflicted claims, execution status, query topicality and correction relevance. A generic topicality invariant prevents a correction from passing by replacing the answer with an unrelated true fact.

Results return to Judge:

- passed with zero remaining contradictions → eligible for `ACCEPT`;
- remaining correctable contradictions with budget → another `CORRECT` cycle;
- exhausted correction budget → `REJECT`;
- failed/unavailable reverification → `ABSTAIN` and human review.

No post-correction pass is claimed when Corrector did not run.
