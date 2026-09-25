# Verification semantics

HalluciGuard evaluates atomic claims, not an answer as one undifferentiated string. Claim IDs remain stable through retrieval, evidence scoring, Judge authorization, correction and memory.

## Evidence lifecycle

1. Normalize the claim and resolve relevant entities/relations.
2. Generate retrieval queries, including relation-aware variants.
3. Retrieve from configured domain adapters, n8n, Wikipedia and/or web fallback.
4. Deduplicate documents and passages.
5. Rerank claim–passage pairs with BGE.
6. Drop or zero-weight irrelevant passages.
7. Apply deterministic relation and named-entity safeguards.
8. Classify entailment, contradiction or neutrality with DeBERTa NLI.
9. Aggregate decision-grade evidence into support, contradiction and confidence scores.
10. Emit one public claim verdict.

## Four-state verdict

| Verdict | Interpretation |
|---|---|
| `VERIFIED` | Evidence materially supports the exact claim |
| `CONTRADICTED` | Evidence materially refutes the exact claim |
| `CONFLICTED` | Strong support and contradiction coexist |
| `UNVERIFIED` | Evidence is missing, irrelevant, neutral, weak, or incomplete |

`UNVERIFIED` does not mean false. `CONTRADICTED` requires evidence rather than lack of support. `CONFLICTED` prevents competing passages from being collapsed into a confident binary answer.

## Detector versus Verifier

The grounded Detector estimates calibrated hallucination risk over `(evidence, sentence)` inputs. The Verifier produces the evidence verdict. Judge may use Detector risk as a workflow signal, but Detector output is never counted as factual evidence.

## Confidence limitations

Scores depend on retrieval coverage, passage quality, model calibration and aggregation rules. A high numerical confidence is not a guarantee, particularly for time-sensitive, numerical, legal, medical, financial or adversarial claims.
