# Agent contracts and responsibilities

HalluciGuard separates generation, factual analysis, verification, policy, repair, validation and persistence. The separation prevents any single model from generating a statement and certifying it as true.

| Stage | Input | Output | Can decide | Cannot decide |
|---|---|---|---|---|
| Base LLM | Query, history, generation mode | Candidate answer, provider/model, latency, attempt trail | Whether generation succeeded | Whether its own facts are true |
| Claim Analyzer | Candidate answer, query, domain | Factual claims, search queries, discarded spans | Whether text is independently checkable | Support, contradiction, release |
| Detector | Query + answer; later, Verifier evidence | Risk, confidence, per-sentence labels, execution diagnostics | Evidence-conditioned triage | Final factual verdict or release |
| Verifier | Atomic claims | Claim reports and decision-grade evidence | `VERIFIED`, `CONTRADICTED`, `UNVERIFIED`, `CONFLICTED` | Product release policy |
| Judge | Detector + Verifier + retry/reverification state | Decision, reason, confidence, optional correction request | Accept, correct, retry, reject, abstain | Retrieval, fact creation or rewriting |
| Corrector | Authorized claims, original answer, bound evidence | Minimal corrected candidate or failure | Targeted repair within authorization | Editing preserved spans or inventing facts |
| ReVerifier | Corrected answer | Fresh Verifier result, pass/fail, remaining contradictions | Whether correction passes independent checks | Automatically accepting failed/ungrounded repair |
| Memory | Accepted verified claim reports | Store/duplicate/failure/skip result | Persistence and reuse | Declaring stored information permanently current |

## Normal path

```text
query → generation → triage → factual claims → evidence verification
      → grounded Detector → Judge ACCEPT → Memory → final answer
```

## Correction path

```text
Verifier contradiction → Judge CORRECT → authorized minimal repair
→ fresh verification → Judge ACCEPT / CORRECT / REJECT / ABSTAIN
```

## Failure policy

- Generation failure stops factual processing.
- Evidence-free Detector triage reports probability unavailable and routes to verification.
- Grounded Detector failure is marked degraded and fails closed; Verifier evidence is preserved.
- Verifier failure routes to human review rather than a fabricated verdict.
- Judge retry and correction loops are bounded.
- Corrector model echo, no-op output, unsupported additions, ambiguous targeting and provider failures are explicit failures.
- ReVerifier failure cannot be interpreted as a passed correction.
- Memory failure does not change the factual decision, but is reported as partial persistence failure.

## Canonical models

`orchestration/schemas.py` defines `DetectorResult`, `Evidence`, `ClaimReport`, `VerifierResult`, `CorrectionRequest`, `JudgeResult`, `CorrectionResult`, `ReverificationResult`, and `MemoryResult`. The runtime detector bridge also exposes operational fields such as `model_loaded`, `inference_executed`, `calibration_applied`, `model_version`, `calibrator_version`, `probability_available`, `per_claim_results`, warnings and diagnostics.

## Detailed guides

- [Base LLM](agents/base-llm.md)
- [Claim Analyzer](agents/claim-analyzer.md)
- [Detector](../halluciguard_detector/README.md)
- [Verifier](agents/verifier.md)
- [Judge](agents/judge.md)
- [Corrector](agents/corrector.md)
- [ReVerifier](agents/reverifier.md)
- [Memory](agents/memory.md)
