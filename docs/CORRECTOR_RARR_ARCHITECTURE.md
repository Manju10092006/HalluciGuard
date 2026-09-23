# Corrector Agent — Evidence-Bound Revision Architecture

## Decision

HalluciGuard now uses a **RARR-inspired, claim-localized, evidence-bound revision loop** for the production Corrector.

The design combines ideas from:
- RARR: post-hoc research/revision rather than regenerating an answer.
- RefChecker/FActScore: fine-grained claim/atomic-fact localization.
- Existing HalluciGuard work: authorized sentence targeting, evidence binding, strict JSON parsing, validation gates, bounded retry, and deterministic reconstruction.

We are **not cloning an external repository**. HalluciGuard keeps ownership of the verification, evidence, authorization, validation, and reconstruction layers.

## Runtime path

~~~text
Base LLM draft
    |
    v
Detector
    |
    v
Verifier
    |
    v
Judge
    | CORRECT
    v
Corrector targeting
    | authorized claim -> exact original sentence span
    v
Evidence binding
    | supporting evidence only
    v
Groq Corrector (API)
    | one target, one JSON correction candidate
    v
Strict output parser
    |
    v
HalluciGuard validation gates
    |
    +---- invalid ----> bounded retry with same evidence/target
    |
    v
Deterministic reconstruction
    |
    v
Re-verifier
    |
    +---- fail ----> bounded correction/reverification path
    |
    v
Final response
~~~

## Why this solves the stateless-LLM problem

The Corrector never relies on the model remembering an earlier API call.

Every generation call explicitly receives:
- the immutable original target sentence;
- the authorized claim IDs;
- the original claim text;
- supporting evidence bound to those claims;
- strict edit rules;
- a deterministic target sentence ID.

The model is therefore a **stateless editor**. HalluciGuard owns execution state.

## Why Groq is only the generation layer

The hosted model is not trusted as a verifier.

Groq produces a candidate. HalluciGuard then performs:
1. structural validation;
2. target authorization;
3. original-span identity;
4. preservation checks;
5. number/date consistency;
6. entity consistency;
7. unsupported-addition detection;
8. evidence grounding;
9. contradiction detection;
10. minimal-edit validation;
11. optional entailment validation;
12. deterministic offset-based reconstruction.

A provider/API failure never bypasses these checks.

## Current hosted model

Default:

~~~text
provider = groq
model = openai/gpt-oss-120b
reasoning_effort = low
temperature = 0
JSON mode = enabled
~~~

The model is configurable through environment variables. The API client honors Groq 429 responses and retry-after before retrying.

This does **not** claim unlimited free API access. Groq account/model limits still apply.

## Environment

~~~env
HG_CORRECTOR_PROVIDER=groq
GROQ_API_KEY=<server-side-secret>
HG_CORRECTOR_GROQ_MODEL=openai/gpt-oss-120b
HG_CORRECTOR_GROQ_REASONING_EFFORT=low
HG_CORRECTOR_GROQ_MAX_RETRIES=2
HG_CORRECTOR_GROQ_TIMEOUT_SECONDS=45
~~~

Set HG_CORRECTOR_PROVIDER=local to retain the existing local fine-tuned model path.

## Important invariant

The Corrector may propose text, but it never decides whether that text is true.

Only evidence + HalluciGuard validation + re-verification can authorize the corrected answer.