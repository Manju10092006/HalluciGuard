"""
halluciguard_judge / prompts.py
────────────────────────────────
Prompt templates for the optional LLM-as-a-judge pass.

Triggered ONLY for MEDIUM-risk claims (uncertain band).
Implements the Datadog two-stage approach:
    Stage 1: Chain-of-thought rubric (free-form, no format restriction)
    Stage 2: Structured JSON extraction from the CoT output

References:
    Datadog (2025): "Detecting hallucinations with LLM-as-a-judge"
    Patronus AI Lynx: rubric-based judge prompting
"""

# ── Stage 1: Chain-of-thought rubric ─────────────────────────────────────────
# Free-form reasoning — no format constraints so the LLM can think freely.
# We frame the context as "expert advice" and the claim as "candidate statement"
# to create asymmetry (context = ground truth).

COT_SYSTEM_PROMPT = """\
You are an expert fact-checker evaluating whether a statement is factually correct.
Your job is ONLY to assess factual accuracy — not grammar, style, or opinion.

Rules:
1. Treat the USER QUESTION as the grounding context for what is being asked.
2. Evaluate the CANDIDATE STATEMENT against your knowledge and the question context.
3. Think step-by-step. Consider multiple interpretations before reaching a verdict.
4. Be especially careful about: names, dates, numbers, organisations, causality.
5. Do NOT guess. If you are genuinely uncertain, say so explicitly.
"""

COT_USER_TEMPLATE = """\
USER QUESTION:
{user_query}

CANDIDATE STATEMENT (from an AI assistant's response):
{claim_text}

Think step-by-step:
1. What factual claims are made in the Candidate Statement?
2. Are there any names, dates, or numbers that could be wrong?
3. Does the Candidate Statement contradict well-established facts?
4. Is the Candidate Statement unsupported (it goes beyond what we can verify)?
5. Reach a conclusion: is this statement likely hallucinated or likely correct?

Write your full reasoning below:
"""


# ── Stage 2: Structured extraction ───────────────────────────────────────────
# Converts the free-form CoT reasoning into a clean JSON verdict.
# Uses a smaller, faster LLM call.

EXTRACT_SYSTEM_PROMPT = """\
You are a JSON extraction assistant. 
Read the provided reasoning trace and extract a structured verdict.
Respond ONLY with valid JSON — no markdown, no explanation.
"""

EXTRACT_USER_TEMPLATE = """\
Based on the following fact-checking reasoning:

---
{cot_reasoning}
---

Extract the verdict as JSON:
{{
  "is_hallucination": <true or false>,
  "probability": <float 0.0-1.0, where 1.0 = definitely hallucinated>,
  "reasoning_summary": "<1-2 sentence summary of why>",
  "flagged_as_contradiction": <true if claim directly contradicts a known fact>,
  "flagged_as_unsupported": <true if claim goes beyond verifiable knowledge>
}}

JSON output:
"""


# ── Context-aware variant (for RAG faithfulness) ───────────────────────────────
# Used when a context document is available (faithfulness check).

COT_FAITHFULNESS_TEMPLATE = """\
EXPERT ADVICE (ground truth document):
{context}

USER QUESTION:
{user_query}

CANDIDATE STATEMENT (from an AI assistant's response):
{claim_text}

Think step-by-step:
1. Does the Candidate Statement agree with the Expert Advice?
2. Does the Candidate Statement contradict any part of the Expert Advice?
3. Does the Candidate Statement make claims NOT supported by the Expert Advice?
4. Is this a direct contradiction or an unsupported claim?
5. Final verdict: hallucinated (contradicts/unsupported) or faithful?

Write your full reasoning below:
"""


__all__ = [
    "COT_SYSTEM_PROMPT",
    "COT_USER_TEMPLATE",
    "EXTRACT_SYSTEM_PROMPT",
    "EXTRACT_USER_TEMPLATE",
    "COT_FAITHFULNESS_TEMPLATE",
]
