"""HalluciGuard Detector Agent — Identifies suspicious claims in LLM-generated outputs.

Responsible for:
    - Parsing LLM output into individual claims
    - Scoring claim suspiciousness using perplexity, entity density, and hedging analysis
    - Classifying claims by domain (healthcare, cybersecurity, finance, legal, AI research, etc.)
    - Generating SuspiciousClaim objects for the Verifier Agent

Input: Raw LLM text output
Output: List of SuspiciousClaim with claim_id, text, domain, suspicion_score

Status: NOT YET IMPLEMENTED — Awaiting contributor implementation.
"""
