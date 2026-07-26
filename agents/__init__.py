"""HalluciGuard Multi-Agent System — 5-Agent Trust Layer for LLM Output Verification.

Agents:
    - Detector Agent: Identifies suspicious claims in LLM outputs
    - Verifier Agent: Retrieves real evidence and scores claim support/contradiction
    - Judge Agent: Makes final accept/reject/flag decisions using evidence
    - Corrector Agent: Rewrites hallucinated content with verified facts
    - Memory Agent: Maintains persistent knowledge graph and learning history
"""
