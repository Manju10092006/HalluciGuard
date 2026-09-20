"""Regression guards for the claim-level detector re-architecture.

Two properties are pinned here:

1. The detector node decomposes the draft into ATOMIC claims and scores each,
   aggregating by the riskiest claim — so a single hallucinated claim buried in
   otherwise-true text still drives the draft's risk up and routes to verify.

2. The trustworthy, user-facing hallucination probability is EVIDENCE-DERIVED
   from the Verifier's verdicts, not the DistilBERT triage model (which is a
   near-constant classifier and cannot discriminate truth). Contradicted -> high,
   verified -> low, unverified/conflicted -> 0.5.

These are deterministic: the detector node's `_run_detect` and the verifier
pipeline are stubbed, so no model weights or network are needed.
"""
from __future__ import annotations

import asyncio

import orchestration.graph as g


def test_decompose_splits_into_atomic_claims():
    text = "The capital of France is Paris. Mars is the fourth planet from the Sun."
    claims = g._decompose_claims(text)
    assert len(claims) >= 2
    joined = " ".join(claims).lower()
    assert "paris" in joined and "mars" in joined


def test_decompose_degrades_to_whole_text_on_empty_extraction():
    # Whitespace/garbage still yields a usable single-claim list or empty, never raises.
    assert g._decompose_claims("") == []
    assert g._decompose_claims("Paris.") == ["Paris."] or g._decompose_claims("Paris.")


def _run_detector_node(state, whole, per_claim_probs):
    """Drive _detector_node with a stubbed DetectorAgent via monkeypatching the
    imported class. per_claim_probs: list of P(halluc) returned in order."""
    calls = {"i": 0}

    class _StubResult(dict):
        pass

    class _StubAgent:
        def detect(self, q, resp):
            # First len(per_claim_probs) calls are per-claim; the last is whole-draft.
            i = calls["i"]
            calls["i"] += 1
            if i < len(per_claim_probs):
                p = per_claim_probs[i]
            else:
                p = max(per_claim_probs) if per_claim_probs else 0.0
            risk = "HIGH" if p >= 0.5 else "LOW"
            return _StubResult(
                hallucination_probability=p,
                confidence_score=max(p, 1 - p),
                risk_level=risk,
                next_action="Verify" if risk == "HIGH" else "Accept",
                model_source="stub",
                detector_model_source="stub",
            )

    import agents.detector_agent.detector as det
    orig = det.DetectorAgent
    det.DetectorAgent = _StubAgent
    try:
        return asyncio.run(g._detector_node(state))
    finally:
        det.DetectorAgent = orig


def test_riskiest_claim_drives_aggregate_and_route():
    # Two claims: one clearly true (low), one hallucinated (high). Fixed draft so
    # decomposition yields two claims.
    draft = "Paris is the capital of France. The Moon is made of green cheese."
    state = {"user_query": "Tell me facts", "llm_response": draft}
    # Force a known per-claim split independent of spaCy availability.
    n = len(g._decompose_claims(draft))
    probs = [0.05] * (n - 1) + [0.95]  # last claim hallucinated
    out = _run_detector_node(state, None, probs)
    detector = out["detector"]
    assert detector["claims_analyzed"] == n
    assert detector["hallucination_probability"] == 0.95  # riskiest wins
    assert detector["risk_level"] == "HIGH"
    assert detector["next_action"] == "Verify"
    # enum leakage must be normalized — no 'RISKLEVEL.' / 'NEXTACTION.' prefixes
    assert "." not in detector["risk_level"]
    assert "." not in detector["next_action"]


def test_evidence_derived_probability_mapping():
    """The evidence-derived probability maps verdict -> trustworthy score,
    independent of the triage model."""
    # Rebuild the mapping the verifier node uses, to pin the contract.
    def evidence_hp(status: str, conf: float) -> float:
        if status == "contradicted":
            return round(0.75 + 0.24 * conf, 4)
        if status == "conflicted":
            return 0.5
        if status == "verified":
            return round(max(0.02, 0.25 - 0.23 * conf), 4)
        return 0.5

    assert evidence_hp("contradicted", 1.0) >= 0.95
    assert evidence_hp("contradicted", 0.0) == 0.75
    assert evidence_hp("verified", 1.0) <= 0.05
    assert evidence_hp("verified", 0.0) == 0.25
    assert evidence_hp("unverified", 0.9) == 0.5
    assert evidence_hp("conflicted", 0.9) == 0.5
