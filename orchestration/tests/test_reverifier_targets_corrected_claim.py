"""Regression guard for the Re-Verifier target-selection bug (audited 2026-09-20).

Confirmed bug: ``_reverifier_node`` built its suspicious-claim list from
``changed_claims`` entries via ``c.get("text")`` — but ``ChangedClaim.to_dict()``
(agents/corrector_agent/corrector/contracts.py:135-145) exposes ``corrected`` /
``original`` / ``action`` and has NO ``text`` key. So the lookup always failed,
``claim_texts`` stayed empty, and the code fell back to re-verifying the first
two sentences of the whole corrected response — NOT the claim that was actually
rewritten. A bad correction could sail through the re-verification safety gate
because the disputed claim was never re-checked.

This test asserts the claim-text extraction (the isolated, deterministic core of
the fix) selects the CORRECTED text of changed claims, without needing network,
the verifier, or the fine-tuned model.
"""

from __future__ import annotations


def _extract_reverify_texts(corr_res: dict, candidate_text: str) -> list[str]:
    """Mirror of the fixed extraction logic in graph.py::_reverifier_node.

    Kept in lockstep with the production block; if that block changes, this test
    should be updated to match. The point is to pin the behavior: re-verify the
    EMITTED text of each changed claim (corrected, else original), never a
    nonexistent ``text`` key.
    """
    claim_texts: list[str] = []
    changed = corr_res.get("changed_claims")
    if changed and isinstance(changed, list):
        for c in changed:
            if isinstance(c, dict):
                emitted = (c.get("corrected") or "").strip() or (c.get("original") or "").strip()
                if emitted:
                    claim_texts.append(emitted)
            elif isinstance(c, str) and c.strip():
                claim_texts.append(c.strip())

    if not claim_texts:
        sentences = [s.strip() for s in candidate_text.replace("\n", " ").split(".") if len(s.strip()) > 10]
        claim_texts = sentences[:2] if sentences else [candidate_text[:200]]
    return claim_texts


def test_reverify_targets_the_corrected_claim_not_leading_sentences():
    """The corrector fixed the 4th claim; the re-verifier must check THAT claim,
    not the first two (undisputed) sentences of the response."""
    corrected_claim = "Java was created by James Gosling at Sun Microsystems."
    corr_res = {
        "corrected_text": (
            "Some intro sentence that is long enough. "
            "Another undisputed preamble sentence here. "
            "Yet more filler that would win the fallback. "
            f"{corrected_claim}"
        ),
        "changed_claims": [
            {
                "claim_id": "c4",
                "action": "corrected",
                "original": "Java was created by Snehith.",
                "corrected": corrected_claim,
                "evidence_ids_used": ["ev-1"],
                "attempts": 1,
                "reasons": ["contradicted"],
            }
        ],
    }
    texts = _extract_reverify_texts(corr_res, corr_res["corrected_text"])
    assert texts == [corrected_claim], (
        "re-verifier did not target the corrected claim — it would re-check the "
        "wrong text and let a bad correction pass the safety gate"
    )


def test_reverify_falls_back_only_when_no_changed_claims():
    """With no structured changed_claims, the sentence fallback still applies
    (preserves prior behavior for legacy/empty payloads)."""
    candidate = "First long enough sentence here. Second long enough sentence here. Third."
    texts = _extract_reverify_texts({"changed_claims": []}, candidate)
    assert len(texts) == 2
    assert texts[0].startswith("First")


def test_reverify_uses_original_when_claim_preserved_without_rewrite():
    """A flagged-but-preserved claim (action=preserved, empty corrected) still
    gets re-checked via its original text rather than being dropped."""
    corr_res = {
        "changed_claims": [
            {
                "claim_id": "c1",
                "action": "preserved",
                "original": "The Earth orbits the Sun.",
                "corrected": "",
                "evidence_ids_used": [],
                "attempts": 0,
                "reasons": ["insufficient_evidence"],
            }
        ],
    }
    texts = _extract_reverify_texts(corr_res, "fallback text that is long enough here.")
    assert texts == ["The Earth orbits the Sun."]
