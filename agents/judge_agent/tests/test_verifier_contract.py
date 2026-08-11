from __future__ import annotations

from judge_agent import JudgeAgent


def test_verifier_v2_claim_evidence_is_normalized_for_judge():
    verifier_output = {
        "domain": "healthcare",
        "claim_evidence": [
            {
                "claim_id": "c1",
                "claim_text": "The treatment reduced symptoms.",
                "verdict": "verified",
                "trust_score": 0.91,
                "evidence": [
                    {
                        "title": "Clinical study",
                        "source": "pubmed",
                        "url": "https://example.com/study",
                        "publication_date": "2026-01-01",
                        "snippet": "The treatment reduced symptoms in the study group.",
                        "entailment_label": "entailment",
                        "entailment_score": 0.95,
                        "credibility_score": 0.97,
                    }
                ],
            }
        ],
    }

    pairs = JudgeAgent._normalize_verifier_output(verifier_output)

    assert len(pairs) == 1
    assert pairs[0]["claim"] == "The treatment reduced symptoms."
    assert pairs[0]["evidence"].startswith("The treatment reduced symptoms")
    assert pairs[0]["source"] == "pubmed"
    assert pairs[0]["publication_date"] == "2026-01-01"
    assert pairs[0]["verifier_verdict"] == "verified"
    assert pairs[0]["verifier_trust_score"] == 0.91
    assert 0.0 <= pairs[0]["evidence_confidence"] <= 1.0


def test_existing_claim_evidence_pairs_remain_backward_compatible():
    verifier_output = {
        "claim_evidence_pairs": [
            {"claim": "A", "evidence": "B", "source": "source_a"}
        ]
    }

    pairs = JudgeAgent._normalize_verifier_output(verifier_output)

    assert pairs == verifier_output["claim_evidence_pairs"]


def test_verifier_claim_without_evidence_is_not_fabricated():
    verifier_output = {
        "claim_evidence": [
            {
                "claim_id": "c1",
                "claim_text": "Unsupported claim",
                "verdict": "insufficient_evidence",
                "evidence": [],
            }
        ]
    }

    assert JudgeAgent._normalize_verifier_output(verifier_output) == []
