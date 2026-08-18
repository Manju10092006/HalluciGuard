"""Test script for the Decision Intelligence Engine — Verifier Contract Alignment."""
import sys, os, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decision_intelligence import DecisionIntelligenceEngine

engine = DecisionIntelligenceEngine()

# Test 1: Verifier Contract with Claims Schema (CONTRADICTED -> CORRECT)
print("=" * 70)
print("TEST 1: Verifier Claim Schema — Contradicted Python Claim")
print("=" * 70)
v1 = engine.evaluate(
    user_query="Who created Python and when?",
    draft_response="Python was created by Elon Musk in 1999.",
    detector_output={"hallucination_probability": 0.45, "confidence_score": 0.88},
    verifier_output={
        "claims": [
            {
                "claim_id": "C1",
                "claim_text": "Python was created by Elon Musk in 1999.",
                "verdict": "CONTRADICTED",
                "confidence_score": 0.97,
                "trust_score": 0.94,
                "explanation": "Authoritative evidence identifies Guido van Rossum as the creator of Python.",
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "title": "Python History",
                        "source": "Wikipedia",
                        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                        "snippet": "Python was created by Guido van Rossum in 1991.",
                        "entailment_label": "contradiction",
                        "entailment_score": 0.98,
                        "credibility_score": 0.80
                    }
                ]
            }
        ]
    },
    domain="General Knowledge"
)

print(f"Overall Decision: {v1.overall_decision}")
print(f"Correction Required: {v1.correction_required}")
print(f"Re-verification Required: {v1.re_verification_required}")
for cd in v1.claim_decisions:
    print(f"  Claim [{cd['claim_id']}]: Status={cd['status']} | Action={cd['action']} | EvidenceIDs={cd['evidence_ids']}")
    print(f"    Reason: {cd['reason']}")
print()

# Test 2: Cybersecurity — CVE claim
print("=" * 70)
print("TEST 2: Cybersecurity — Verified CVE Claim")
print("=" * 70)
v2 = engine.evaluate(
    user_query="Is CVE-2024-1234 a critical vulnerability?",
    draft_response="CVE-2024-1234 is a critical RCE vulnerability in Apache Log4j allowing remote code execution.",
    detector_output={"hallucination_probability": 0.20, "confidence_score": 0.90},
    verifier_output={"claim_evidence_pairs": [
        {"claim": "CVE-2024-1234 is a critical RCE vulnerability in Apache Log4j",
         "evidence": "CVE-2024-1234: Critical remote code execution vulnerability in Apache Log4j 2.x. CVSS: 9.8.",
         "source": "NVD - National Vulnerability Database"}
    ]},
    domain="Cybersecurity"
)
print(f"Overall Decision: {v2.overall_decision}")
for cd in v2.claim_decisions:
    print(f"  Claim [{cd['claim_id']}]: Status={cd['status']} | Action={cd['action']}")
print()

# Test 3: Finance with numeric mismatch
print("=" * 70)
print("TEST 3: Finance — Numeric Revenue Mismatch")
print("=" * 70)
v3 = engine.evaluate(
    user_query="What was Apple's 2023 revenue?",
    draft_response="Apple reported revenue of $450 billion in FY2023.",
    detector_output={"hallucination_probability": 0.35, "confidence_score": 0.88},
    verifier_output={"claim_evidence_pairs": [
        {"claim": "Apple reported revenue of $450 billion in FY2023",
         "evidence": "Apple Inc. reported total net revenue of $383.3 billion for fiscal year 2023.",
         "source": "SEC EDGAR 10-K Filing"}
    ]},
    domain="Finance"
)
print(f"Overall Decision: {v3.overall_decision}")
for cd in v3.claim_decisions:
    print(f"  Claim [{cd['claim_id']}]: Status={cd['status']} | Action={cd['action']}")
print()

print("=" * 70)
print("ALL UNIT TESTS COMPLETE")
print("=" * 70)
