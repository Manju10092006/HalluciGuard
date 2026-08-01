"""Test script for the Decision Intelligence Engine."""
import sys, os, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decision_intelligence import DecisionIntelligenceEngine

engine = DecisionIntelligenceEngine()

# Test 1: Healthcare with conflicting evidence
print("=" * 70)
print("TEST 1: Healthcare — Contradicting drug dosage claim")
print("=" * 70)
v1 = engine.evaluate(
    user_query="What is the max daily dose of ibuprofen?",
    draft_response="The max daily dose of ibuprofen is 3200mg for adults.",
    detector_output={"hallucination_probability": 0.45, "confidence_score": 0.82},
    verifier_output={"claim_evidence_pairs": [
        {"claim": "The max daily dose of ibuprofen is 3200mg for adults.",
         "evidence": "OTC ibuprofen maximum is 1200mg/day. Prescription max is 3200mg under medical supervision only.",
         "source": "FDA Drug Label"}
    ]},
    domain="Healthcare"
)
print(f"Decision: {v1.decision.value}")
print(f"Severity: {v1.severity.value}")
print(f"Risk Level: {v1.risk_assessment['level']}")
print(f"Safe to Release: {v1.risk_assessment['safe_to_release']}")
print(f"Evidence Quality: {v1.evidence_governance['quality']}")
for cv in v1.claim_verdicts:
    print(f"  Claim: {cv['claim_text'][:60]}... -> {cv['status_label']}")
print()
print("Reasoning:")
for step in v1.reasoning_chain:
    print(f"  {step}")
print()

# Test 2: Cybersecurity — CVE claim
print("=" * 70)
print("TEST 2: Cybersecurity — CVE vulnerability claim")
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
print(f"Decision: {v2.decision.value}")
print(f"Severity: {v2.severity.value}")
print(f"Risk Level: {v2.risk_assessment['level']}")
for cv in v2.claim_verdicts:
    print(f"  Claim: {cv['claim_text'][:60]}... -> {cv['status_label']}")
print()

# Test 3: General Knowledge — No conflicts
print("=" * 70)
print("TEST 3: General Knowledge — Well-supported claim")
print("=" * 70)
v3 = engine.evaluate(
    user_query="Who created Python?",
    draft_response="Python was created by Guido van Rossum and first released in 1991.",
    detector_output={"hallucination_probability": 0.05, "confidence_score": 0.95},
    verifier_output={"claim_evidence_pairs": [
        {"claim": "Python was created by Guido van Rossum",
         "evidence": "Python was conceived by Guido van Rossum at CWI in the Netherlands.",
         "source": "Wikipedia"},
        {"claim": "first released in 1991",
         "evidence": "Python was first released in 1991 as version 0.9.0.",
         "source": "Python.org official documentation"}
    ]},
    domain="General Knowledge"
)
print(f"Decision: {v3.decision.value}")
print(f"Severity: {v3.severity.value}")
print(f"Risk Level: {v3.risk_assessment['level']}")
for cv in v3.claim_verdicts:
    print(f"  Claim: {cv['claim_text'][:60]}... -> {cv['status_label']}")
print()

# Test 4: High hallucination, no evidence
print("=" * 70)
print("TEST 4: Healthcare — High hallucination, no evidence")
print("=" * 70)
v4 = engine.evaluate(
    user_query="What are the side effects of a new drug?",
    draft_response="The drug causes liver failure in 50% of patients.",
    detector_output={"hallucination_probability": 0.85, "confidence_score": 0.75},
    verifier_output={"claim_evidence_pairs": []},
    domain="Healthcare"
)
print(f"Decision: {v4.decision.value}")
print(f"Severity: {v4.severity.value}")
print(f"Risk Level: {v4.risk_assessment['level']}")
print(f"Safe to Release: {v4.risk_assessment['safe_to_release']}")
print()

# Test 5: Finance with numeric mismatch
print("=" * 70)
print("TEST 5: Finance — Numeric revenue mismatch")
print("=" * 70)
v5 = engine.evaluate(
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
print(f"Decision: {v5.decision.value}")
print(f"Severity: {v5.severity.value}")
print(f"Risk Level: {v5.risk_assessment['level']}")
for cv in v5.claim_verdicts:
    print(f"  Claim: {cv['claim_text'][:60]}... -> {cv['status_label']}")
    print(f"    Conflict: {cv['conflict_type']} — {cv['conflict_implication'][:80]}")
print()

print("=" * 70)
print("ALL 5 TESTS COMPLETE")
print("=" * 70)
