"""
HalluciGuard Judge Agent - Interactive CLI Simulator
Test the Decision Intelligence Engine from the terminal with preset scenarios
or custom inputs. Full reasoning chain and claim verdicts displayed.
"""

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decision_intelligence import DecisionIntelligenceEngine

engine = DecisionIntelligenceEngine()

PRESET_SCENARIOS = {
    "1": {
        "label": "Healthcare: Correct drug dosage, authoritative source",
        "query": "What is the adult dosage of ibuprofen?",
        "response": "Adults take 200-400mg ibuprofen every 4-6 hours. Maximum 1200mg per day OTC.",
        "detector": {"hallucination_probability": 0.08, "confidence_score": 0.92},
        "verifier": {"claim_evidence_pairs": [
            {"claim": "Adults take 200-400mg ibuprofen every 4-6 hours. Maximum 1200mg per day OTC.",
             "evidence": "OTC dosage: 200-400mg every 4-6 hours. Do not exceed 1200mg in 24 hours.",
             "source": "FDA Drug Label"}
        ]},
        "domain": "Healthcare"
    },
    "2": {
        "label": "Healthcare: Dangerous hallucinated drug interaction",
        "query": "Can I take aspirin with warfarin?",
        "response": "Yes, combining aspirin and warfarin daily is perfectly safe.",
        "detector": {"hallucination_probability": 0.85, "confidence_score": 0.88},
        "verifier": {"claim_evidence_pairs": [
            {"claim": "Combining aspirin and warfarin daily is perfectly safe.",
             "evidence": "Aspirin is contraindicated with warfarin due to increased risk of fatal bleeding. Do not combine without medical supervision.",
             "source": "FDA Drug Label"}
        ]},
        "domain": "Healthcare"
    },
    "3": {
        "label": "Finance: Numeric revenue mismatch",
        "query": "What was Apple's FY2023 revenue?",
        "response": "Apple reported $450 billion in total revenue for fiscal year 2023.",
        "detector": {"hallucination_probability": 0.35, "confidence_score": 0.88},
        "verifier": {"claim_evidence_pairs": [
            {"claim": "Apple reported $450 billion in total revenue for fiscal year 2023.",
             "evidence": "Apple Inc. total net revenue for fiscal year 2023: $383.3 billion.",
             "source": "SEC EDGAR 10-K Filing"}
        ]},
        "domain": "Finance"
    },
    "4": {
        "label": "Cybersecurity: Verified CVE with dual sources",
        "query": "Is CVE-2021-44228 critical?",
        "response": "CVE-2021-44228 (Log4Shell) is a critical RCE vulnerability in Apache Log4j 2.x.",
        "detector": {"hallucination_probability": 0.05, "confidence_score": 0.95},
        "verifier": {"claim_evidence_pairs": [
            {"claim": "CVE-2021-44228 is a critical RCE vulnerability in Apache Log4j 2.x.",
             "evidence": "CVE-2021-44228: JNDI injection vulnerability in Apache Log4j2. CVSS 10.0 Critical.",
             "source": "NVD - National Vulnerability Database"},
            {"claim": "CVE-2021-44228 is a critical RCE vulnerability in Apache Log4j 2.x.",
             "evidence": "Log4Shell is actively exploited. All organizations should patch Log4j immediately.",
             "source": "CISA Advisory"}
        ]},
        "domain": "Cybersecurity"
    },
    "5": {
        "label": "General Knowledge: Well-supported fact",
        "query": "Who created Python?",
        "response": "Python was created by Guido van Rossum and first released in 1991.",
        "detector": {"hallucination_probability": 0.03, "confidence_score": 0.97},
        "verifier": {"claim_evidence_pairs": [
            {"claim": "Python was created by Guido van Rossum",
             "evidence": "Python was conceived by Guido van Rossum at CWI in the Netherlands.",
             "source": "Wikipedia"},
            {"claim": "first released in 1991",
             "evidence": "Python 0.9.0 was released in February 1991.",
             "source": "Python.org"}
        ]},
        "domain": "General Knowledge"
    },
    "6": {
        "label": "Entertainment: Low-risk movie fact",
        "query": "Who directed Inception?",
        "response": "Inception was directed by Christopher Nolan and released in 2010.",
        "detector": {"hallucination_probability": 0.02, "confidence_score": 0.98},
        "verifier": {"claim_evidence_pairs": [
            {"claim": "Inception was directed by Christopher Nolan and released in 2010.",
             "evidence": "Inception is a 2010 science fiction action film written and directed by Christopher Nolan.",
             "source": "Wikipedia"}
        ]},
        "domain": "Entertainment"
    },
    "7": {
        "label": "Healthcare: High hallucination, ZERO evidence",
        "query": "What are the effects of a new experimental drug?",
        "response": "The drug cures cancer in 95% of patients with no side effects.",
        "detector": {"hallucination_probability": 0.92, "confidence_score": 0.80},
        "verifier": {"claim_evidence_pairs": []},
        "domain": "Healthcare"
    },
    "8": {
        "label": "Mixed: One claim verified, one contradicted",
        "query": "Tell me about the speed of light",
        "response": "The speed of light is approximately 300,000 km/s. Einstein discovered it in 1820.",
        "detector": {"hallucination_probability": 0.40, "confidence_score": 0.75},
        "verifier": {"claim_evidence_pairs": [
            {"claim": "The speed of light is approximately 300,000 km/s.",
             "evidence": "The speed of light in vacuum is 299,792,458 metres per second (approximately 300,000 km/s).",
             "source": "NIST - National Institute of Standards"},
            {"claim": "Einstein discovered it in 1820.",
             "evidence": "The finite speed of light was first determined in 1676 by Ole Romer. Einstein published special relativity in 1905.",
             "source": "Nature Physics Journal"}
        ]},
        "domain": "Scientific Research"
    },
}


def display_verdict(verdict):
    d = verdict
    dec_icons = {"ACCEPT": "✅", "CORRECT": "🔧", "REJECT": "🛑",
                 "VERIFY_AGAIN": "🔄", "ABSTAIN": "⏸", "ESCALATE_HUMAN": "👨‍⚕️"}
    icon = dec_icons.get(d.decision.value, "❓")

    print()
    print("═" * 72)
    print(f"  {icon}  DECISION: {d.decision.value}   |   SEVERITY: {d.severity.value}")
    print("═" * 72)
    print()

    # Key Metrics
    print("  ┌─ KEY METRICS ─────────────────────────────────────────────┐")
    ra = d.risk_assessment
    eg = d.evidence_governance
    print(f"  │ Evidence Quality:  {eg['quality']:20s} │ Risk Level: {ra['level']:12s} │")
    print(f"  │ Safe to Release:   {'YES' if ra['safe_to_release'] else 'NO':20s} │ Human Review: {'YES' if ra['human_review'] else 'NO':8s} │")
    cov = d.coverage
    print(f"  │ Coverage:          {cov['status']:20s} │ Consensus:  {d.consensus['status'][:12]:12s} │")
    print(f"  │ Pipeline Health:   {d.runtime_health['health']:20s} │ Trustworthy: {'YES' if d.runtime_health['trustworthy'] else 'NO':8s} │")
    ds = d.detector_signal
    print(f"  │ Detector Halluc:   {ds['hallucination_probability']*100:5.1f}%{' ':14s} │ Det. Conf:  {ds['confidence_score']*100:5.1f}%{' ':5s} │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    # Claim Verdicts
    if d.claim_verdicts:
        print("  CLAIM-LEVEL VERDICTS:")
        print("  " + "-" * 68)
        for cv in d.claim_verdicts:
            status = cv["status_label"]
            claim = cv["claim_text"][:55] if cv["claim_text"] else "—"
            entail = cv["nli_entailment"] * 100
            contra = cv["nli_contradiction"] * 100
            conflict = cv["conflict_type"]
            print(f"  {status:16s} | {claim:55s}")
            print(f"  {'':16s} | Entail: {entail:4.0f}%  Contra: {contra:4.0f}%  Conflict: {conflict}")
            if cv["has_conflict"]:
                print(f"  {'':16s} | Implication: {cv['conflict_implication'][:60]}")
            print("  " + "-" * 68)
        print()

    # Reasoning Chain
    print("  REASONING CHAIN:")
    for step in d.reasoning_chain:
        print(f"    {step}")
    print()

    # Risk Factors
    if ra["factors"]:
        print("  RISK FACTORS:")
        for f in ra["factors"]:
            print(f"    ⚠ {f}")
    if ra["mitigating"]:
        print("  MITIGATING FACTORS:")
        for m in ra["mitigating"]:
            print(f"    ✓ {m}")
    print()

    # Workflow Action
    wa = d.workflow_action
    print(f"  NEXT ACTION: {wa['type']} → {wa['target']}")
    print(f"  Priority: {wa['priority']}")
    print(f"  Reasoning: {wa['reasoning']}")
    print()

    # Audit
    audit = d.audit_record
    print(f"  AUDIT: {audit['id']}  |  {audit['timestamp']}")
    print()


def run_interactive():
    print()
    print("=" * 72)
    print("  HALLUCIGUARD JUDGE AGENT — INTERACTIVE CLI SIMULATOR")
    print("  AI Decision Intelligence Platform v3.0")
    print("=" * 72)
    print()

    while True:
        print("  PRESET SCENARIOS:")
        print("  " + "-" * 60)
        for key, sc in PRESET_SCENARIOS.items():
            print(f"    [{key}] {sc['label']}")
        print(f"    [C] Custom input")
        print(f"    [Q] Quit")
        print()

        choice = input("  Select scenario > ").strip().upper()

        if choice == "Q":
            print("  Exiting.")
            break

        if choice == "C":
            print()
            query = input("  User Query: ")
            response = input("  Draft Response: ")
            det_prob = float(input("  Detector Hallucination Prob (0-1): ") or "0.5")
            det_conf = float(input("  Detector Confidence (0-1): ") or "0.8")
            domain = input("  Domain (Healthcare/Cybersecurity/Finance/General Knowledge/Entertainment): ") or "General Knowledge"
            claim = input("  Claim: ") or response
            evidence = input("  Evidence: ")
            source = input("  Source: ") or "Unknown"
            print()

            v = engine.evaluate(
                user_query=query,
                draft_response=response,
                detector_output={"hallucination_probability": det_prob, "confidence_score": det_conf},
                verifier_output={"claim_evidence_pairs": [
                    {"claim": claim, "evidence": evidence, "source": source}
                ] if evidence else []},
                domain=domain
            )
            display_verdict(v)

        elif choice in PRESET_SCENARIOS:
            sc = PRESET_SCENARIOS[choice]
            print(f"\n  Running: {sc['label']}")
            print(f"  Domain:  {sc['domain']}")
            print(f"  Query:   {sc['query']}")
            print(f"  Response: {sc['response'][:70]}{'...' if len(sc['response'])>70 else ''}")

            v = engine.evaluate(
                user_query=sc["query"],
                draft_response=sc["response"],
                detector_output=sc["detector"],
                verifier_output=sc["verifier"],
                domain=sc["domain"]
            )
            display_verdict(v)

        else:
            print("  Invalid selection.\n")


if __name__ == "__main__":
    run_interactive()
