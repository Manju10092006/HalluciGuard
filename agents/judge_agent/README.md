# HalluciGuard Judge Agent — AI Decision Intelligence Platform

> **Enterprise AI Operating System & Chief Decision Officer for Hallucination Governance**

---

## 📌 Executive Summary

The **HalluciGuard Judge Agent** is the **Chief Decision Officer (CDO)** of the HalluciGuard multi-agent platform. 

Unlike traditional decision engines driven by basic threshold rules or score calculations, the Judge Agent behaves as an **AI Operating System**. It does **not** duplicate the responsibilities of downstream agents (Detector, Verifier, Memory, or Corrector). Instead, it **governs the entire verification workflow**, evaluates evidence quality, system health, policy compliance, multi-source consensus, and domain-specific safety risks before issuing a reproducible, audit-ready decision.

---

## 🏛 Core Architectural Philosophy

| Principle | Description |
|---|---|
| **No Duplicate Scoring** | Does **not** re-calculate hallucination probability (Detector's job) or re-fetch evidence (Verifier's job). |
| **Trust Reasoning** | Evaluates whether the system, evidence, and verification process itself can be trusted. |
| **Domain Governance** | Enforces behavioral policies per domain (`Healthcare`, `Finance`, `Cybersecurity`, `Law`, `General Knowledge`, `Entertainment`). |
| **12-Phase Pipeline** | Executes an end-to-end governance pipeline from runtime health inspection to reproducible audit trail generation. |
| **Claim-Level Granularity** | Automatically extracts claims, aligns evidence, and computes per-claim hallucination risk scores ($0.0\%$–$100.0\%$). |

---

## 🔄 The 12-Phase Decision Pipeline

```
  1. GOVERNANCE CONTEXT       → Load domain-specific policy (strictness, required source tiers)
  2. CLAIM EXTRACTION         → Segment response into claims & align with evidence
  3. RUNTIME INSPECTION       → Validate pipeline health (API failures, degraded state)
  4. EVIDENCE GOVERNANCE      → Assess evidence quality, independence & source authority
  5. COVERAGE ANALYSIS        → Identify ungrounded or partially covered claims
  6. CONFLICT RESOLUTION      → Classify conflicts (Direct, Numeric, Temporal, Safety)
  7. CONSENSUS ANALYSIS       → Evaluate agreement among independent evidence sources
  8. MEMORY INTELLIGENCE      → Query historical hallucination patterns & source reliability
  9. RISK ASSESSMENT          → Compute non-numeric, policy-driven risk level
 10. DECISION REASONING       → Execute governance logic to select final verdict
 11. WORKFLOW ORCHESTRATION   → Determine next action & target agent (Corrector, Verifier, User, Human)
 12. EXPLAINABILITY & AUDIT   → Generate human-readable reasoning chain & audit record
```

---

## 📂 File Architecture (29 Subsystems)

All files reside inside `agents/judge_agent/`:

### 🎯 Core Decision Engine & Governance
- **`decision_intelligence.py`**: The central orchestrator (Chief Decision Officer) executing the 12-phase pipeline.
- **`decision_policies.py`**: Domain policy registry defining governance behavior per domain (`Healthcare`, `Finance`, `Cybersecurity`, etc.).
- **`domain_policies.py`**: Policy definitions, source tier requirements, and strictness levels.
- **`decision_engine.py`**: High-level interface wrapper for backward compatibility.
- **`judge_agent.py`**: Multi-agent message handler and orchestration adapter.

### 🛡 Evidence, Coverage & Source Authority
- **`evidence_governance.py`**: Evaluates evidence quality (`AUTHORITATIVE`, `STRONG`, `MODERATE`, `WEAK`, `ABSENT`).
- **`evidence_intelligence.py`**: Evidence set analysis and gap identification.
- **`source_reliability.py`**: Classifies sources into authority tiers (`OFFICIAL_STANDARD`, `PEER_REVIEWED`, `ENTERPRISE_VENDOR`, `COMMUNITY`, `UNVERIFIED`).
- **`source_consensus.py`**: Multi-source agreement and independence analyzer.
- **`consensus_analyzer.py`**: Source consensus scoring and discrepancy reporting.
- **`coverage_analyzer.py`**: Identifies unverified, partially covered, or fully covered claims.

### 🔍 Conflict, NLI & Risk Reasoning
- **`conflict_resolver.py`**: Detects and categorizes conflict types (`DIRECT_REFUTATION`, `NUMERIC_MISMATCH`, `TEMPORAL_MISMATCH`, `SAFETY_VIOLATION`).
- **`contradiction_analyzer.py`**: Refutation keyword, negation, and claim-evidence contradiction analysis.
- **`nli_engine.py`**: Natural Language Inference engine with HuggingFace pipeline and robust heuristic fallback.
- **`claim_criticality.py`**: Assesses claim domain risk (e.g., drug dosage, revenue figures, CVEs).
- **`risk_intelligence.py`**: Policy-driven risk evaluator (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFORMATIONAL`).

### 🧠 System Health, Memory & Multi-Agent Routing
- **`runtime_inspector.py`**: Pipeline health check before trusting results (`HEALTHY`, `DEGRADED`, `CRITICAL_FAILURE`).
- **`memory_intelligence.py`**: Integrates Memory Agent signals, recurring hallucination flags, and historical outcomes.
- **`memory_integration.py`**: Adapter layer for querying long-term agent memory.
- **`workflow_orchestrator.py`**: Routes decisions to target agents (`CORRECTOR`, `VERIFIER`, `HUMAN_REVIEWER`, `USER`).
- **`negotiation_engine.py`**: Generates structured, traceable cross-agent inter-process messages.
- **`confidence_calibrator.py`**: Calibrates multi-agent confidence signals.

### 📜 Explainability, Audit & Interface
- **`explainability.py`**: Generates step-by-step reasoning chains and alternatives considered.
- **`audit_trail.py`**: Produces immutable, audit-ready decision records with UUIDs and timestamps.
- **`config.py`**: Centralized enums (`Decision`, `Severity`, `EvidenceQuality`, `SystemHealth`, `SourceTier`) and dataclasses.

### 🌐 Dashboard, Benchmarks & Simulation
- **`app.py`**: Enterprise Flask web dashboard with live real-time simulation UI.
- **`benchmark.py`**: 11-scenario ground-truth evaluation suite across 6 domains (100% pass rate).
- **`simulator.py`**: Interactive CLI simulator with 8 preset domain scenarios.
- **`test_judge.py`**: Comprehensive unit test suite.

---

## ⚡ Decisions & Verdicts Summary

The Judge Agent issues one of 6 primary decisions:

| Decision | Icon | Trigger Condition | Target Action |
|---|:---:|---|---|
| **`ACCEPT`** | ✅ | Evidence is strong, safe to release, low risk | Release response to `USER` |
| **`CORRECT`** | 🔧 | Fixable hallucinations or numeric mismatches detected | Route to `CORRECTOR` agent |
| **`VERIFY_AGAIN`** | 🔄 | Evidence is insufficient, unverified, or lacks multi-source consensus | Route to `VERIFIER` agent for expanded search |
| **`REJECT`** | 🛑 | High hallucination risk with no evidence, or unfixable claims | Block response and output safe fallback to `USER` |
| **`ESCALATE_HUMAN`** | 👨‍⚕️ | Safety-critical contradiction (e.g., drug dosage/contraindication) | Escalate to `HUMAN_REVIEWER` |
| **`ABSTAIN`** | ⏸️ | Pipeline runtime failure or empty response | Terminate workflow with explanation |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure Python 3.9+ is installed:
```bash
pip install flask
```

### 2. Launch the Web Dashboard
```bash
python app.py
```
Open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

### 3. Run the Benchmark Suite (11 Tests)
```bash
python benchmark.py
```

### 4. Run Interactive CLI Simulator
```bash
python simulator.py
```

### 5. Run Unit Tests
```bash
python test_judge.py
```

---

## 🔌 API Reference (`/evaluate`)

### POST `/evaluate`

**Request Body:**
```json
{
  "user_query": "What is the recommended dosage of ibuprofen for adults?",
  "draft_response": "The recommended adult dosage of ibuprofen is 200-400mg every 4-6 hours. Do not exceed 1200mg per day OTC.",
  "domain": "Healthcare",
  "detector_output": {
    "hallucination_probability": 0.10,
    "confidence_score": 0.90
  },
  "verifier_output": {
    "claim_evidence_pairs": [
      {
        "claim": "The recommended adult dosage of ibuprofen is 200-400mg every 4-6 hours.",
        "evidence": "Adult OTC dosage: 200-400mg every 4 to 6 hours as needed.",
        "source": "FDA Drug Label"
      }
    ]
  }
}
```

**Response Body:**
```json
{
  "decision": "ACCEPT",
  "severity": "LOW",
  "risk_assessment": {
    "level": "LOW",
    "safe_to_release": true,
    "human_review": false
  },
  "evidence_governance": {
    "quality": "STRONG",
    "authoritative": 1,
    "sufficient": true
  },
  "claim_verdicts": [
    {
      "claim_text": "The recommended adult dosage of ibuprofen is 200-400mg every 4-6 hours",
      "hallucination_pct": "9.0%",
      "status": "VERIFIED",
      "status_label": "✅ Verified"
    }
  ],
  "workflow_action": {
    "type": "RELEASE_RESPONSE",
    "target": "USER",
    "priority": "NORMAL"
  }
}
```

---

## 📊 Benchmark Results

| Scenario | Domain | Expected Decision | Actual Decision | Status |
|---|---|---|---|:---:|
| `HC-001` | Healthcare | `ACCEPT` | `ACCEPT` | ✅ PASS |
| `HC-002` | Healthcare | `REJECT` | `REJECT` | ✅ PASS |
| `HC-003` | Healthcare | `ESCALATE_HUMAN` | `ESCALATE_HUMAN` | ✅ PASS |
| `FN-001` | Finance | `ACCEPT` | `ACCEPT` | ✅ PASS |
| `FN-002` | Finance | `CORRECT` | `CORRECT` | ✅ PASS |
| `CS-001` | Cybersecurity | `ACCEPT` | `ACCEPT` | ✅ PASS |
| `GK-001` | General Knowledge | `ACCEPT` | `ACCEPT` | ✅ PASS |
| `GK-002` | General Knowledge | `REJECT` | `REJECT` | ✅ PASS |
| `EN-001` | Entertainment | `ACCEPT` | `ACCEPT` | ✅ PASS |
| `EC-001` | Edge Case | `ABSTAIN` | `ABSTAIN` | ✅ PASS |
| `EC-002` | Edge Case | `CORRECT` | `CORRECT` | ✅ PASS |

**Overall Benchmark Score: 11/11 (100.0% Pass Rate)**

---

## 📄 License

Part of the **HalluciGuard** Multi-Agent Ecosystem.
