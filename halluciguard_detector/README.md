# HalluciGuard Detector V2 — Standalone Hallucination Risk Detector

```
┌────────────────────────────────────────────┐
│ HALLUCIGUARD STANDALONE DETECTOR           │
├────────────────────────────────────────────┤
│ Implementation             ✅ COMPLETE     │
│ Architecture               ✅ COMPLETE     │
│ API                       ✅ COMPLETE     │
│ Fail-closed behavior       ✅ VERIFIED     │
│ Contract tests             ✅ 18/18        │
│ Import isolation           ✅ VERIFIED     │
│ Real inference benchmark   ✅ VERIFIED     │
│                                            │
│ Model superiority          ⚠️ UNPROVEN     │
│ Calibration                ❌ FAIL          │
│ Production test size      ❌ INSUFFICIENT  │
│ Shadow mode                ❌ NOT RUN      │
│ Production integration     ❌ NOT YET      │
└────────────────────────────────────────────┘
```

---

## 1. Architecture Overview

```
USER QUERY
   │
   ▼
BASE LLM (OpenRouter)
   │
DRAFT ANSWER
   │
   ▼
Claim Extractor (claim_extraction/)  ── FIRST (LLM + Sentence Fallback)
   │
[claim_001, claim_002, ...]          ── Atomic claims with immutable IDs
   │
   ▼
DeBERTa-v3-base (models/deberta.py)  ── PRIMARY CANDIDATE (M2)
   │
   ▼
Calibration & Prior-Shift Adjustment ── Platt / Isotonic scaling
   │
   ▼
DetectorResponse (schemas.py)        ── Output contract (VERIFY, UNKNOWN/DEEP)
```

---

## 2. Status & Next Scientific Steps

1. **Architecture & Software Contracts**: Complete, locked, and fully verified by 18 automated unit and failure tests.
2. **Fail-Closed Behavior**: Verified. Uncalibrated state returns `UNKNOWN` risk, `DEEP` verification hint, and `VERIFY` routing. Never returns `LOW_RISK`.
3. **Legacy Adapter**: Updated to map uncertainty to `UNKNOWN_UNCERTAIN` — never converts uncertainty to `NO_HALLUCINATION`.
4. **Next Step (Stage 3.5)**: Expand production dataset P4 to ~100–150 answers (~400–600 claims) with 2 human annotators (Cohen's kappa) before performing final scientific validation.
