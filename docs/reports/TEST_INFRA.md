# E2E Test Infra: HalluciGuard Verifier

## Test Philosophy
- Opaque-box, requirement-driven. No dependency on internal implementation details.
- Multi-tier validation: Unit & Invariant Tests -> Contract & Integration Tests -> Hardening & Regression Tests -> Multi-Domain Benchmark Suite.

## Feature Inventory & Test Coverage
| # | Feature | Scope / Requirement | Tier 1 (Unit) | Tier 2 (Boundary) | Tier 3 (Integration) | Tier 4 (E2E Benchmark) |
|---|---------|---------------------|:-------------:|:-----------------:|:--------------------:|:----------------------:|
| F0 | Architecture & Baseline Freeze | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ (35 claims frozen) |
| F1 | Quality-Gated Retrieval & Fallback | ORIGINAL_REQUEST §R2 | 8 | 6 | ✓ | ✓ |
| F2 | Evidence Semantics & Honest Confidence | ORIGINAL_REQUEST §R3 | 12 | 8 | ✓ | ✓ (0% false verified) |
| F3 | Domain Adapters (WHO, PubMed, FDA) | ORIGINAL_REQUEST §R4 | 15 | 8 | ✓ | ✓ (multi-domain) |
| F4 | LangGraph Supervisor & Pipeline E2E | Acceptance Criteria | 10 | 5 | ✓ | ✓ |

## Test Architecture & Invocation
- **Verifier Unit & Quality Gate Tests**:
  ```powershell
  pytest agents/verifier_agent/tests/test_retrieval_quality_gate.py agents/verifier_agent/tests/test_healthcare_sources.py agents/verifier_agent/tests/test_domain_adapters.py agents/verifier_agent/tests/test_web_retriever.py -v
  ```
- **Evidence Semantics & Scoring Invariant Tests**:
  ```powershell
  pytest agents/verifier_agent/tests/test_evidence_semantics.py agents/verifier_agent/tests/test_pipeline_hardening.py agents/verifier_agent/tests/test_v2_regression_failures.py -v
  ```
- **Full Verifier Test Suite**:
  ```powershell
  pytest agents/verifier_agent/tests -v
  ```
- **Memory Agent Test Suite**:
  ```powershell
  pytest agents/memory_agent/tests -v
  ```
- **Judge Agent Scenario Benchmark**:
  ```powershell
  python agents/judge_agent/benchmark.py
  ```
- **Pre-flight Deployment Readiness Check**:
  ```powershell
  python scripts/deployment_readiness_check.py
  ```
- **End-to-End Multi-Domain 35-Claim Benchmark**:
  ```powershell
  python scripts/benchmark_eval.py --use-cache
  ```

## Acceptance Criteria Thresholds
1. **0% False Verified Rate**: No negative, entity-swapped, relation-swapped, or contradicted claims may receive `VERIFIED`.
2. **Mutually Exclusive Evidence Invariant**: $\text{SUPPORTING} + \text{CONTRADICTING} + \text{NEUTRAL} + \text{IRRELEVANT} == \text{Total Passages}$.
3. **Canonical Verdict Enum**: Public verdicts strictly restricted to `["verified", "contradicted", "conflicted", "unverified"]`.
4. **Deterministic Pass Rate**: 100% pass on all deterministic unit, invariant, and contract tests.
5. **Accuracy Target**: Multi-domain benchmark accuracy strictly superior to Phase 0 baseline ($\ge 65.71\%$).
