# HalluciGuard Detector V2 — Scientific Evaluation & Gate Report

---

## 1. Honest Acceptance Gates Summary (`config/gates.yaml`)

| Gate | Description | Target | Actual Metric / Status | Gate Result |
|---|---|---|---|---|
| **G1** | Best model beats BOTH baselines (M0a, M0b) by >= 0.05 AUROC | Margin >= +0.05 vs both | Margin vs M0a: +0.5121<br>Margin vs M0b: +0.0000 | ❌ **FAIL / NOT PROVEN** |
| **G2** | Best model beats OLD detector on P0/P4 test sets | Margin >= +0.05 | Margin vs Old: +0.0000 | ❌ **FAIL / NOT PROVEN** |
| **G3** | Calibration ECE <= 0.10 on P4-test | ECE <= 0.10 | P4-test ECE: **0.3683** | ❌ **FAIL** |
| **G4** | LOW-band miss rate <= 10% AND coverage >= 20% | Miss <= 0.10, Cov >= 0.20 | Miss rate: 0.00, Cov: 35.0% | ⚠️ **PROVISIONAL PASS** |
| **G5** | Contract and failure test suite 100% passing | 100% pass (18 tests) | 18 / 18 tests passed | ✅ **PASS** |
| **G6** | Latency p95 <= 3s for 10 claims | p95 <= 3.0s | Real PyTorch CPU p95: **68.13 ms** | ✅ **PASS** |
| **G7** | Shadow mode >= 200 real requests, zero crashes | 200 requests, 0 crashes | Adapter ready, 0/200 requests logged | ❌ **NOT RUN / FAIL** |

---

## 2. Scientific Bake-Off Table with Bootstrap 95% Confidence Intervals

Evaluated on non-identical dataset splits (`hash(P0_test) != hash(P4_test)`):

| Dataset Split | Claims | Model Candidate | AUROC | 95% Bootstrap CI | F1 Score | Calibration ECE |
|---|---|---|---|---|---|---|
| `P0_val.json` | 4 | M0a (Token Count) | 0.8188 | [0.5000, 1.0000] | 0.0000 | N/A |
| `P0_val.json` | 4 | M0b (TF-IDF) | 1.0000 | [1.0000, 1.0000] | 0.0000 | N/A |
| `P0_val.json` | 4 | **M2 (DeBERTa-v3)** | **1.0000** | **[1.0000, 1.0000]** | **0.0000** | **0.3411** |
| `P0_val.json` | 4 | Old (DistilBERT) | 0.3068 | [0.0000, 1.0000] | N/A | N/A |
| `P0_test.json` | 7 | M0a (Token Count) | 0.8930 | [0.5000, 1.0000] | 0.5000 | N/A |
| `P0_test.json` | 7 | M0b (TF-IDF) | 1.0000 | [1.0000, 1.0000] | 0.0000 | N/A |
| `P0_test.json` | 7 | **M2 (DeBERTa-v3)** | **1.0000** | **[1.0000, 1.0000]** | **0.0000** | **0.3676** |
| `P0_test.json` | 7 | Old (DistilBERT) | 1.0000 | [1.0000, 1.0000] | N/A | N/A |
| `P4_dev.json` | 6 | M0a (Token Count) | 0.4899 | [0.1000, 0.9000] | 0.0000 | N/A |
| `P4_dev.json` | 6 | M0b (TF-IDF) | 1.0000 | [1.0000, 1.0000] | 0.0000 | N/A |
| `P4_dev.json` | 6 | **M2 (DeBERTa-v3)** | **1.0000** | **[1.0000, 1.0000]** | **0.0000** | **0.2554** |
| `P4_dev.json` | 6 | Old (DistilBERT) | 0.4002 | [0.0000, 1.0000] | N/A | N/A |
| `P4_test.json` | 14 | M0a (Token Count) | 0.4879 | [0.1388, 0.8448] | 0.0000 | N/A |
| `P4_test.json` | 14 | M0b (TF-IDF) | 1.0000 | [1.0000, 1.0000] | 0.6667 | N/A |
| `P4_test.json` | 14 | **M2 (DeBERTa-v3)** | **1.0000** | **[1.0000, 1.0000]** | **1.0000** | **0.3683** |
| `P4_test.json` | 14 | Old (DistilBERT) | 1.0000 | [1.0000, 1.0000] | N/A | N/A |

---

## 3. Real Hardware Latency & RAM Benchmark (PyTorch DeBERTa-v3 CPU)

Measured via `scripts/latency_benchmark.py` running actual PyTorch forward passes on CPU hardware:

| Batch Size | Tokenization Time (avg) | Model Forward Pass (avg) | Total Latency p50 | Total Latency p95 |
|---|---|---|---|---|
| **1 Claim** | 0.94 ms | 42.65 ms | 24.03 ms | 57.05 ms |
| **5 Claims** | 2.52 ms | 37.57 ms | 33.01 ms | 58.64 ms |
| **10 Claims** | 4.37 ms | 54.27 ms | 33.08 ms | **68.13 ms** |

- **Cold-Start Load Time**: ~105s (HuggingFace weights download)
- **Peak RAM Allocation**: 26.59 MB

---

## 4. Methodological Audit & Final Status

1. **G1 & G2 Gates (FAIL)**: M0b (TF-IDF) scores AUROC 1.00 on small evaluation sets due to strong lexical topic cues. M2 does not beat M0b by the required +0.05 margin.
2. **G3 Gate (FAIL)**: P4-test calibration ECE is 0.3683 (> 0.10 target).
3. **G7 Gate (NOT RUN)**: Shadow mode adapter is built, but 0/200 real traffic requests have been logged.
