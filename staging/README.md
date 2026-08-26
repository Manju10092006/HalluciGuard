---
title: HalluciGuard Detector Agent
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.0
app_file: gradio_app.py
pinned: false
---

# 🛡️ HalluciGuard Detector Agent (HaluEval)

Production-ready hallucination detection service and interactive Gradio Space for the **HalluciGuard Detector Agent**.

The agent loads the fine-tuned HaluEval classification model directly from Hugging Face Hub (`Manjunath2000006/halluciguard-detector`) and evaluates hallucination risk in LLM generations.

---

## 📁 Space Repository Structure

```
├── gradio_app.py               # Hugging Face Spaces Gradio entrypoint
├── app.py                      # FastAPI REST API entrypoint (/detect, /health, /model-info)
├── requirements.txt            # Runtime dependencies
├── .env.example                # Configuration template
├── README.md                   # Space metadata & service documentation
├── detector/                   # Isolated Detector Agent runtime
│   ├── __init__.py             # Package exports
│   ├── detector.py             # DetectorAgent with execution diagnostics & routing
│   ├── halueval_inference.py   # Model inference loader (HF Hub integration)
│   ├── halueval_dataset.py     # Canonical format_detector_input() formatter
│   ├── config.py               # DetectorConfig with threshold & env settings
│   └── models.py               # Pydantic schemas (DetectionInput, DetectionResult)
└── tests/
    └── test_detector_staging.py # Integration test suite
```

---

## 🎯 Risk Classification & Routing Policy

| Hallucination Probability | Risk Level | Next Action | Routing Logic |
| :--- | :--- | :--- | :--- |
| `P <= 0.30` | **LOW** | `ACCEPT` | Direct pass-through |
| `0.30 < P < 0.50` | **MEDIUM** | `ACCEPT` | Direct pass-through |
| `P >= 0.50` | **HIGH** | `VERIFY` | Route to Verifier Agent |

---

## 🚀 Live Endpoints

- **Interactive UI**: Gradio interface on root URL (`/`)
- **FastAPI Endpoints**:
  - `POST /detect` — JSON hallucination risk analysis
  - `GET /health` — Health check & model readiness
  - `GET /model-info` — Classifier metadata and routing policies
