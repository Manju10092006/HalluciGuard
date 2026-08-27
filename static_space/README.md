---
title: HalluciGuard Detector
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
---

# HalluciGuard Detector (Static / Transformers.js)

Real-time, in-browser hallucination detection powered by a fine-tuned HaluEval DistilBERT model (`Manjunath2000006/halluciguard-detector`).

### Architecture & Capabilities:
- **100% Client-Side Inference**: Runs directly in your browser using `@huggingface/transformers` and ONNX Runtime Web.
- **Zero Server Compute**: No backend server, no GPU instance, and no API rate limits.
- **Singleton Model Caching**: Downloaded once, cached in browser IndexedDB for instant repeat inferences.
- **FP32 Numerical Parity**: Exact alignment with the reference PyTorch implementation down to 5 decimal places.

### Thresholds & Routing Rules:
- **LOW** (`<= 0.30`): Next Action &rarr; **Accept**
- **MEDIUM** (`> 0.30` and `< 0.50`): Next Action &rarr; **Accept**
- **HIGH** (`>= 0.50`): Next Action &rarr; **Verify**
