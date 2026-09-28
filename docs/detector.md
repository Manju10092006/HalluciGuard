# Detector documentation

The canonical Detector guide and model card live with the implementation:

- [Detector implementation, inference, failure behavior and training](../halluciguard_detector/README.md)
- [Checkpoint model card and held-out metrics](../halluciguard_detector/MODEL_CARD.md)
- [Training and live experiments](experiments.md)
- [Research provenance](research.md)

The production graph performs evidence-free triage before retrieval and trained, calibrated grounded inference after Verifier evidence. The second pass—not the triage placeholder—is the model result consumed by Judge.
