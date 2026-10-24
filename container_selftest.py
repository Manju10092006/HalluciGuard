"""Build-time actual Detector pair smoke; not model-quality or graph certification."""
import hashlib
import json
import math
from pathlib import Path


def main():
    root = Path("/app")
    manifest = json.loads((root / "source-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Reviewed source identity mismatch: " + name)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from halluciguard_detector.detector import Detector
    # Pair classification deliberately avoids retrieval model loading here.
    detector = Detector(root / "artifacts/detector-best", "cpu", evidence=object())
    probabilities = detector._classify("Hanoi is the capital of Vietnam.", "Hanoi is the capital of Vietnam.")
    scores = {label.value: value for label, value in probabilities.items()}
    assert set(scores) == {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in scores.values())
    assert abs(sum(scores.values()) - 1) < 0.00001
    from fastapi.testclient import TestClient
    from orchestration.api import app
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    print(json.dumps({"scope": "Actual CPU Detector pair and shallow HTTP health only",
                      "probabilities": scores, "review_commit": manifest["review_commit"],
                      "model_quality_certified": False, "seven_agent_certified": False,
                      "memory_durable": False}))


if __name__ == "__main__":
    main()
