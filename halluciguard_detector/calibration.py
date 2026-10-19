"""Calibration and decision thresholds for the three-class detector.

The detector emits a three-class softmax over
``SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO``. That distribution supports **two
different binary questions**, and they must never share a threshold:

``contradiction``
    Is the evidence refuting the claim? Score ``P(CONTRADICTED)``,
    positive class ``CONTRADICTED``, negatives ``SUPPORTED +
    NOT_ENOUGH_INFO``.

``verification_needed``
    Does this claim still require checking? Score
    ``P(CONTRADICTED) + P(NOT_ENOUGH_INFO)``, positive class
    ``CONTRADICTED + NOT_ENOUGH_INFO``, negatives ``SUPPORTED``.

A single ``hallucination_threshold`` used to serve the second question while
being *named* after the first, which is how a verification-risk score ended up
being compared against falsity. Both thresholds are now stored and read
separately.

A softmax output is a model score, not a calibrated probability of real-world
truth. Only temperature scaling plus the reliability numbers in
``training.calibration_report`` say anything about calibration, and even then
only for the data distribution they were measured on.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


#: Written to ``calibration.json`` for backward compatibility only. It is a
#: mirror of ``verification_risk_threshold`` and must never be used to decide
#: contradiction.
DEPRECATED_HALLUCINATION_THRESHOLD = "hallucination_threshold"

#: Single source of truth for sequence length, shared by training, evaluation
#: and calibration persistence. Must equal the shipped checkpoint's
#: ``tokenizer_config.json`` ``model_max_length``. It lives here rather than in
#: ``training`` because ``training`` already imports this module.
DEFAULT_MAX_LENGTH = 256


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if not math.isfinite(float(temperature)) or temperature <= 0 or not torch.isfinite(logits).all():
        raise ValueError("temperature and logits must be finite; temperature must be positive")
    return torch.softmax(logits / max(temperature, 1e-4), dim=-1)


def class_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Return the calibrated three-class distribution for raw logits."""
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = torch.tensor(np.asarray(logits, dtype=np.float32)) / max(float(temperature), 1e-4)
    return apply_temperature(scaled, 1.0).numpy()


def contradiction_score(probabilities: np.ndarray) -> np.ndarray:
    """``P(CONTRADICTED)`` -- the only score that speaks about falsity."""
    return probabilities[:, 1]


def verification_risk_score(probabilities: np.ndarray) -> np.ndarray:
    """``P(CONTRADICTED) + P(NOT_ENOUGH_INFO)`` -- the operational triage score."""
    return probabilities[:, 1] + probabilities[:, 2]


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    values = torch.tensor(logits, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(values / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(0.05, 10.0).item())


def save_calibration(
    path: Path,
    temperature: float,
    contradiction_threshold: float,
    verification_risk_threshold: float,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> None:
    """Persist temperature plus both task-specific thresholds.

    ``hallucination_threshold`` is retained as a deprecated mirror of
    ``verification_risk_threshold`` so older readers keep their previous value
    instead of silently picking up a different number.
    """
    if path.exists():
        raise FileExistsError("calibration artifact already exists")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid calibration temperature")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (contradiction_threshold, verification_risk_threshold)):
        raise ValueError("invalid calibration thresholds")
    path.write_text(
        json.dumps(
            {
                "temperature": float(temperature),
                "contradiction_threshold": float(contradiction_threshold),
                "verification_risk_threshold": float(verification_risk_threshold),
                DEPRECATED_HALLUCINATION_THRESHOLD: float(verification_risk_threshold),
                "max_length": int(max_length),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_calibration(path: Path) -> dict[str, Any]:
    """Read calibration metadata, tolerating the deprecated threshold key.

    Older checkpoints only carry ``hallucination_threshold``. That value was
    fitted for the verification-needed question, so it is mapped onto
    ``verification_risk_threshold`` rather than onto
    ``contradiction_threshold`` -- guessing a contradiction threshold from it
    would reintroduce exactly the confusion this module exists to remove.
    """
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("calibration must be an object")
    # Use `.get(...) is not None` (not `in`): an explicit JSON null is treated as
    # absent rather than crashing float(None)/int(None) with an undocumented
    # TypeError the legacy-tolerance path below was meant to handle (audit M8).
    if data.get("temperature") is not None and (not math.isfinite(float(data["temperature"])) or float(data["temperature"]) <= 0):
        raise ValueError("invalid calibration temperature")
    for key in ("contradiction_threshold", "verification_risk_threshold", DEPRECATED_HALLUCINATION_THRESHOLD):
        if data.get(key) is not None and (not math.isfinite(float(data[key])) or not 0 <= float(data[key]) <= 1):
            raise ValueError(f"invalid calibration {key}")
    if data.get("max_length") is not None and int(data["max_length"]) <= 0:
        raise ValueError("invalid calibration max_length")
    legacy = data.get(DEPRECATED_HALLUCINATION_THRESHOLD)
    if data.get("verification_risk_threshold") is None and legacy is not None:
        data["verification_risk_threshold"] = float(legacy)
    return data
