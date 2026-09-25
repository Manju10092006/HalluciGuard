import json
from pathlib import Path

import numpy as np
import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / max(temperature, 1e-4), dim=-1)


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


def save_calibration(path: Path, temperature: float, threshold: float, max_length: int = 384) -> None:
    path.write_text(
        json.dumps(
            {
                "temperature": temperature,
                "hallucination_threshold": threshold,
                "max_length": max_length,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
