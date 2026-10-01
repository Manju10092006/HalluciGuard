"""Train, calibrate, and evaluate the Phase 1 logit-feature adaptation.

The input must contain human-labeled claims from the exact generator event.
The final test split is read only by ``evaluate``.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

from .phase1 import FEATURE_NAMES, FEATURE_SCHEMA


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing prepared split: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"empty prepared split: {path}")
    for record in records:
        features = record.get("features")
        if (record.get("feature_schema") != FEATURE_SCHEMA or
                not isinstance(features, list) or len(features) != len(FEATURE_NAMES) or
                not all(isinstance(x, (float, int)) and math.isfinite(x) for x in features) or
                record.get("label_id") not in (0, 1)):
            raise ValueError(f"invalid prepared row in {path}")
    return records


def preflight(data_dir: Path, output_dir: Path, *, include_test: bool = False) -> dict[str, Any]:
    import torch

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("dataset feature schema mismatch")
    label_unit = manifest.get("label_unit")
    if label_unit not in ("sentence", "atomic_claim"):
        raise ValueError("dataset label unit missing or invalid")
    names = ("train", "validation", "calibration", "test") if include_test else ("train", "validation", "calibration")
    splits = {name: _rows(data_dir / f"{name}.jsonl") for name in names}
    # Check group identifiers in the held-out file without opening its labels.
    if not include_test:
        test_groups = set()
        test_path = data_dir / "test.jsonl"
        if not test_path.is_file():
            raise FileNotFoundError(f"missing prepared split: {test_path}")
        for line in test_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                test_groups.add(json.loads(line)["group_id"])
        if not test_groups:
            raise ValueError("empty test split")
        if test_groups & set().union(*({r["group_id"] for r in values} for values in splits.values())):
            raise ValueError("group leakage into final test split")
    groups = {name: {r["group_id"] for r in rows} for name, rows in splits.items()}
    for name, members in groups.items():
        if members & set().union(*(v for k, v in groups.items() if k != name)):
            raise ValueError("group leakage across splits")
    generator = manifest.get("generator") or {}
    fields = ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision")
    if not all(generator.get(k) for k in fields):
        raise ValueError("generator identity is incomplete")
    for rows in splits.values():
        if any(any(r.get(k) != generator[k] for k in fields) for r in rows):
            raise ValueError("mixed model identities in prepared rows")
        if any(r.get("label_unit") != label_unit for r in rows):
            raise ValueError("mixed label units in prepared rows")
    if shutil.disk_usage(output_dir.parent if output_dir.parent.exists() else data_dir).free < 10_000_000:
        raise OSError("less than 10 MB free for Phase 1 checkpoint")
    return {
        "ready": True, "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "generator": generator, "label_unit": label_unit,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "note": "feature head only; no large causal LM is loaded for training",
    }


def _matrix(rows, mean=None, std=None):
    import numpy as np

    values = np.asarray([r["features"] for r in rows], dtype=np.float32)
    if mean is None:
        mean = values.mean(axis=0)
    else:
        mean = np.asarray(mean, dtype=np.float32)
    if std is None:
        std = values.std(axis=0).clip(min=1e-6)
    else:
        std = np.asarray(std, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("non-finite training features")
    return (values - mean) / std, np.asarray([r["label_id"] for r in rows], dtype=np.float32), mean, std


def _temperature(logits, labels) -> float:
    import torch

    scores = torch.tensor(logits, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.float32)
    log_t = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores / log_t.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().detach().clamp(0.05, 10.0))


def _probabilities(logits, temperature):
    import numpy as np

    scaled = np.clip(np.asarray(logits) / temperature, -80, 80)
    return 1.0 / (1.0 + np.exp(-scaled))


def _response_scores(rows, probabilities):
    """An event is positive if any claim is positive; risk is its maximum claim risk."""
    import numpy as np

    events: dict[str, tuple[float, int]] = {}
    for row, probability in zip(rows, probabilities):
        event_id = row["event_id"]
        old_risk, old_label = events.get(event_id, (0.0, 0))
        events[event_id] = (max(old_risk, float(probability)), max(old_label, int(row["label_id"])))
    return np.asarray([v[0] for v in events.values()]), np.asarray([v[1] for v in events.values()])


def _response_groups(rows):
    """Return independent split-group identities in _response_scores order."""
    events: dict[str, str] = {}
    for row in rows:
        event_id, group_id = row["event_id"], row["group_id"]
        if event_id in events and events[event_id] != group_id:
            raise ValueError("event_id spans multiple split groups")
        events[event_id] = group_id
    return list(events.values())


def _select_threshold(probabilities, labels, max_false_accept: float, min_bypass: int = 100,
                      alpha: float = 0.05, groups=None):
    """Simultaneous bound on false-accept events across independent groups.

    A group is a failure if any bypassed response in it is labelled positive.
    This is conservative when a group has multiple correlated responses.
    """
    import numpy as np

    candidates = np.unique(probabilities)
    if groups is None:
        groups = list(range(len(labels)))
    if len(groups) != len(labels) or len(probabilities) != len(labels):
        raise ValueError("response scores, labels and groups must align")
    # Union-bound the data-driven threshold search; a single-candidate bound
    # would be anti-conservative after selecting the most permissive cutoff.
    search_alpha = alpha / max(1, len(candidates))
    best = None
    for threshold in candidates:
        mask = probabilities <= threshold
        count = int(mask.sum())
        if count < min_bypass:
            continue
        group_failures: dict[Any, bool] = {}
        for group, bypassed, label in zip(groups, mask, labels):
            if bypassed:
                group_failures[group] = group_failures.get(group, False) or bool(label)
        independent_count = len(group_failures)
        if independent_count < min_bypass:
            continue
        observed = sum(group_failures.values()) / independent_count
        upper = min(1.0, observed + math.sqrt(math.log(1 / search_alpha) / (2 * independent_count)))
        if upper <= max_false_accept:
            best = {"threshold": float(threshold), "eligible": count,
                    "observed_false_accept": observed, "upper_bound": upper,
                    "max_false_accept": max_false_accept, "alpha": alpha,
                    "independent_groups": independent_count,
                    "candidate_count": len(candidates), "bound": "group_level_simultaneous_hoeffding_union"}
    return best


def train(data_dir: Path, output_dir: Path, *, epochs: int = 100, learning_rate: float = 0.001,
          seed: int = 42, max_false_accept: float = 0.01, min_bypass: int = 100) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Phase 1 checkpoint output directory is not empty")
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from sklearn.metrics import average_precision_score

    if not 1 <= epochs <= 10000 or not 0 < learning_rate <= 1 or not 0 < max_false_accept < 1:
        raise ValueError("invalid training configuration")
    audit = preflight(data_dir, output_dir)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_rows = _rows(data_dir / "train.jsonl")
    val_rows = _rows(data_dir / "validation.jsonl")
    cal_rows = _rows(data_dir / "calibration.jsonl")
    x_train, y_train, mean, std = _matrix(train_rows)
    x_val, y_val, _, _ = _matrix(val_rows, mean, std)
    x_cal, y_cal, _, _ = _matrix(cal_rows, mean, std)
    if len(set(y_train)) < 2 or len(set(y_val)) < 2 or len(set(y_cal)) < 2:
        raise ValueError("train, validation and calibration each require both labels")
    head = torch.nn.Linear(len(FEATURE_NAMES), 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    x_tensor = torch.tensor(x_train)
    y_tensor = torch.tensor(y_train)
    x_validation = torch.tensor(x_val)
    best = -1.0
    best_state = None
    best_epoch = 0
    for epoch in range(epochs):
        head.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x_tensor).flatten(), y_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            val_logits = head(x_validation).flatten().numpy()
        score = float(average_precision_score(y_val, _probabilities(val_logits, 1.0)))
        if score > best:
            best, best_epoch = score, epoch + 1
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        calibration_logits = head(torch.tensor(x_cal)).flatten().numpy()
        validation_logits = head(x_validation).flatten().numpy()
    temperature = _temperature(calibration_logits, y_cal)
    val_response_p, val_response_y = _response_scores(val_rows, _probabilities(validation_logits, temperature))
    threshold = (
        _select_threshold(val_response_p, val_response_y, max_false_accept, min_bypass,
                          groups=_response_groups(val_rows))
        if audit["label_unit"] == "sentence" else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    head_path = output_dir / "head.safetensors"
    save_file({k: v.contiguous() for k, v in best_state.items()}, str(head_path))
    head_sha = hashlib.sha256(head_path.read_bytes()).hexdigest()
    metadata = {
        "schema": "hg-phase1-head-v1", "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES), "feature_mean": mean.tolist(), "feature_std": std.tolist(),
        "head_sha256": head_sha, "dataset_manifest_sha256": audit["dataset_manifest_sha256"],
        "label_unit": audit["label_unit"],
        **audit["generator"], "seed": seed, "best_epoch": best_epoch,
        "validation_pr_auc": best,
    }
    calibration = {
        "schema": "hg-phase1-calibration-v1", "head_sha256": head_sha,
        "temperature": temperature, "target": "human_labeled_hallucination_related_claim",
        "release_validated": False,
        "release_block_reason": "research_head_requires_independent_production_qualification",
        "threshold_selection_note": None if threshold is not None else (
            "atomic_claim_head_not_aligned_to_sentence_inference" if audit["label_unit"] != "sentence"
            else "validation_risk_constraint_not_met"
        ),
        "threshold": threshold["threshold"] if threshold else None,
        "validation_policy": threshold,
        "calibration_samples": len(y_cal), "validation_responses": len(val_response_y),
    }
    calibration["calibration_id"] = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    # A loadability check is part of successful training.
    from .phase1 import Phase1Service, Phase1Config
    service = Phase1Service(Phase1Config(mode="shadow", head_dir=output_dir))
    service._load()
    report = {"checkpoint": str(output_dir), "preflight": audit,
              "validation_pr_auc": best, "best_epoch": best_epoch,
              "calibration_temperature": temperature, "release_validated": False,
              "validation_policy": threshold}
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate(data_dir: Path, head_dir: Path) -> dict[str, Any]:
    if (head_dir / "test_report.json").exists():
        raise FileExistsError("Phase 1 test report already exists")
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

    preflight(data_dir, head_dir, include_test=True)
    meta = json.loads((head_dir / "metadata.json").read_text(encoding="utf-8"))
    cal = json.loads((head_dir / "calibration.json").read_text(encoding="utf-8"))
    if hashlib.sha256((head_dir / "head.safetensors").read_bytes()).hexdigest() != meta["head_sha256"]:
        raise ValueError("head checksum mismatch")
    if hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest() != meta["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest differs from training")
    rows = _rows(data_dir / "test.jsonl")
    values, labels, _, _ = _matrix(rows, np.asarray(meta["feature_mean"]), np.asarray(meta["feature_std"]))
    head = torch.nn.Linear(len(FEATURE_NAMES), 1)
    head.load_state_dict(load_file(str(head_dir / "head.safetensors")))
    head.eval()
    with torch.inference_mode():
        logits = head(torch.tensor(values)).flatten().numpy()
    p = _probabilities(logits, cal["temperature"])
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(float((bins == k).mean() * abs(p[bins == k].mean() - labels[bins == k].mean())) for k in range(10) if (bins == k).any())
    binary_labels = len(set(labels)) == 2
    output: dict[str, Any] = {
        "samples": len(rows), "positives": int(labels.sum()),
        "pr_auc": float(average_precision_score(labels, p)) if binary_labels else None,
        "roc_auc": float(roc_auc_score(labels, p)) if len(set(labels)) == 2 else None,
        "brier": float(np.mean((p - labels) ** 2)), "ece_10_equal_width": ece,
        "head_sha256": meta["head_sha256"], "dataset_manifest_sha256": meta["dataset_manifest_sha256"],
    }
    threshold = cal.get("threshold") if cal.get("release_validated") else None
    response_p, response_y = _response_scores(rows, p)
    output["responses"] = len(response_y)
    output["response_pr_auc"] = float(average_precision_score(response_y, response_p)) if len(set(response_y)) == 2 else None
    output["response_roc_auc"] = float(roc_auc_score(response_y, response_p)) if len(set(response_y)) == 2 else None
    if threshold is not None:
        accepted = response_p <= threshold
        output["routing"] = {"verification_coverage": float((~accepted).mean()),
                              "bypass_count": int(accepted.sum()),
                              "false_accept_count": int(response_y[accepted].sum()),
                              "false_accept_rate": float(response_y[accepted].mean()) if accepted.any() else None}
        predicted = p > threshold
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
        output.update({"precision": float(precision), "recall": float(recall), "f1": float(f1)})
    (head_dir / "test_report.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output
