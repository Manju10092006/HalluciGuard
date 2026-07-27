from collections import defaultdict
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ClassificationMetrics(BaseModel):
    """Structured container holding classification performance metrics."""

    accuracy: float = Field(..., ge=0.0, le=1.0, description="Overall classification accuracy.")
    precision_macro: float = Field(..., ge=0.0, le=1.0, description="Macro-averaged precision across classes.")
    recall_macro: float = Field(..., ge=0.0, le=1.0, description="Macro-averaged recall across classes.")
    f1_macro: float = Field(..., ge=0.0, le=1.0, description="Macro-averaged F1 score across classes.")
    precision_weighted: float = Field(..., ge=0.0, le=1.0, description="Weighted-averaged precision across classes.")
    recall_weighted: float = Field(..., ge=0.0, le=1.0, description="Weighted-averaged recall across classes.")
    f1_weighted: float = Field(..., ge=0.0, le=1.0, description="Weighted-averaged F1 score across classes.")
    confusion_matrix: Dict[str, Dict[str, int]] = Field(
        ..., description="Nested dictionary representing confusion matrix: matrix[true_label][pred_label]"
    )
    class_metrics: Dict[str, Dict[str, float]] = Field(
        ..., description="Per-class precision, recall, and F1 score breakdown."
    )


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """Compute standard classification accuracy ratio."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return round(correct / len(y_true), 4)


def compute_class_breakdown(
    y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, and F1 score."""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    class_metrics = {}
    for lbl in labels:
        true_pos = tp[lbl]
        false_pos = fp[lbl]
        false_neg = fn[lbl]

        prec = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        rec = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        class_metrics[lbl] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true if t == lbl)
        }

    return class_metrics


def compute_confusion_matrix(
    y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None
) -> Dict[str, Dict[str, int]]:
    """Compute 2D confusion matrix: matrix[true_label][pred_label]."""
    if labels is None:
        labels = ["LOW", "MEDIUM", "HIGH"]

    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1

    return matrix


def calculate_metrics(
    y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None
) -> ClassificationMetrics:
    """Compute full ClassificationMetrics model given true and predicted labels."""
    if labels is None:
        labels = ["LOW", "MEDIUM", "HIGH"]

    accuracy = compute_accuracy(y_true, y_pred)
    class_metrics = compute_class_breakdown(y_true, y_pred, labels)
    confusion = compute_confusion_matrix(y_true, y_pred, labels)

    # Compute Macro averages
    macro_prec = sum(m["precision"] for m in class_metrics.values()) / len(labels) if labels else 0.0
    macro_rec = sum(m["recall"] for m in class_metrics.values()) / len(labels) if labels else 0.0
    macro_f1 = sum(m["f1"] for m in class_metrics.values()) / len(labels) if labels else 0.0

    # Compute Weighted averages
    total_samples = len(y_true)
    weighted_prec = (
        sum(m["precision"] * m["support"] for m in class_metrics.values()) / total_samples
        if total_samples > 0 else 0.0
    )
    weighted_rec = (
        sum(m["recall"] * m["support"] for m in class_metrics.values()) / total_samples
        if total_samples > 0 else 0.0
    )
    weighted_f1 = (
        sum(m["f1"] * m["support"] for m in class_metrics.values()) / total_samples
        if total_samples > 0 else 0.0
    )

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_macro=round(macro_prec, 4),
        recall_macro=round(macro_rec, 4),
        f1_macro=round(macro_f1, 4),
        precision_weighted=round(weighted_prec, 4),
        recall_weighted=round(weighted_rec, 4),
        f1_weighted=round(weighted_f1, 4),
        confusion_matrix=confusion,
        class_metrics=class_metrics
    )
