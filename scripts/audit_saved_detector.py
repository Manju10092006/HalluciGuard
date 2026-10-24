"""Audit existing Detector results without importing ML libraries or running inference."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import zipfile


LABELS = ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]


def read_npy(data: bytes) -> tuple[tuple[int, ...], list[int | float]]:
    """Read numeric, C-order NPY data; never load pickle/object arrays."""
    if data[:6] != b"\x93NUMPY" or len(data) < 10:
        raise ValueError("invalid_npy_header")
    major, minor = data[6:8]
    if (major, minor) not in {(1, 0), (2, 0), (3, 0)}:
        raise ValueError("unsupported_npy_version")
    start = 10 if major == 1 else 12
    if len(data) < start:
        raise ValueError("truncated_npy_header")
    length = int.from_bytes(data[8:start], "little")
    header = ast.literal_eval(data[start:start + length].decode("utf-8" if major == 3 else "latin1"))
    if not isinstance(header, dict) or header.get("fortran_order") is not False:
        raise ValueError("unsupported_npy_layout")
    dtype = header.get("descr", "")
    match = re.fullmatch(r"([<>|=])([fi])([48])", dtype) if isinstance(dtype, str) else None
    shape = header.get("shape")
    if not match or not isinstance(shape, tuple) or not shape or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("unsupported_npy_dtype_or_shape")
    order, kind, width = match.groups()
    if order == "|":
        raise ValueError("unsupported_npy_byte_order")
    endian = ("<" if sys.byteorder == "little" else ">") if order == "=" else order
    code = {("f", "4"): "f", ("f", "8"): "d", ("i", "4"): "i", ("i", "8"): "q"}[(kind, width)]
    body = data[start + length:]
    if len(body) != math.prod(shape) * int(width):
        raise ValueError("invalid_npy_body_length")
    values = [item[0] for item in struct.iter_unpack(endian + code, body)]
    if kind == "f" and not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite_saved_logits")
    return shape, values


def read_predictions(path: Path) -> tuple[list[list[float]], list[int]]:
    with zipfile.ZipFile(path) as archive:
        for name in ("logits.npy", "labels.npy"):
            if archive.getinfo(name).file_size > 20_000_000:
                raise ValueError("saved_array_exceeds_audit_budget")
        shape, logits = read_npy(archive.read("logits.npy"))
        label_shape, labels = read_npy(archive.read("labels.npy"))
    if len(shape) != 2 or shape[1] != len(LABELS) or label_shape != (shape[0],):
        raise ValueError("incompatible_saved_prediction_shapes")
    if any(type(label) is not int or label not in range(len(LABELS)) for label in labels):
        raise ValueError("invalid_saved_labels")
    return [logits[i:i + 3] for i in range(0, len(logits), 3)], labels


def probabilities(logits: list[float], temperature: float) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid_saved_temperature")
    weights = [math.exp((value - max(logits)) / temperature) for value in logits]
    return [value / sum(weights) for value in weights]


def text_summary(text: str) -> dict:
    return {"characters": len(text), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def compare_inputs(original: dict, sentences: list[dict]) -> dict:
    factual = [sentence for sentence in sentences if not sentence.get("non_factual", False)]
    return {
        "original_claim": original["claim"],
        "original_evidence": text_summary(original["evidence"]),
        "factual_claim_count": len(factual),
        "non_factual_claim_count": len(sentences) - len(factual),
        "single_claim_text_identical": len(factual) == 1 and factual[0]["text"] == original["claim"],
        "single_claim_equal_ignoring_terminal_period": len(factual) == 1 and factual[0]["text"].rstrip(".").strip() == original["claim"].rstrip(".").strip(),
        "all_factual_evidence_text_identical": bool(factual) and all(s.get("model_input_evidence") == original["evidence"] for s in factual),
        "runtime_claims": [{
            "text": sentence["text"],
            "non_factual": bool(sentence.get("non_factual", False)),
            "label": sentence.get("label"),
            "requires_verification": sentence.get("requires_verification"),
            "evidence_identical": sentence.get("model_input_evidence") == original["evidence"],
            "selected_evidence": text_summary(sentence.get("model_input_evidence") or ""),
            "selection_route": sentence.get("evidence_route"),
            "tokenization_observed": sentence.get("evidence_selection_trace", {}).get("model_consumption_observed", False),
        } for sentence in sentences],
    }


def confusion(cases: list[dict], prediction_field: str) -> list[list[int]]:
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for case in cases:
        matrix[case["label_id"]][case[prediction_field]] += 1
    return matrix


def macro_f1(matrix: list[list[int]]) -> float:
    scores = []
    for label in range(len(LABELS)):
        denominator = sum(matrix[label]) + sum(row[label] for row in matrix)
        scores.append(2 * matrix[label][label] / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def audit(runtime_path: Path, predictions_path: Path, test_path: Path, calibration_path: Path) -> dict:
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    logits, labels = read_predictions(predictions_path)
    saved_cases = runtime["cases"]
    indices = [case["index"] for case in saved_cases]
    if len(set(indices)) != len(indices) or any(type(index) is not int or not 0 <= index < len(logits) for index in indices):
        raise ValueError("duplicate_or_invalid_runtime_index")
    wanted = set(indices)
    original_rows = {}
    with test_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in wanted:
                original_rows[index] = json.loads(line)
            if len(original_rows) == len(wanted):
                break
    if len(original_rows) != len(wanted):
        raise ValueError("missing_original_test_row")
    cases = []
    for saved in saved_cases:
        index = saved["index"]
        row = original_rows[index]
        if row["id"] != saved["id"] or row["label_id"] != saved["label_id"] or labels[index] != saved["label_id"]:
            raise ValueError("runtime_and_saved_pair_identity_mismatch")
        raw_prediction = max(range(len(LABELS)), key=lambda label: logits[index][label])
        response = saved["response"]
        reproduced = 1 if response["contradicted_count"] else 2 if response["unknown_count"] or not response["claim_count"] else 0
        if reproduced != saved["prediction"]:
            raise ValueError("runtime_evaluator_prediction_mismatch")
        cases.append({
            "index": index, "id": row["id"], "label_id": saved["label_id"],
            "raw_pairwise_prediction": raw_prediction, "runtime_prediction": saved["prediction"],
            "raw_pairwise_probabilities": dict(zip(LABELS, probabilities(logits[index], float(calibration["temperature"])))),
            "input_comparison": compare_inputs(row, response["sentences"]),
        })
    contradictions = [case for case in cases if case["label_id"] == 1]
    raw_matrix = confusion(cases, "raw_pairwise_prediction")
    runtime_matrix = confusion(cases, "runtime_prediction")
    return {
        "scope": "Existing saved predictions and runtime responses only; no model imports, inference, threshold fitting or quality-improvement claim.",
        "inputs": {name: str(path.resolve()) for name, path in {
            "runtime": runtime_path, "predictions": predictions_path, "test_rows": test_path, "calibration": calibration_path,
        }.items()},
        "calibration_temperature_used_to_display_probabilities": calibration["temperature"],
        "labels_in_matrix_order": LABELS,
        "rows": len(cases),
        "raw_pairwise_confusion_matrix": raw_matrix,
        "runtime_confusion_matrix": runtime_matrix,
        "raw_pairwise_macro_f1_on_same_rows": macro_f1(raw_matrix),
        "runtime_macro_f1_recomputed_from_saved_predictions": macro_f1(runtime_matrix),
        "saved_runtime_macro_f1": runtime["macro_f1"],
        "contradiction_cases": {
            "total": len(contradictions),
            "raw_pairwise_misses": sum(case["raw_pairwise_prediction"] != 1 for case in contradictions),
            "runtime_misses": sum(case["runtime_prediction"] != 1 for case in contradictions),
            "all_selected_evidence_identical_to_original": bool(contradictions) and all(case["input_comparison"]["all_factual_evidence_text_identical"] for case in contradictions),
        },
        "caveats": [
            "Raw historical pairwise predictions and production runtime decisions have different units when decomposition changes a sentence or filters nonfactual content.",
            "The source evaluator sorts its stratified 32-per-class sample by dataset index, then takes the first 8 per class; this 24-row subset is biased toward early dataset rows.",
            "A historical SUPPORTED label can belong to nonfactual content (including Sure!); the runtime evaluator maps zero factual claims to NOT_ENOUGH_INFO, so that mismatch is not a factual false negative.",
            "Evidence text equality does not establish identical token IDs after claim rewriting; token consumption was not recorded for every case.",
            "The small balanced sample does not estimate population accuracy; the full historical set is heavily imbalanced and has separate quality limitations.",
            "Displayed probabilities use the supplied saved calibration temperature; predictions use argmax and do not fit or change any threshold.",
        ],
        "cases": cases,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=root / "runtime-logs/fresh-model-evaluation/fresh-runtime.json")
    parser.add_argument("--predictions", type=Path, default=root / "artifacts/detector-best/test_predictions.npz")
    parser.add_argument("--calibration", type=Path, default=root / "artifacts/detector-best/calibration.json")
    parser.add_argument("--output", type=Path, help="New JSON file; existing files are never overwritten.")
    args = parser.parse_args()
    report = audit(args.runtime, args.predictions, args.data_dir / "test.jsonl", args.calibration)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
