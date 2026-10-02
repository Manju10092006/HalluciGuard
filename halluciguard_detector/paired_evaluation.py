"""Read-only paired evidence-representation evaluation, not training.

Both arms use one unchanged checkpoint, calibration and genuine prepared rows.
The candidate applies runtime normalization to the already prepared evidence.
It does NOT reproduce external retrieval or prove pre-verification safety.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .calibration import class_probabilities, load_calibration
from .structured_evidence import normalize_evidence
from .text import lexical_evidence, lexical_documents
from .training import ID_TO_LABEL, evaluate_saved_predictions, load_rows, predict_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_splits(splits: dict, *, allow_exact_overlap=False) -> dict:
    groups, pairs = {}, {}
    counts = {}
    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"empty split: {split}")
        ids = set()
        groups[split], pairs[split] = set(), set()
        for row in rows:
            if not row.get("id") or row["id"] in ids:
                raise ValueError(f"missing or duplicate id in {split}")
            ids.add(row["id"])
            if row.get("source_id") is None or not str(row["source_id"]).strip():
                raise ValueError(f"missing source group in {split}")
            if type(row.get("label_id")) is not int or row["label_id"] not in ID_TO_LABEL:
                raise ValueError(f"invalid gold id in {split}")
            if row.get("label") != ID_TO_LABEL[row["label_id"]]:
                raise ValueError(f"gold label mismatch in {split}")
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("evidence", "claim")):
                raise ValueError(f"empty or invalid input in {split}")
            groups[split].add(str(row["source_id"]))
            pairs[split].add((row["evidence"], row["claim"]))
        counts[split] = {"samples": len(rows), "groups": len(groups[split]),
                         "classes": dict(Counter(r["label"] for r in rows)),
                         "duplicate_input_pairs": len(rows) - len(pairs[split])}
    overlap = {}
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if groups[left] & groups[right]:
                raise ValueError(f"source overlap: {left}/{right}")
            overlap[f"{left}/{right}"] = len(pairs[left] & pairs[right])
    if any(overlap.values()) and not allow_exact_overlap:
        raise ValueError("exact evidence/claim pairs cross split boundaries")
    return {"splits": counts, "source_disjoint": True, "cross_split_exact_pairs": overlap,
            "near_duplicate_check": "not performed"}


def candidate_rows(rows, *, selection="lexical_fallback", selector=None):
    result, stats = [], Counter()
    for row in rows:
        candidates, meta = normalize_evidence(row["evidence"])
        if not candidates:
            raise ValueError("normalization erased a genuine evaluation input")
        if selection == "production":
            if len(candidates) == 1:
                # A successful ranker cannot change the identity of one input.
                # Do not claim a retrieval model ran on these shortcut rows.
                selected = candidates[0]
                stats["single_candidate_identity_shortcut"] += 1
            else:
                if selector is None:
                    from .evidence import select_evidence
                    selector = select_evidence
                trace = {}
                snippets = selector(row["claim"], candidates, trace=trace)
                if not snippets:
                    raise ValueError("production selection returned no evaluation evidence")
                selected = snippets[0]
                stats["actual_multi_candidate_selector_calls"] += 1
                stats["degraded_multi_candidate_selections"] += bool(trace.get("degraded", True))
        elif selection == "lexical_fallback":
            selected = lexical_evidence(row["claim"], candidates, limit=1)[0]
        elif selection == "context_preserving_fallback":
            selected = lexical_documents(row["claim"], candidates, limit=1)[0]
        else:
            raise ValueError("unknown selection arm")
        result.append({**row, "evidence": selected})
        stats["changed_inputs"] += selected != row["evidence"]
        stats["multi_candidate_selection"] += len(candidates) > 1
        for key in ("field_truncated", "character_truncated", "document_truncated", "malformed_structured"):
            stats[key] += bool(meta[key])
    return result, dict(stats)


def wilson(errors, count):
    if not count:
        return None
    z = 1.959963984540054
    fraction = errors / count
    denominator = 1 + z * z / count
    middle = (fraction + z * z / (2 * count)) / denominator
    half = z * math.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count * count)) / denominator
    return [max(0., middle - half), min(1., middle + half)]


def false_accept_diagnostic(logits, labels, temperature, threshold):
    probabilities = class_probabilities(logits, temperature)
    labels = np.asarray(labels)
    eligible = probabilities[:, 1:].sum(axis=1) < threshold
    count = int(eligible.sum())
    errors = int((eligible & (labels != 0)).sum())
    return {"threshold": threshold, "eligible": count, "coverage": count / len(labels),
            "non_supported_eligible": errors,
            "contradicted_eligible": int((eligible & (labels == 1)).sum()),
            "conditional_false_accept_rate": errors / count if count else None,
            "wilson_95_interval": wilson(errors, count),
            "interpretation": "hypothetical grounded-score bypass only; NOT a Phase 1 certificate or enabled production route"}


def run(data: Path, checkpoint: Path, output: Path, batch_size=32, raw_responses: Path | None = None,
        exclude_cross_split_duplicates=False):
    # Reserve a fresh artifact directory before expensive inference. Never write
    # inside checkpoint/data; incomplete runs remain explicit, not overwritten.
    if output.exists():
        raise FileExistsError("paired report directory already exists")
    for protected in (data.resolve(), checkpoint.resolve()):
        if output.resolve() == protected or protected in output.resolve().parents:
            raise ValueError("reports must not be written inside data or checkpoint")
    splits = {s: load_rows(data / f"{s}.jsonl") for s in ("train", "dev", "test")}
    audit = validate_splits(splits, allow_exact_overlap=exclude_cross_split_duplicates)
    rows = splits["test"]
    known = {(r["evidence"], r["claim"]) for s in ("train", "dev") for r in splits[s]}
    excluded = [r for r in rows if (r["evidence"], r["claim"]) in known]
    if excluded:
        rows = [r for r in rows if (r["evidence"], r["claim"]) not in known]
    if not rows:
        raise ValueError("no test rows remain after exact-overlap exclusion")
    audit["excluded_test_ids"] = [str(r["id"]) for r in excluded]
    audit["excluded_test_classes"] = dict(Counter(r["label"] for r in excluded))
    audit["evaluated_test_count"] = len(rows)
    audit["benchmark_definition"] = "prepared test minus exact train/dev input pairs; same frozen IDs in both arms"
    if raw_responses:
        raw = {str(r["id"]): r for r in load_rows(raw_responses)}
        for row in rows:
            response = raw.get(str(row["id"]).split(":", 1)[0])
            if response is None or response.get("split") != "test" or str(response.get("source_id")) != str(row["source_id"]):
                raise ValueError("prepared test row does not match official response/source split")
        audit["official_test_source_checked"] = True
    else:
        audit["official_test_source_checked"] = False
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config["id2label"] != {str(k): v for k, v in ID_TO_LABEL.items()}:
        raise ValueError("checkpoint class map differs")
    calibration = load_calibration(checkpoint / "calibration.json")
    required = ("temperature", "contradiction_threshold", "verification_risk_threshold", "max_length")
    if any(calibration.get(k) is None for k in required):
        raise ValueError("complete frozen calibration required")
    immutable = {str(p.resolve()): sha256(p) for p in checkpoint.iterdir() if p.is_file()}
    immutable.update({str((data / f"{s}.jsonl").resolve()): sha256(data / f"{s}.jsonl") for s in splits})
    if raw_responses:
        immutable[str(raw_responses.resolve())] = sha256(raw_responses)
    lexical, lexical_stats = candidate_rows(rows)
    contextual, contextual_stats = candidate_rows(rows, selection="context_preserving_fallback")
    # Keep the full production evidence identity when there is just one input;
    # use the actual canonical selector (and its models) for ambiguous pools.
    production, production_stats = candidate_rows(rows, selection="production")
    output.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True,
                                                             trust_remote_code=False).to(device)
    report = {"checkpoint": str(checkpoint.resolve()), "audit": audit, "immutable_sha256": immutable,
              "calibration": calibration, "normalization": {
                  "production": production_stats, "lexical_fallback": lexical_stats,
                  "context_preserving_fallback": contextual_stats}, "device": str(device),
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "batch_size": batch_size, "arms": {}, "no_training": True,
              "limitations": ["prepared claim labels, not independent human reannotation",
                  "runtime normalization/selection of prepared evidence only; no external retrieval replay",
                  "single-candidate identity shortcuts do not certify backend execution",
                  "gold claims refer to full original source; selected-snippet adequacy is not independently reannotated",
                  "no test-set threshold fitting; saved calibration provenance requires separate audit",
                  "same source document may contain multiple correlated claims; intervals are descriptive",
                  "no reliable before/after accuracy claim about telemetry or certification fixes"]}
    logits_by_arm = {}
    for name, arm in (("historical_prepared", rows), ("production_representation", production),
                      ("lexical_fallback_representation", lexical),
                      ("context_preserving_fallback_representation", contextual)):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        logits, labels = predict_rows(model, tokenizer, arm, device=device,
            max_length=int(calibration["max_length"]), batch_size=batch_size)
        duration = time.perf_counter() - started
        logits_by_arm[name] = logits
        metrics = evaluate_saved_predictions(logits, labels, float(calibration["temperature"]),
            float(calibration["contradiction_threshold"]), float(calibration["verification_risk_threshold"]))
        metrics["false_accept_diagnostic"] = false_accept_diagnostic(logits, labels,
            float(calibration["temperature"]), float(calibration["verification_risk_threshold"]))
        metrics["seconds"] = duration
        metrics["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        metrics["task_type_breakdown"] = {}
        tasks = np.asarray([r.get("task_type", "unknown") for r in rows])
        for task in sorted(set(tasks)):
            mask = tasks == task
            metrics["task_type_breakdown"][str(task)] = evaluate_saved_predictions(logits[mask], labels[mask],
                float(calibration["temperature"]), float(calibration["contradiction_threshold"]),
                float(calibration["verification_risk_threshold"]))
        np.savez_compressed(output / f"{name}_predictions.npz", logits=logits, labels=labels,
                            ids=np.asarray([str(r["id"]) for r in rows]))
        report["arms"][name] = metrics
        print(json.dumps({"arm": name, "samples": len(labels), "seconds": duration,
                          "accuracy": metrics["three_class"]["accuracy"],
                          "contradiction": metrics["task_a_contradiction"]}), flush=True)
    report["paired_differences"] = {name: {
        "prediction_changes": int((logits_by_arm["historical_prepared"].argmax(1) != logits.argmax(1)).sum()),
        "maximum_logit_difference": float(np.max(np.abs(logits_by_arm["historical_prepared"] - logits)))}
        for name, logits in logits_by_arm.items() if name != "historical_prepared"}
    if any(sha256(Path(p)) != h for p, h in immutable.items()):
        raise RuntimeError("protected input changed during evaluation")
    report["protected_artifacts_unchanged"] = True
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
