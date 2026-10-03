"""Offline locked-checkpoint evaluation; no retrieval, tuning or live n8n calls.

python -m halluciguard_detector.independent_evaluation --help
Raw model classes and hypothetical routing are always separate. No result
enables fast-path acceptance. Synthetic bundles cannot establish a real gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from .evaluation_data import (GateBlocked, LABELS, ROLE_FILES, audit_splits, canonical, digest,
                              file_hash, grouped_split, lock_splits, read_rows, verify_bundle,
                              verify_checkpoint_exposure)


def probabilities(logits, temperature):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 3 or not len(logits) or not np.isfinite(logits).all():
        raise ValueError("finite nonempty N x 3 logits required")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive finite frozen temperature required")
    values = logits / temperature
    values -= values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def metric_report(logits, labels, *, temperature, contradiction_threshold, risk_threshold):
    p = probabilities(logits, temperature)
    labels = np.asarray(labels)
    if labels.shape != (len(p),) or not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("aligned exact three-class integer labels required")
    if any(not math.isfinite(t) or not 0 <= t <= 1 for t in (contradiction_threshold, risk_threshold)):
        raise ValueError("frozen thresholds must lie in [0,1]")
    pred = p.argmax(axis=1)
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (labels, pred), 1)
    per_class = {}
    for i, name in enumerate(LABELS):
        tp, actual, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        per_class[name] = {"support": actual, "predicted": predicted, "precision": _ratio(tp, predicted),
                           "recall": _ratio(tp, actual), "f1": _ratio(2 * tp, actual + predicted)}
    confidence, correct = p.max(axis=1), (pred == labels)
    bins, ece = [], 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        n = int(mask.sum())
        if n:
            ece += n / len(p) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
        bins.append({"lower": float(low), "upper": float(high), "count": n,
                     "mean_confidence": float(confidence[mask].mean()) if n else None,
                     "accuracy": float(correct[mask].mean()) if n else None})
    onehot = np.eye(3)[labels]
    risk = p[:, 1:].sum(axis=1)
    eligible = risk < risk_threshold
    contradiction = p[:, 1] >= contradiction_threshold
    true_c = labels == 1
    tp = int((true_c & contradiction).sum())
    fp = int((~true_c & contradiction).sum())
    fn = int((true_c & ~contradiction).sum())
    tn = int((~true_c & ~contradiction).sum())
    errors = int((eligible & (labels != 0)).sum())
    supported = labels == 0
    coverage = float(eligible.mean())
    return {"samples": len(p), "model_quality": {
        "accuracy": float(correct.mean()), "confusion_matrix": {"labels": list(LABELS), "rows_are_true": True, "matrix": matrix.tolist()},
        "per_class": per_class, "macro_f1": sum(v['f1'] or 0 for v in per_class.values()) / 3,
        "micro_f1": float(correct.mean()), "micro_f1_definition": "single-label multiclass micro-F1 equals accuracy",
        "calibration": {"top_label_ece_10_equal_width_bins": ece, "reliability_bins": bins,
                        "multiclass_brier_sum_over_classes": float(((p - onehot) ** 2).sum(axis=1).mean()),
                        "brier_definition": "mean per-row sum of squared three-class errors (range 0..2)"}},
        "operational_policy": {
            "fast_path_enabled": False, "scope": "hypothetical grounded-score bypass only; not production Phase-1 eligibility or truth certification",
            "contradiction_threshold": contradiction_threshold, "risk_threshold": risk_threshold,
            "contradiction_binary": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": _ratio(tp, tp + fp),
                                     "recall": _ratio(tp, tp + fn), "f1": _ratio(2 * tp, 2 * tp + fp + fn)},
            "hypothetical_bypass_count": int(eligible.sum()), "coverage": coverage,
            "non_supported_bypass_count": errors, "contradicted_bypass_count": int((eligible & true_c).sum()),
            "conditional_false_accept_rate": _ratio(errors, int(eligible.sum())),
            "false_accept_rate_among_non_supported": _ratio(errors, int((labels != 0).sum())),
            "false_reject_or_route_supported_count": int((~eligible & supported).sum()),
            "false_reject_or_route_supported_rate": _ratio(int((~eligible & supported).sum()), int(supported.sum())),
            "routing_or_abstention_rate": 1 - coverage,
            "abstention_definition": "fraction routed for verification; Detector does not itself establish a final abstention verdict"}}


def _critical(report):
    quality, policy = report['model_quality'], report['operational_policy']
    result = {"accuracy": quality['accuracy'], "macro_f1": quality['macro_f1'],
              "ece": quality['calibration']['top_label_ece_10_equal_width_bins'],
              "brier": quality['calibration']['multiclass_brier_sum_over_classes']}
    for k in ('precision', 'recall', 'f1'):
        result['contradiction_threshold_' + k] = policy['contradiction_binary'][k]
    for label in LABELS:
        for k in ('precision', 'recall', 'f1'):
            result[label.lower() + '_' + k] = quality['per_class'][label][k]
    for k in ('coverage', 'conditional_false_accept_rate', 'false_accept_rate_among_non_supported', 'false_reject_or_route_supported_rate'):
        result[k] = policy[k]
    return result


def bootstrap_intervals(logits, labels, source_ids, *, temperature, contradiction_threshold, risk_threshold,
                        seed=20261003, replicates=2000, confidence=0.95):
    """Percentile source-cluster bootstrap; all claims of sampled sources move together."""
    logits, labels = np.asarray(logits), np.asarray(labels)
    if len(source_ids) != len(labels) or replicates < 2 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap inputs")
    metric_report(logits, labels, temperature=temperature, contradiction_threshold=contradiction_threshold, risk_threshold=risk_threshold)
    groups = {source: np.array([i for i, s in enumerate(source_ids) if s == source]) for source in sorted(set(source_ids))}
    if len(groups) < 2:
        raise GateBlocked("bootstrap requires at least two source clusters")
    indexes = list(groups.values())
    rng = np.random.default_rng(seed)
    values = {}
    for _ in range(replicates):
        sample = np.concatenate([indexes[i] for i in rng.integers(0, len(indexes), size=len(indexes))])
        report = metric_report(logits[sample], labels[sample], temperature=temperature,
                               contradiction_threshold=contradiction_threshold, risk_threshold=risk_threshold)
        for name, value in _critical(report).items():
            values.setdefault(name, [])
            if value is not None:
                values[name].append(value)
    alpha = (1 - confidence) / 2
    return {"method": "source-cluster percentile bootstrap; equal-probability clusters sampled with replacement; preserve all rows per source; variable replicate row counts",
            "seed": seed, "replicates": replicates, "confidence": confidence, "source_clusters": len(groups),
            "limitations": "conditional on supplied frozen checkpoint/cohort; not model-training uncertainty; few clusters or unmodelled across-source correlations limit inference",
            "intervals": {name: {"lower": float(np.quantile(v, alpha)) if v else None,
                                 "upper": float(np.quantile(v, 1 - alpha)) if v else None,
                                 "valid_replicates": len(v), "undefined_replicates": replicates - len(v)} for name, v in values.items()}}


def evaluate_locked(bundle_dir: Path, manifest_sha256: str, checkpoint: Path, output: Path, *, batch_size=32,
                    bootstrap_seed=20261003, bootstrap_replicates=2000, predictor=None) -> dict:
    bundle, splits = verify_bundle(bundle_dir, manifest_sha256)
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError("evaluation output already exists")
    if any(output.resolve() == p.resolve() or p.resolve() in output.resolve().parents for p in (checkpoint, Path(bundle_dir))):
        raise GateBlocked("evaluation output cannot modify checkpoint or locked bundle")
    required = ('model.safetensors', 'config.json', 'calibration.json', 'tokenizer.json', 'tokenizer_config.json')
    protected = {name: file_hash(checkpoint / name) for name in required}
    exposure = verify_checkpoint_exposure(bundle, splits['locked_test'], protected['model.safetensors'])
    declaration = json.loads(Path(bundle['exposure_record']['path']).read_text())
    if declaration.get('calibration_sha256') != protected['calibration.json']:
        raise GateBlocked("frozen calibration not linked to historical development exposure")
    config = json.loads((checkpoint / 'config.json').read_text())
    if config.get('id2label') != {str(i): label for i, label in enumerate(LABELS)}:
        raise GateBlocked("checkpoint class-map mismatch")
    calibration = json.loads((checkpoint / 'calibration.json').read_text())
    if any(calibration.get(k) is None for k in ('temperature', 'contradiction_threshold', 'verification_risk_threshold', 'max_length')):
        raise GateBlocked("complete frozen calibration missing")
    temperature, ct, rt = (float(calibration[k]) for k in ('temperature', 'contradiction_threshold', 'verification_risk_threshold'))
    if not math.isfinite(temperature) or temperature <= 0 or any(not math.isfinite(t) or not 0 <= t <= 1 for t in (ct, rt)) or type(calibration['max_length']) is not int or calibration['max_length'] <= 0:
        raise GateBlocked("invalid frozen calibration")
    if batch_size < 1 or bootstrap_replicates < 2:
        raise ValueError("invalid batch/bootstrap configuration")
    if predictor is not None and bundle['provenance_kind'] != 'synthetic_fixture':
        raise GateBlocked("injected predictors are allowed only for synthetic framework fixtures")
    rows = splits['locked_test']
    plan = {"dataset_manifest_sha256": manifest_sha256, "checkpoint_files": protected,
            "temperature": temperature, "contradiction_threshold": ct, "risk_threshold": rt,
            "max_length": calibration['max_length'], "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": bootstrap_replicates, "batch_size": batch_size, "fast_path_enabled": False}
    # Write the preregistered immutable plan before any checkpoint inference.
    output.mkdir(parents=True)
    with (output / 'evaluation-plan.json').open('xb') as f:
        f.write(canonical(plan) + b'\n')
    if predictor is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .training import predict_rows
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device)
        logits, labels = predict_rows(model, tokenizer, rows, device=device, max_length=calibration['max_length'], batch_size=batch_size)
        inference = {"execution": "REAL_FROZEN_CHECKPOINT", "device": str(device)}
    else:
        logits, labels = predictor(rows)
        inference = {"execution": "SYNTHETIC_FRAMEWORK_TEST_ONLY", "device": "stub"}
    if not np.array_equal(labels, [r['label_id'] for r in rows]):
        raise GateBlocked("prediction/gold row order mismatch")
    metrics = metric_report(logits, labels, temperature=temperature, contradiction_threshold=ct, risk_threshold=rt)
    ci = bootstrap_intervals(logits, labels, [r['source_id'] for r in rows], temperature=temperature,
                             contradiction_threshold=ct, risk_threshold=rt, seed=bootstrap_seed, replicates=bootstrap_replicates)
    p = probabilities(logits, temperature)
    with (output / 'predictions.jsonl').open('xb') as f:
        for row, raw, prob in zip(rows, logits, p):
            risk = float(prob[1:].sum())
            f.write(canonical({"row_id": row['id'], "source_id": row['source_id'], "gold_label": row['label'],
                               "raw_logits": np.asarray(raw).tolist(), "raw_class_prediction": LABELS[int(np.argmax(raw))],
                               "calibrated_probabilities": {label: float(prob[i]) for i, label in enumerate(LABELS)},
                               "operational_risk": risk, "route_for_verification": risk >= rt,
                               "contradiction_threshold_decision": bool(prob[1] >= ct),
                               "hypothetical_bypass_eligible": risk < rt, "fast_path_enabled": False}) + b'\n')
    verify_bundle(bundle_dir, manifest_sha256)
    if any(file_hash(checkpoint / name) != value for name, value in protected.items()):
        raise GateBlocked("checkpoint/calibration changed during inference")
    verify_checkpoint_exposure(bundle, rows, protected['model.safetensors'])
    result = {"status": 'SYNTHETIC_FRAMEWORK_TEST_ONLY' if bundle['provenance_kind'] == 'synthetic_fixture' else 'INDEPENDENT_EVALUATION_ESTABLISHED',
              "production_ready": False, "fast_path_enabled": False, "live_verifier_n8n": "unverified",
              "inference": inference, "metrics": metrics, "bootstrap": ci, "exposure_provenance": exposure,
              "plan": plan, "plan_sha256": file_hash(output / 'evaluation-plan.json'),
              "predictions_sha256": file_hash(output / 'predictions.jsonl'), "split_counts": bundle['audit']['splits']}
    with (output / 'evaluation.json').open('xb') as f:
        f.write(canonical(result) + b'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('audit', 'lock'):
        cmd = sub.add_parser(name)
        for field in ('train', 'dev', 'test'):
            cmd.add_argument('--' + field, type=Path, required=True)
        cmd.add_argument('--near-threshold', type=float, default=0.85)
        cmd.add_argument('--max-candidates', type=int, default=10_000_000)
    build = sub.add_parser('build')
    build.add_argument('--pool', type=Path, required=True)
    build.add_argument('--seed', type=int, default=42)
    build.add_argument('--near-threshold', type=float, default=0.85)
    build.add_argument('--max-candidates', type=int, default=10_000_000)
    for name in ('build', 'lock'):
        cmd = sub.choices[name]
        for field in ('dataset-name', 'dataset-version', 'git-sha'):
            cmd.add_argument('--' + field, required=True)
        cmd.add_argument('--output', type=Path, required=True)
        cmd.add_argument('--created-at')
        cmd.add_argument('--exposure-record', type=Path)
    for name in ('verify', 'evaluate'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--bundle', type=Path, required=True)
        cmd.add_argument('--manifest-sha256', required=True)
    evaluate = sub.choices['evaluate']
    evaluate.add_argument('--checkpoint', type=Path, required=True)
    evaluate.add_argument('--output', type=Path, required=True)
    evaluate.add_argument('--batch-size', type=int, default=32)
    evaluate.add_argument('--bootstrap-seed', type=int, default=20261003)
    evaluate.add_argument('--bootstrap-replicates', type=int, default=2000)
    args = parser.parse_args(argv)
    try:
        if args.command in ('audit', 'lock'):
            splits = {role: read_rows(path) for role, path in zip(ROLE_FILES, (args.train, args.dev, args.test))}
            if args.command == 'audit':
                result = audit_splits(splits, near_threshold=args.near_threshold, max_candidates=args.max_candidates)
                print(json.dumps(result, indent=2, allow_nan=False))
                return 2 if result['cross_split_contamination'] else 0
        if args.command == 'build':
            splits = grouped_split(read_rows(args.pool), seed=args.seed, near_threshold=args.near_threshold, max_candidates=args.max_candidates)
        if args.command in ('build', 'lock'):
            sources = (args.pool,) if args.command == 'build' else (args.train, args.dev, args.test)
            h = lock_splits(splits, args.output, dataset_name=args.dataset_name, dataset_version=args.dataset_version,
                            git_sha=args.git_sha, created_at=args.created_at, seed=getattr(args, 'seed', None),
                            source_artifacts=sources, exposure_record=args.exposure_record, near_threshold=args.near_threshold,
                            max_candidates=args.max_candidates)
            result = {"status": "SPLITS_LOCKED_NOT_YET_EVALUATED", "manifest_sha256": h, "production_ready": False}
        elif args.command == 'verify':
            bundle, _ = verify_bundle(args.bundle, args.manifest_sha256)
            result = {"status": "HASH_AND_SPLIT_ISOLATION_VERIFIED", "independence": bundle['independence'], "production_ready": False}
        elif args.command == 'evaluate':
            result = evaluate_locked(args.bundle, args.manifest_sha256, args.checkpoint, args.output,
                                     batch_size=args.batch_size, bootstrap_seed=args.bootstrap_seed,
                                     bootstrap_replicates=args.bootstrap_replicates)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (GateBlocked, FileNotFoundError, ValueError, FileExistsError, OSError, KeyError) as exc:
        print(json.dumps({"status": "INDEPENDENT_EVALUATION_BLOCKED", "reason": str(exc), "production_ready": False,
                          "fast_path_enabled": False, "live_verifier_n8n": "unverified"}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
