"""Read-only measurements of existing artifacts; never trains or overwrites them.

Run with python -m scripts.audit_detector_measurements --checkpoint ABSOLUTE_PATH.
Curated pairs below are software diagnostics, not a hallucination benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from transformers.utils import logging as transformers_logging
from halluciguard_detector.calibration import load_calibration
from halluciguard_detector.detector import Detector
from halluciguard_detector.training import evaluate, evaluate_saved_predictions


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    transformers_logging.disable_progress_bar()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path)
    parser.add_argument("--raw-data", type=Path)
    parser.add_argument("--retrieval-smoke", action="store_true")
    args = parser.parse_args()
    files = [args.checkpoint / name for name in
             ("model.safetensors", "config.json", "calibration.json", "test_predictions.npz")]
    before = {p.name: digest(p) for p in files}
    output = {"checkpoint": str(args.checkpoint), "device": "cuda" if torch.cuda.is_available() else "cpu",
              "checkpoint_sha256": before["model.safetensors"],
              "not_a_before_after_comparison": True}
    calibration = load_calibration(args.checkpoint / "calibration.json")
    output["calibration_metadata"] = calibration
    output["contradiction_threshold_provenance"] = (
        "checkpoint" if "contradiction_threshold" in calibration else "existing_runtime_default_not_fitted")
    saved = np.load(args.checkpoint / "test_predictions.npz", allow_pickle=False)
    output["archived_predictions_recomputed"] = evaluate_saved_predictions(
        saved["logits"], saved["labels"], calibration["temperature"],
        calibration.get("contradiction_threshold", 0.5), calibration["verification_risk_threshold"])
    if args.raw_data:
        summary = {}
        for filename in ("source_info.jsonl", "response.jsonl"):
            counts, malformed, seen, duplicate_ids, missing_ids, splits = 0, 0, set(), 0, 0, Counter()
            with (args.raw_data / filename).open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("not an object")
                        identifier = row.get("source_id" if filename == "source_info.jsonl" else "id")
                        if identifier is None:
                            missing_ids += 1
                        elif identifier in seen:
                            duplicate_ids += 1
                        else:
                            seen.add(identifier)
                        if row.get("split"):
                            splits[str(row["split"])] += 1
                        counts += 1
                    except (ValueError, TypeError):
                        malformed += 1
            summary[filename] = {"rows": counts, "malformed": malformed,
                                 "duplicate_ids": duplicate_ids, "missing_ids": missing_ids,
                                 "splits": dict(splits)}
        output["raw_file_inventory_not_annotation_validation"] = summary
    model = Detector(args.checkpoint)
    output["real_inference_curated_pairs"] = []
    for claim, evidence in (
        ("Java was created by James Gosling.", "Java was created by James Gosling."),
        ("Java was created by Snehith.", "Java was created by James Gosling."),
        ("Java was created in 2005.", "Java was first released in 1995."),
    ):
        trace = {}
        probabilities = model._classify(claim, evidence, trace=trace)
        guarded, warnings = model._guard_signals(claim, evidence, probabilities)
        output["real_inference_curated_pairs"].append({
            "claim": claim, "evidence": evidence,
            "model_probabilities": {key.value: value for key, value in probabilities.items()},
            "guard_preserved_probabilities": guarded == probabilities,
            "warnings": warnings, "tokenization": trace})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.prepared_data:
        inventory, groups = {}, {}
        for split in ("train", "dev", "test"):
            path = args.prepared_data / f"{split}.jsonl"
            labels, split_groups, missing_groups, rows = Counter(), set(), 0, 0
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows += 1
                    labels[str(row.get("label"))] += 1
                    if row.get("source_id") is None:
                        missing_groups += 1
                    else:
                        split_groups.add(str(row["source_id"]))
            groups[split] = split_groups
            inventory[split] = {"rows": rows, "labels": dict(labels),
                                "groups": len(split_groups), "missing_groups": missing_groups,
                                "sha256": digest(path)}
        output["prepared_split_inventory"] = inventory
        output["prepared_group_overlap"] = {
            f"{left}_{right}": len(groups[left] & groups[right])
            for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))}
        output["real_inference_prepared_test"] = evaluate(args.prepared_data, args.checkpoint, save=False)
        output["prepared_test_qualification"] = (
            "Existing file named test; its independent split provenance and historical "
            "training membership must be established before production accuracy claims.")
    if args.retrieval_smoke:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "verifier_agent"))
        from config.settings import get_settings
        get_settings().allow_model_downloads = False
        from schemas.models import Passage
        from retrievers.hybrid import HybridRetriever
        from rerankers.cross_encoder import CrossEncoderReranker
        passages = [Passage(title="Diagnostic", snippet=text, source="diagnostic_fixture",
                            source_id=f"fixture-{index}", url=f"https://example.com/{index}",
                            publication_date="")
                    for index, text in enumerate(("Java was created by James Gosling.",
                                                  "Python was created by Guido van Rossum."))]
        retriever, reranker = HybridRetriever(), CrossEncoderReranker()
        selected = retriever.retrieve("Who created Java?", passages, k=2)
        reranked = reranker.rerank("Who created Java?", selected, k=2)
        output["real_retrieval_attempt"] = {
            "retrieval": retriever.diagnostics(), "reranker": reranker.diagnostics(),
            "selected": len(selected), "reranked": len(reranked)}
    output["artifacts_unchanged"] = all(digest(p) == before[p.name] for p in files)
    def compact(value):
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items()
                    if key != "reliability_bins"}
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value
    print(json.dumps(compact(output), indent=2))


if __name__ == "__main__":
    main()
