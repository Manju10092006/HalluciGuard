import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from .text import lexical_evidence, sentence_spans


LABEL_TO_ID = {"SUPPORTED": 0, "CONTRADICTED": 1, "NOT_ENOUGH_INFO": 2}


def _overlaps(start: int, end: int, annotation: dict) -> bool:
    return start < int(annotation["end"]) and end > int(annotation["start"])


def _sentence_label(start: int, end: int, annotations: list[dict]) -> str:
    relevant = [a for a in annotations if _overlaps(start, end, a) and not a.get("implicit_true")]
    if any("conflict" in a["label_type"].lower() for a in relevant):
        return "CONTRADICTED"
    if relevant:
        return "NOT_ENOUGH_INFO"
    return "SUPPORTED"


def iter_ragtruth_examples(dataset_dir: Path, split: str) -> Iterable[dict]:
    sources = {}
    with (dataset_dir / "source_info.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sources[str(row["source_id"])] = row
    with (dataset_dir / "response.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != split or row["quality"] != "good":
                continue
            source = sources[str(row["source_id"])]
            source_info = source["source_info"]
            if not isinstance(source_info, str):
                source_info = json.dumps(source_info, ensure_ascii=False, sort_keys=True)
            for span in sentence_spans(row["response"]):
                label = _sentence_label(span.start, span.end, row["labels"])
                snippets = lexical_evidence(span.text, [source_info])
                yield {
                    "id": f"{row['id']}:{span.start}-{span.end}",
                    "source_id": str(row["source_id"]),
                    "task_type": source["task_type"],
                    "evidence": " ".join(snippets),
                    "claim": span.text,
                    "label": label,
                    "label_id": LABEL_TO_ID[label],
                }


def prepare_ragtruth(
    dataset_dir: Path,
    output_dir: Path,
    seed: int = 42,
    dev_fraction: float = 0.1,
    max_supported_ratio: float = 2.0,
) -> dict:
    """Create leakage-safe splits grouped by source document."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_all = list(iter_ragtruth_examples(dataset_dir, "train"))
    test = list(iter_ragtruth_examples(dataset_dir, "test"))
    source_ids = sorted({x["source_id"] for x in train_all})
    rng = random.Random(seed)
    rng.shuffle(source_ids)
    dev_ids = set(source_ids[: max(1, round(len(source_ids) * dev_fraction))])
    raw_splits = {
        "train": [x for x in train_all if x["source_id"] not in dev_ids],
        "dev": [x for x in train_all if x["source_id"] in dev_ids],
        "test": test,
    }
    # Downsample only the training majority class. Evaluation remains natural-distribution.
    train = raw_splits["train"]
    positives = [x for x in train if x["label"] != "SUPPORTED"]
    supported = [x for x in train if x["label"] == "SUPPORTED"]
    rng.shuffle(supported)
    supported = supported[: int(max_supported_ratio * max(1, len(positives)))]
    raw_splits["train"] = positives + supported
    rng.shuffle(raw_splits["train"])
    stats = {}
    for name, records in raw_splits.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats[name] = {"total": len(records), "labels": dict(Counter(x["label"] for x in records))}
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
