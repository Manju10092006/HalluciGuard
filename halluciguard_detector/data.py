import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .text import lexical_evidence, sentence_spans
from .evidence_shapes import normalize_evidence


LABEL_TO_ID = {"SUPPORTED": 0, "CONTRADICTED": 1, "NOT_ENOUGH_INFO": 2}
_ANNOTATION_LABELS = {
    "evident conflict": "CONTRADICTED",
    "subtle conflict": "CONTRADICTED",
    "evident baseless info": "NOT_ENOUGH_INFO",
    "subtle baseless info": "NOT_ENOUGH_INFO",
    "evident baseless information": "NOT_ENOUGH_INFO",
    "subtle baseless information": "NOT_ENOUGH_INFO",
}


def _annotation_label(value: str) -> str:
    normalized = re.sub(r"[\s_-]+", " ", str(value).strip().lower())
    try:
        return _ANNOTATION_LABELS[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported RAGTruth label_type: {value!r}") from exc


def _overlaps(start: int, end: int, annotation: dict) -> bool:
    return start < int(annotation["end"]) and end > int(annotation["start"])


def _sentence_label(start: int, end: int, annotations: list[dict]) -> str:
    relevant = [a for a in annotations if _overlaps(start, end, a) and not a.get("implicit_true")]
    labels = [_annotation_label(a["label_type"]) for a in relevant]
    if "CONTRADICTED" in labels:
        return "CONTRADICTED"
    if labels:
        return "NOT_ENOUGH_INFO"
    return "SUPPORTED"


def iter_ragtruth_examples(dataset_dir: Path, split: str) -> Iterable[dict]:
    def records(name: str):
        path = dataset_dir / name
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{name} line {number}: empty JSONL row")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{name} line {number}: malformed JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{name} line {number}: expected JSON object")
                yield number, row

    def identifier(row: dict, key: str, location: str) -> str:
        value = row.get(key)
        if value is None or not str(value).strip():
            raise ValueError(f"{location}: missing {key}")
        return str(value)

    sources = {}
    for number, row in records("source_info.jsonl"):
        source_id = identifier(row, "source_id", f"source_info.jsonl line {number}")
        if source_id in sources:
            raise ValueError(f"duplicate source_id: {source_id}")
        if "source_info" not in row or "task_type" not in row:
            raise ValueError(f"source_info.jsonl line {number}: missing source_info or task_type")
        sources[source_id] = row
    seen_response_ids: set[str] = set()
    for number, row in records("response.jsonl"):
        if row.get("split") != split or row.get("quality") != "good":
            continue
        location = f"response.jsonl line {number}"
        response_id = identifier(row, "id", location)
        if response_id in seen_response_ids:
            raise ValueError(f"duplicate response id in {split}: {response_id}")
        seen_response_ids.add(response_id)
        source_id = identifier(row, "source_id", location)
        if source_id not in sources:
            raise ValueError(f"missing source_id for response {response_id}: {source_id}")
        source = sources[source_id]
        normalized = normalize_evidence(source["source_info"])
        if not normalized.documents or normalized.malformed_records or normalized.truncated_records:
            raise ValueError(f"source {source_id} has incomplete or malformed evidence")
        response = row.get("response")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"response {response_id} is empty or malformed")
        annotations = row.get("labels")
        if not isinstance(annotations, list):
            raise ValueError(f"response {response_id} has missing or malformed labels")
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError(f"response {response_id} has malformed annotation")
            try:
                start, end = annotation["start"], annotation["end"]
                if type(start) is not int or type(end) is not int or not (0 <= start < end <= len(response)):
                    raise ValueError("invalid offsets")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"response {response_id} has out-of-range annotation or malformed offsets") from exc
            if not annotation.get("implicit_true"):
                if "label_type" not in annotation:
                    raise ValueError(f"response {response_id} has missing annotation label_type")
                _annotation_label(annotation["label_type"])
        for span in sentence_spans(response):
            label = _sentence_label(span.start, span.end, annotations)
            snippets = lexical_evidence(span.text, normalized.documents)
            yield {
                "id": f"{response_id}:{span.start}-{span.end}",
                "source_id": source_id,
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
    if not 0.0 < dev_fraction < 1.0 or max_supported_ratio < 0:
        raise ValueError("invalid split fraction or supported ratio")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("prepared output directory is not empty; preserve existing artifacts")
    train_all = list(iter_ragtruth_examples(dataset_dir, "train"))
    test = list(iter_ragtruth_examples(dataset_dir, "test"))
    if not train_all or not test:
        raise ValueError("RAGTruth preparation requires nonempty train and test examples")
    source_ids = sorted({x["source_id"] for x in train_all})
    if len(source_ids) < 2:
        raise ValueError("RAGTruth preparation requires at least two train source groups")
    rng = random.Random(seed)
    rng.shuffle(source_ids)
    dev_ids = set(source_ids[: max(1, round(len(source_ids) * dev_fraction))])
    if len(dev_ids) >= len(source_ids):
        raise ValueError("dev split would consume every train source group")
    raw_splits = {
        "train": [x for x in train_all if x["source_id"] not in dev_ids],
        "dev": [x for x in train_all if x["source_id"] in dev_ids],
        "test": test,
    }
    if {x["source_id"] for x in train_all} & {x["source_id"] for x in test}:
        raise ValueError("source_id leakage between official train and test splits")
    if not raw_splits["train"] or not raw_splits["dev"]:
        raise ValueError("grouped split produced an empty train or dev set")
    # Downsample only the training majority class. Evaluation remains natural-distribution.
    train = raw_splits["train"]
    positives = [x for x in train if x["label"] != "SUPPORTED"]
    supported = [x for x in train if x["label"] == "SUPPORTED"]
    rng.shuffle(supported)
    supported = supported[: int(max_supported_ratio * max(1, len(positives)))]
    raw_splits["train"] = positives + supported
    if not raw_splits["train"]:
        raise ValueError("training downsampling produced an empty train set")
    rng.shuffle(raw_splits["train"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, records in raw_splits.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats[name] = {"total": len(records), "labels": dict(Counter(x["label"] for x in records)),
                       "source_groups": len({x["source_id"] for x in records})}
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
