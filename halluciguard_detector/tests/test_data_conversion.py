"""Regression tests for the RAGTruth -> three-class conversion.

The bugs guarded here are the ones that silently corrupt a training set rather
than crash it: label vocabulary drifting out of the alias table, overlapping
annotations summing to full coverage, and mixed sentences being force-labelled
instead of split.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from halluciguard_detector.data import (
    FULL_COVERAGE,
    _coverage,
    _match_concept,
    _non_factual_only,
    _regions,
    _snap_region,
    _uniform_sentence_label,
    annotation_concept,
    iter_ragtruth_examples,
    unknown_label_types,
)


# --------------------------------------------------------------------- labels


@pytest.mark.parametrize(
    "label_type, expected",
    [
        # The four real RAGTruth values.
        ("Evident Conflict", "CONTRADICTED"),
        ("Subtle Conflict", "CONTRADICTED"),
        ("Evident Baseless Info", "NOT_ENOUGH_INFO"),
        ("Subtle Baseless Info", "NOT_ENOUGH_INFO"),
        # Bare/alternative spellings seen in dataset revisions.
        ("conflict", "CONTRADICTED"),
        ("contradiction", "CONTRADICTED"),
        ("hallucinated", "NOT_ENOUGH_INFO"),
        ("evidence_insufficient", "NOT_ENOUGH_INFO"),
        ("insufficient_evidence", "NOT_ENOUGH_INFO"),
        ("evidence_sufficient", "SUPPORTED"),
        ("non_hallucinated", "SUPPORTED"),
        ("supported", "SUPPORTED"),
        ("non_factual", "NON_FACTUAL"),
        ("Non Factual", "NON_FACTUAL"),
        ("opinion", "NON_FACTUAL"),
    ],
)
def test_real_and_alternative_label_vocabulary(label_type, expected):
    assert _match_concept(label_type) == expected


def test_non_hallucinated_beats_embedded_hallucinat():
    """"non_hallucinated" contains "hallucinat"; the longer marker must win.

    If precedence were reversed, every "non_hallucinated" span would be read as
    NOT_ENOUGH_INFO -- the exact opposite of its meaning.
    """
    assert annotation_concept({"label_type": "non_hallucinated"}) == "SUPPORTED"
    assert annotation_concept({"label_type": "hallucinated"}) == "NOT_ENOUGH_INFO"


def test_unknown_vocabulary_falls_back_conservatively_and_is_reported():
    annotation = {"label_type": "Totally New Category", "start": 0, "end": 5}
    assert annotation_concept(annotation) == "NOT_ENOUGH_INFO"
    # Reported under the same normalised key the stats counter uses, so a new
    # dataset spelling shows up instead of silently skewing the label mix.
    assert unknown_label_types([annotation]) == {"totally_new_category"}


def test_known_vocabulary_is_not_reported_as_unknown():
    annotations = [
        {"label_type": "Evident Conflict", "start": 0, "end": 5},
        {"label_type": "Subtle Baseless Info", "start": 6, "end": 11},
    ]
    assert unknown_label_types(annotations) == set()


# ------------------------------------------------------------------- coverage


def test_coverage_uses_union_not_sum_of_overlaps():
    """Two annotations over the same half must not read as full coverage.

    Summing overlaps gave 1.0 for a sentence that was only half annotated, so it
    got force-labelled as if a human had marked every word.
    """
    annotations = [
        {"label_type": "Evident Conflict", "start": 0, "end": 10},
        {"label_type": "Evident Conflict", "start": 0, "end": 10},
    ]
    assert _coverage(0, 20, annotations, {"CONTRADICTED"}) == pytest.approx(0.5)
    assert _uniform_sentence_label(0, 20, annotations) is None


def test_fully_covered_sentence_gets_one_label():
    annotations = [{"label_type": "Evident Conflict", "start": 0, "end": 20}]
    assert _uniform_sentence_label(0, 20, annotations) == "CONTRADICTED"
    assert _uniform_sentence_label(0, 20, [{"label_type": "Evident Conflict", "start": 0, "end": 20}]) == "CONTRADICTED"


def test_barely_covered_sentence_is_mixed_not_labelled():
    span = 40
    covered = int(span * (FULL_COVERAGE - 0.05))
    annotations = [{"label_type": "Evident Conflict", "start": 0, "end": covered}]
    assert _uniform_sentence_label(0, span, annotations) is None


def test_unannotated_sentence_is_supported():
    assert _uniform_sentence_label(0, 20, []) == "SUPPORTED"


def test_implicit_true_annotations_are_ignored():
    annotations = [
        {"label_type": "Subtle Baseless Info", "start": 0, "end": 20, "implicit_true": True}
    ]
    assert _uniform_sentence_label(0, 20, annotations) == "SUPPORTED"


def test_non_factual_only_sentence_is_detected():
    annotations = [{"label_type": "Non Factual", "start": 0, "end": 20}]
    assert _non_factual_only(0, 20, annotations) is True
    assert _non_factual_only(0, 20, [{"label_type": "Evident Conflict", "start": 0, "end": 20}]) is False


# --------------------------------------------------------------------- regions


def test_snap_region_grows_to_word_boundaries():
    text = "Java was created by James Gosling in 1995."
    lo, hi = _snap_region(text, 22, 27)
    grown = text[lo:hi]
    # Grows outwards, so it is a superset of the requested slice ...
    assert lo <= 22 and hi >= 27
    assert text[22:27] in grown
    # ... and every edge lands on a word boundary, so no mangled fragment
    # like "ames Gosling in" is ever emitted as a training claim.
    assert lo == 0 or text[lo - 1].isspace() or text[lo - 1] in ".,;:!?()[]\"'"
    assert hi == len(text) or text[hi].isspace() or text[hi] in ".,;:!?()[]\"'"
    assert grown == "James Gosling"
    assert grown == grown.strip()


def test_mixed_sentence_splits_into_per_region_examples():
    text = "Java was created by Snehith in 1995. It supports many databases."
    annotations = [
        {"label_type": "Evident Conflict", "start": 20, "end": 27},  # "Snehith"
        {"label_type": "Evident Baseless Info", "start": 45, "end": 47},
    ]
    regions = _regions(0, len(text), annotations, text)
    labels = {label for _, label in regions}
    assert labels <= {"CONTRADICTED", "NOT_ENOUGH_INFO", "SUPPORTED"}
    assert regions, "a mixed sentence must still yield its annotated regions"
    for region_text, _ in regions:
        assert region_text.strip()
        assert region_text[0].isupper() or region_text[0].islower()


def test_region_split_keeps_the_human_label_of_each_span():
    text = "Java was created by Snehith in 1995. It supports many databases."
    annotations = [
        {"label_type": "Evident Conflict", "start": 20, "end": 27},
        {"label_type": "Evident Baseless Info", "start": 45, "end": 47},
    ]
    regions = dict(_regions(0, len(text), annotations, text))
    contradicted = [t for t, lab in regions.items() if lab == "CONTRADICTED"]
    assert contradicted, "the conflict span must be emitted as CONTRADICTED"
    assert any("Snehith" in t for t in contradicted)


def test_non_factual_regions_are_dropped():
    text = "This is the best language ever. It runs everywhere quickly."
    annotations = [
        {"label_type": "Non Factual", "start": 0, "end": 29},
        {"label_type": "Evident Baseless Info", "start": 30, "end": 50},
    ]
    regions = _regions(0, len(text), annotations, text)
    for region_text, label in regions:
        assert label != "NON_FACTUAL"


# ------------------------------------------------------------- end-to-end iter


def _write_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "ragtruth"
    dataset.mkdir()
    (dataset / "source_info.jsonl").write_text(
        json.dumps(
            {
                "source_id": 0,
                "task_type": "QA",
                "source_info": "Java was created by James Gosling at Sun Microsystems in 1995.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    response = (
        "Java was created by Snehith in 1995. It supports many databases "
        "and runs on every platform."
    )
    annotations = [
        {"start": 20, "end": 27, "label_type": "Evident Conflict", "implicit_true": False},
        {"start": 40, "end": 52, "label_type": "Evident Baseless Info", "implicit_true": False},
    ]
    (dataset / "response.jsonl").write_text(
        json.dumps(
            {
                "id": 1,
                "source_id": 0,
                "split": "train",
                "quality": "good",
                "response": response,
                "labels": annotations,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset


def test_iter_ragtruth_yields_region_examples_for_mixed_sentence(tmp_path):
    dataset = _write_dataset(tmp_path)
    stats: Counter = Counter()
    examples = list(iter_ragtruth_examples(dataset, "train", stats=stats))

    assert examples, "mixed sentences must still produce training examples"
    assert all(e["label"] in {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"} for e in examples)
    assert all(isinstance(e["label_id"], int) for e in examples)
    assert all(e["evidence"] for e in examples)
    assert all(e["claim"].strip() for e in examples)
    # The conflict span must survive as its own CONTRADICTED example.
    assert any(e["label"] == "CONTRADICTED" and "Snehith" in e["claim"] for e in examples)
    assert stats["span_level_examples"] >= 1
    # Granularity is recorded so train/serve claim-shape skew stays visible.
    assert {e["granularity"] for e in examples} == {"span"}


def test_iter_ragtruth_reports_unknown_vocabulary_in_stats(tmp_path):
    dataset = _write_dataset(tmp_path)
    rows = (dataset / "response.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[0])
    payload["labels"] = [{"start": 0, "end": 5, "label_type": "Brand New Label"}]
    (dataset / "response.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    stats: Counter = Counter()
    list(iter_ragtruth_examples(dataset, "train", stats=stats))
    assert stats["unknown_label_type:brand_new_label"] >= 1


def test_iter_ragtruth_skips_non_good_quality(tmp_path):
    dataset = _write_dataset(tmp_path)
    rows = (dataset / "response.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[0])
    payload["quality"] = "dev"
    (dataset / "response.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert list(iter_ragtruth_examples(dataset, "train")) == []


def test_iter_ragtruth_drops_pure_opinion_sentences(tmp_path):
    dataset = _write_dataset(tmp_path)
    payload = json.loads((dataset / "response.jsonl").read_text(encoding="utf-8").splitlines()[0])
    payload["response"] = "This is by far the best language ever written."
    payload["labels"] = [{"start": 0, "end": 44, "label_type": "Non Factual", "implicit_true": False}]
    (dataset / "response.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    stats: Counter = Counter()
    examples = list(iter_ragtruth_examples(dataset, "train", stats=stats))
    assert examples == []
    assert stats["non_factual_sentences_dropped"] >= 1


# ------------------------------------------------------------ prepare + stats


def _write_two_split_dataset(tmp_path: Path) -> Path:
    dataset = _write_dataset(tmp_path)
    train_row = json.loads((dataset / "response.jsonl").read_text(encoding="utf-8").splitlines()[0])
    dev_row = dict(train_row, id=2, split="dev", response="Java was created by James Gosling in 1995.", labels=[])
    test_row = dict(train_row, id=3, split="test", response="Python was created by Guido van Rossum in 1991.", labels=[])
    with (dataset / "response.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dev_row) + "\n")
        handle.write(json.dumps(test_row) + "\n")
    return dataset


def test_prepare_ragtruth_writes_stats_without_nameerror(tmp_path):
    """``stats.json`` must actually be writable.

    It previously dereferenced the removed ``_ALIAS_LOOKUP``, so the whole
    preparation step raised ``NameError`` after writing every data file -- the
    expensive part had already happened.
    """
    from halluciguard_detector.data import prepare_ragtruth

    dataset = _write_two_split_dataset(tmp_path)
    output = tmp_path / "out"
    stats = prepare_ragtruth(dataset, output, seed=42, dev_fraction=0.5)

    on_disk = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    assert on_disk["label_noise_controls"]["known_label_markers"]
    assert "conflict" in on_disk["label_noise_controls"]["known_label_markers"]
    assert on_disk["label_noise_controls"]["full_coverage_threshold"] == pytest.approx(FULL_COVERAGE)
    assert set(stats) >= {"train", "dev", "test", "label_noise_controls"}


def test_prepare_ragtruth_preserves_real_label_vocabulary_on_disk(tmp_path):
    """The decisive end-to-end check for the alias-table bug.

    If "Evident Conflict" stops resolving, every example silently becomes
    NOT_ENOUGH_INFO and the model trains on a corrupted label distribution.
    """
    from halluciguard_detector.data import prepare_ragtruth

    dataset = _write_two_split_dataset(tmp_path)
    output = tmp_path / "out"
    prepare_ragtruth(dataset, output, seed=42, dev_fraction=0.5)

    labels = set()
    for split in ("train", "dev", "test"):
        path = output / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            labels.add(json.loads(line)["label"])
    assert "CONTRADICTED" in labels, "the conflict span must survive as CONTRADICTED"
