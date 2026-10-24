"""Annotation integrity through the actual converter; no ML runtime is needed."""
from itertools import permutations
import subprocess
import sys

import pytest

from halluciguard_detector.data import _regions, _uniform_sentence_label


def test_data_converter_import_does_not_load_models():
    result = subprocess.run(
        [sys.executable, "-S", "-c",
         "import halluciguard_detector.data, sys; "
         "assert not {'torch', 'transformers', 'numpy', 'sentence_transformers'} & set(sys.modules)"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("text,annotations", [
    # Official RAGTruth response 5714, sentence 154:272, offsets made local.
    ("Their hours of operation are from 5:00 PM to 10:00 PM Monday through Friday, "
     "and from 5:00 PM to 10:00 PM on Saturday.", [
         {"start": 34, "end": 60, "label_type": "Evident Baseless Info"},
         {"start": 34, "end": 76, "label_type": "Evident Conflict"},
     ]),
    # Response 9700, sentence 624:722: the conflict does not cover the union.
    ("The store offers free WiFi and outdoor seating, and accepts reservations "
     "for parties of all sizes.", [
         {"start": 52, "end": 72, "label_type": "Evident Conflict"},
         {"start": 52, "end": 97, "label_type": "Evident Baseless Info"},
     ]),
])
def test_incompatible_overlaps_do_not_get_an_order_dependent_label(text, annotations):
    for order in permutations(annotations):
        assert _uniform_sentence_label(0, len(text), list(order)) is None
        assert _regions(0, len(text), list(order), text) == []


def test_adjacent_different_labels_remain_separate():
    text = "Wrong fact. Missing fact."
    annotations = [
        {"start": 0, "end": 12, "label_type": "Evident Conflict"},
        {"start": 12, "end": len(text), "label_type": "Evident Baseless Info"},
    ]
    for order in permutations(annotations):
        assert _regions(0, len(text), list(order), text) == [
            ("Wrong fact.", "CONTRADICTED"),
            ("Missing fact.", "NOT_ENOUGH_INFO"),
        ]


def test_same_label_overlaps_and_adjacent_regions_still_merge():
    text = "Wrong fact here."
    annotations = [
        {"start": 0, "end": 7, "label_type": "Evident Conflict"},
        {"start": 5, "end": 10, "label_type": "Subtle Conflict"},
        {"start": 10, "end": len(text), "label_type": "Evident Conflict"},
    ]
    for order in permutations(annotations):
        assert _regions(0, len(text), list(order), text) == [(text, "CONTRADICTED")]


def test_ambiguous_overlap_does_not_remove_an_adjacent_clean_region():
    text = "Wrong fact. Missing fact."
    annotations = [
        {"start": 0, "end": 12, "label_type": "Evident Conflict"},
        {"start": 0, "end": 5, "label_type": "Evident Baseless Info"},
        {"start": 12, "end": len(text), "label_type": "Evident Baseless Info"},
    ]
    for order in permutations(annotations):
        assert _regions(0, len(text), list(order), text) == [
            ("Missing fact.", "NOT_ENOUGH_INFO"),
        ]


def test_adjacent_same_label_is_not_merged_before_resolving_its_overlap():
    text = "Wrong fact. Missing fact."
    annotations = [
        {"start": 0, "end": 12, "label_type": "Evident Conflict"},
        {"start": 12, "end": len(text), "label_type": "Evident Conflict"},
        {"start": 12, "end": 19, "label_type": "Evident Baseless Info"},
    ]
    for order in permutations(annotations):
        assert _regions(0, len(text), list(order), text) == [
            ("Wrong fact.", "CONTRADICTED"),
        ]
