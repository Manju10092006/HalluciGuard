from halluciguard_detector.data import _sentence_label
from halluciguard_detector.text import has_entity_conflict, lexical_evidence, sentence_spans


def test_sentence_offsets_round_trip():
    text = "  Java was created in 1995. It was developed at Sun Microsystems!  "
    spans = sentence_spans(text)
    assert [span.text for span in spans] == [
        "Java was created in 1995.",
        "It was developed at Sun Microsystems!",
    ]
    assert all(text[span.start : span.end] == span.text for span in spans)


def test_annotation_mapping():
    conflict = [{"start": 5, "end": 10, "label_type": "Evident Conflict"}]
    baseless = [{"start": 5, "end": 10, "label_type": "Evident Baseless Info"}]
    implicit = [{"start": 5, "end": 10, "label_type": "Subtle Baseless Info", "implicit_true": True}]
    assert _sentence_label(0, 20, conflict) == "CONTRADICTED"
    assert _sentence_label(0, 20, baseless) == "NOT_ENOUGH_INFO"
    assert _sentence_label(0, 20, implicit) == "SUPPORTED"


def test_lexical_retrieval_prefers_relevant_sentence():
    docs = ["Python was created by Guido van Rossum. Java was created by James Gosling."]
    result = lexical_evidence("James Gosling created Java.", docs, limit=1)
    assert result == ["Java was created by James Gosling."]


def test_named_entity_conflict_guard_is_conservative():
    assert has_entity_conflict(
        "Java was created by Snehith in 1995.",
        "Java was designed by James Gosling at Sun Microsystems in 1995.",
    )
    assert not has_entity_conflict(
        "Java was released in 1995.",
        "Java was designed by James Gosling and released in 1995.",
    )
