from halluciguard_detector.data import _uniform_sentence_label
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
    # The four real RAGTruth label_type values are decorated, not bare. If this
    # regresses to exact matching, every span falls into the unknown fallback and
    # the entire training set flips to NOT_ENOUGH_INFO.
    conflict = [{"start": 0, "end": 20, "label_type": "Evident Conflict"}]
    subtle = [{"start": 0, "end": 20, "label_type": "Subtle Conflict"}]
    baseless = [{"start": 0, "end": 20, "label_type": "Evident Baseless Info"}]
    subtle_baseless = [{"start": 0, "end": 20, "label_type": "Subtle Baseless Info"}]
    implicit = [{"start": 0, "end": 20, "label_type": "Subtle Baseless Info", "implicit_true": True}]
    assert _uniform_sentence_label(0, 20, conflict) == "CONTRADICTED"
    assert _uniform_sentence_label(0, 20, subtle) == "CONTRADICTED"
    assert _uniform_sentence_label(0, 20, baseless) == "NOT_ENOUGH_INFO"
    assert _uniform_sentence_label(0, 20, subtle_baseless) == "NOT_ENOUGH_INFO"
    assert _uniform_sentence_label(0, 20, implicit) == "SUPPORTED"


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
