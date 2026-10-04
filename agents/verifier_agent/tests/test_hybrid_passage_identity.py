"""Deterministic ranker fixtures; no embedding/reranker models are loaded."""
from schemas.models import Passage
from retrievers.hybrid import HybridRetriever


def passage(text, url="https://example.test/company", source_id="same_document"):
    return Passage(title="Company history", source="fixture", url=url,
                   publication_date="unknown", snippet=text, source_id=source_id)


class Backend:
    model_name = "fixture"

    def build_index(self, passages):
        self.passages = list(passages)

    def retrieve(self, query, k):
        return [(p, 1.0 / (i + 1)) for i, p in enumerate(self.passages[:k])]


def retriever():
    result = HybridRetriever.__new__(HybridRetriever)
    result.sparse = Backend()
    result.dense = Backend()
    return result


def test_different_excerpts_same_article_survive_both_retrieval_backends():
    inputs = [passage("Company was founded by Alice in 2002."),
              passage("Company launched its service in May 2003.")]
    model = retriever()
    outputs = model.retrieve("Company founding launch", inputs, k=5)
    assert {p.snippet for p in outputs} == {p.snippet for p in inputs}
    assert len(model.sparse.passages) == len(model.dense.passages) == 2
    assert all(p.url == inputs[0].url and p.source_id == "same_document" for p in outputs)


def test_high_overlap_cannot_discard_a_different_numeric_fact():
    prefix = (
        "The public company headquarters announcement records official founding year "
        "alongside corporate registration details address board membership financial "
        "statements shareholder resolutions incorporation documents historical archives "
        "press releases annual reports audit records management correspondence: "
    )
    inputs = [passage(prefix + "2002."), passage(prefix + "2003.")]
    assert HybridRetriever._compute_overlap(inputs[0].snippet, inputs[1].snippet) >= 0.92
    outputs = retriever().retrieve("company founding year", inputs, k=5)
    assert {p.snippet for p in outputs} == {p.snippet for p in inputs}


def test_exact_same_excerpt_whitespace_variation_is_not_counted_twice():
    inputs = [passage("Company was founded in 2002."),
              passage("Company  was founded\n in 2002.")]
    outputs = retriever().retrieve("company founded", inputs, k=5)
    assert len(outputs) == 1
