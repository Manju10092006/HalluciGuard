"""Retrieval query priority is not an evidence verdict or score floor."""
from types import SimpleNamespace
import pytest
from routers.query_expander import QueryExpander


@pytest.mark.parametrize("claim,canonical,anchor", [
    ("Python was created by Elon Musk.", "Elon Musk", "Python created by"),
    ("Elon Musk created Python.", "Elon Musk", "Python created by"),
    ("Elon Musk is the founder of Microsoft.", "Elon Musk", "Microsoft founder"),
    ("Bangkok is the capital of Vietnam.", "Bangkok", "capital of Vietnam"),
    ("The capital of Vietnam is Bangkok.", "Bangkok", "capital of Vietnam"),
])
def test_fact_anchor_precedes_proposed_entity(claim, canonical, anchor):
    expander = QueryExpander()
    expander.entity_resolver = SimpleNamespace(resolve=lambda *args: SimpleNamespace(canonical_query=canonical))
    queries = expander.generate_search_queries(claim, "general")
    assert queries[0] == claim
    assert queries[1] == anchor
    if canonical in queries:
        assert queries.index(anchor) < queries.index(canonical)
    assert len(queries) <= 4


def test_nonrelational_query_keeps_resolver_fallback():
    expander = QueryExpander()
    expander.entity_resolver = SimpleNamespace(resolve=lambda *args: SimpleNamespace(canonical_query="Example"))
    assert expander.generate_search_queries("Example appeared in a report.", "general") == [
        "Example appeared in a report.", "Example"]


def test_actual_finance_resolver_keeps_relation_before_entity_only_query():
    expander = QueryExpander()
    queries = expander.generate_search_queries("Microsoft was founded by Elon Musk.", "finance")
    assert queries[1] == "Microsoft founded by"
    assert queries.index("Microsoft founded by") < queries.index("Microsoft Corp MSFT")
