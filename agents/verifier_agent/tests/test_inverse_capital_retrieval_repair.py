import pytest
from agents.verifier_agent.routers.query_expander import QueryExpander


@pytest.mark.parametrize("claim", [
    "The capital of Vietnam is Bangkok.",
    "The capital of Vietnam is Hanoi.",
    "Capital of Vietnam is Bangkok.",
    "Bangkok is the capital of Vietnam.",
])
def test_capital_search_does_not_depend_on_claimed_city(claim):
    queries = QueryExpander().generate_search_queries(claim, "general")
    assert "capital of Vietnam" in queries
    assert any("Bangkok" in q or "Hanoi" in q for q in queries)


def test_inverse_capital_preserves_multiword_country():
    queries = QueryExpander().generate_search_queries(
        "The capital of South Korea is Busan.", "general")
    assert "capital of South Korea" in queries
