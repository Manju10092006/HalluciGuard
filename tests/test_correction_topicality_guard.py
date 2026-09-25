"""Phase-4 invariant: a factually TRUE correction that answers a DIFFERENT
question than the user asked must not be silently accepted.

These lock the GENERIC topicality guard (`_answer_addresses_query`) — no
per-domain / hardcoded subject-relation rules. The guard can only ever WITHHOLD
an accept (downgrade a re-verification pass), never manufacture one.
"""
import pytest

from orchestration.graph import _answer_addresses_query


ON_TOPIC = [
    # Real corrections observed in the A/B/C controlled audit.
    ("Who is the founder of Microsoft?",
     "Bill Gates and Paul Allen are the founders of Microsoft."),
    ("What is the capital of India?",
     "The capital of India is New Delhi."),
    # Generic, non-Microsoft, to prove the check is not domain-specific.
    ("Who wrote the novel Dracula?", "Bram Stoker wrote the novel Dracula."),
    ("Who discovered penicillin?",
     "Alexander Fleming discovered penicillin in 1928."),
]

OFF_TOPIC = [
    # True-but-answers-a-different-question traps.
    ("Who founded Microsoft?",
     "Snehith Muvva is a Principal Product Manager at Microsoft."),
    ("What is the capital of India?",
     "Indianapolis is a city in the US state of Indiana."),
    ("Who painted the Mona Lisa?",
     "The Eiffel Tower is located in Paris, France."),
]


@pytest.mark.parametrize("query,answer", ON_TOPIC)
def test_on_topic_corrections_are_allowed(query, answer):
    assert _answer_addresses_query(query, answer) is True


@pytest.mark.parametrize("query,answer", OFF_TOPIC)
def test_off_topic_true_corrections_are_blocked(query, answer):
    assert _answer_addresses_query(query, answer) is False


def test_empty_query_never_blocks():
    # No salient tokens to score against -> must not withhold an accept.
    assert _answer_addresses_query("", "Anything at all.") is True


def test_empty_answer_is_off_topic():
    assert _answer_addresses_query("Who founded Microsoft?", "") is False
