"""No models or network: preserve different facts and source provenance."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregation.duplicate_remover import DuplicateRemover
from schemas.models import Passage


def passage(text, *, url="https://example.org/article", title="Article",
            source="official", source_id="", score=0.8):
    return Passage(title=title, source=source, url=url,
                   publication_date="unknown", snippet=text,
                   source_id=source_id, relevance_score=score)


def test_sections_from_one_article_preserve_complementary_facts():
    founding = passage("The company was founded in December 2002.",
                       url="https://example.org/article#Founding", score=0.9)
    launch = passage("The platform launched on May 5, 2003.",
                     url="https://example.org/article#Overview", score=0.7)
    result = DuplicateRemover().remove_duplicates([launch, founding])
    assert result == [founding, launch]
    assert result[0] is founding and result[1] is launch
    assert result[1].url.endswith("#Overview")


def test_root_url_repeated_with_different_excerpts_keeps_both():
    first = passage("The company was founded in 2002.", score=0.9)
    second = passage("Its first product was released in 2003.", score=0.8)
    assert DuplicateRemover().remove_duplicates([first, second]) == [first, second]


def test_urlless_same_title_source_different_content_keeps_both():
    first = passage("The treatment is effective in adults.", url="")
    second = passage("The treatment is not approved for children.", url="")
    assert DuplicateRemover().remove_duplicates([first, second]) == [first, second]


def test_exact_whitespace_duplicate_keeps_best_original_with_provenance():
    low = passage("The company\n was\t founded in 2002.",
                  url="https://EXAMPLE.org/article/?utm_campaign=test", score=0.5)
    best = passage("The company was founded in 2002.",
                   url="https://example.org/article#history", source_id="best", score=0.9)
    result = DuplicateRemover().remove_duplicates([low, best])
    assert result == [best]
    assert result[0] is best
    assert result[0].source_id == "best"
    assert result[0].url.endswith("#history")


@pytest.mark.parametrize(("original", "changed"), [
    ("The reviewed company was founded in the month of December in the year 2002 by its original team.",
     "The reviewed company was founded in the month of December in the year 2003 by its original team."),
    ("The reported final contract price for this particular product includes a total of 100 dollars in local sales tax.",
     "The reported final contract price for this particular product includes a total of 200 dollars in local sales tax."),
    ("The independently reviewed medicine is approved for use by adult patients under the published regional clinical guidance.",
     "The independently reviewed medicine is not approved for use by adult patients under the published regional clinical guidance."),
    ("The independently reviewed company is currently owned by a private holding group under the published regional corporate agreement.",
     "The independently reviewed company is formerly owned by a private holding group under the published regional corporate agreement."),
    ("Alice helped Bob establish the company.",
     "Bob helped Alice establish the company."),
])
def test_high_overlap_cannot_erase_changed_facts(original, changed):
    remover = DuplicateRemover()
    first, second = passage(original), passage(changed)
    assert remover._compute_token_overlap(original, changed) > 0.85
    assert remover.remove_duplicates([first, second]) == [first, second]


def test_independent_sources_with_same_text_keep_both_citations():
    first = passage("The company was founded in 2002.", url="https://example.org/a")
    second = passage("The company was founded in 2002.", url="https://independent.org/b")
    assert DuplicateRemover().remove_duplicates([first, second]) == [first, second]


def test_urlless_distinct_source_identifiers_keep_provenance():
    first = passage("The company was founded in 2002.", url="", source_id="record1")
    second = passage("The company was founded in 2002.", url="", source_id="record2")
    assert DuplicateRemover().remove_duplicates([first, second]) == [first, second]


def test_relevance_sort_is_stable_and_does_not_mutate_input():
    low = passage("Low ranking excerpt.", score=0.1)
    first = passage("First equally ranked excerpt.", score=0.8)
    second = passage("Second equally ranked excerpt.", score=0.8)
    inputs = [low, first, second]
    assert DuplicateRemover().remove_duplicates(inputs) == [first, second, low]
    assert inputs == [low, first, second]


def test_empty_content_or_unknown_document_identity_is_not_discarded():
    empty1, empty2 = passage(""), passage(" ")
    unknown1 = passage("Some factual excerpt.", url="", title="", source="")
    unknown2 = passage("Some factual excerpt.", url="", title="", source="")
    inputs = [empty1, empty2, unknown1, unknown2]
    assert DuplicateRemover().remove_duplicates(inputs) == inputs


def test_empty_input_and_diagnostic_overlap_contract():
    remover = DuplicateRemover(overlap_threshold=0.0)
    assert remover.remove_duplicates([]) == []
    assert remover.overlap_threshold == 0.0
    assert remover._compute_token_overlap("", "evidence") == 0.0
    first, second = passage("Fact number one."), passage("Fact number two.")
    assert remover.remove_duplicates([first, second]) == [first, second]
