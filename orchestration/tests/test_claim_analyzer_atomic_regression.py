"""Offline regressions for faithful claim extraction and provider JSON framing.

These tests establish segmentation/prompt contracts, not factual correctness.
No LLM request, retrieval request or model inference is made.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

import services.claim_analyzer as analyzer_module
from services.claim_analyzer import (
    ClaimAnalyzer,
    FACTUAL_CLAIM,
    INSTRUCTION,
    NON_FACTUAL,
    _extract_json,
    fallback_analyze,
)


LINKEDIN_DRAFT = """LinkedIn was founded in 2002 and launched in 2003 by a team led by **Reid\u202fHoffman**. The original co\u2011founders who helped build the platform alongside him were:

- **Allen\u202fBlue** \u2013 former PayPal executive and early product lead
- **Konstantin\u202fGuericke** \u2013 former PayPal executive, responsible for sales and business development
- **Eric\u202fLy** \u2013 former PayPal engineer, who built much of the early technology
- **Jean\u2011Luc\u202fVaillant** \u2013 early product and engineering contributor

Reid\u202fHoffman served as the company's founding CEO, and the group launched LinkedIn as a professional networking service in May\u202f2003."""


def factual_texts(draft, **kwargs):
    return [c.claim_text for c in fallback_analyze(draft, **kwargs).factual_claims]


def test_recorded_linkedin_draft_has_separate_date_events_and_list_members():
    analysis = fallback_analyze(LINKEDIN_DRAFT)
    claims = [c.claim_text for c in analysis.factual_claims]
    assert claims[:2] == [
        "LinkedIn was founded in 2002.",
        "LinkedIn was launched in 2003 by a team led by Reid Hoffman.",
    ]
    for person in ("Allen Blue", "Konstantin Guericke", "Eric Ly", "Jean\u2011Luc Vaillant"):
        assert f"LinkedIn was co-founded by {person}." in claims
    assert not any("Allen Blue" in c and "Konstantin Guericke" in c for c in claims)
    assert any("May 2003" in c and "founding CEO" in c for c in claims)
    assert len({c.claim_id for c in analysis.candidates}) == len(analysis.candidates)
    assert all(c.search_queries == [c.claim_text] for c in analysis.factual_claims)
    assert all(c.original_sentence in LINKEDIN_DRAFT for c in analysis.candidates)


def test_role_qualifiers_are_preserved_without_inventing_a_company_for_them():
    claims = factual_texts(LINKEDIN_DRAFT)
    assert "Allen Blue \u2013 former PayPal executive and early product lead" in claims
    assert "Eric Ly \u2013 former PayPal engineer, who built much of the early technology" in claims
    assert not any("LinkedIn product lead" in c or "LinkedIn engineer" in c for c in claims)


def test_punctuation_free_name_records_inherit_only_the_explicit_heading():
    draft = "Microsoft's co-founders were:\n- Bill Gates\n- Paul Allen"
    analysis = fallback_analyze(draft)
    assert [c.claim_text for c in analysis.factual_claims] == [
        "Microsoft was co-founded by Bill Gates.",
        "Microsoft was co-founded by Paul Allen.",
    ]
    assert analysis.discarded[0].claim_type == NON_FACTUAL
    assert all("source heading" in c.reason for c in analysis.factual_claims)


def test_independent_unpunctuated_bullets_are_not_joined():
    claims = factual_texts("- Google was founded in 1998\n- Microsoft was founded in 1975")
    assert claims == ["Google was founded in 1998", "Microsoft was founded in 1975"]


def test_multiline_record_preserves_qualifiers_but_not_next_record():
    claims = factual_texts(
        "Microsoft's co-founders were:\n"
        "- Bill Gates \u2013 a co-founder\n  who worked on BASIC in 1975\n"
        "- Paul Allen \u2013 a co-founder"
    )
    assert "Bill Gates \u2013 a co-founder who worked on BASIC in 1975" in claims
    assert not any("Bill Gates" in c and "Paul Allen" in c for c in claims)


@pytest.mark.parametrize(
    "draft",
    [
        "The original co-founders were:\n- Allen Blue \u2013 former executive",
        "LinkedIn and PayPal were founded in 2002.\nThe original co-founders were:\n- Allen Blue",
        "LinkedIn was founded in 2002. An unrelated organization was founded in 2010.\n"
        "The original co-founders were:\n- Allen Blue",
        "LinkedIn was founded in 2002.\nKnown Microsoft executives:\n- Allen Blue",
    ],
)
def test_ambiguous_or_unrelated_headers_never_fabricate_founder_membership(draft):
    claims = factual_texts(draft, user_query="Who founded LinkedIn?")
    assert not any("co-founded by Allen Blue" in c for c in claims)


def test_trailing_creator_is_not_falsely_assigned_to_the_earlier_event():
    claims = factual_texts("Example was founded in 2002 and launched in 2003 by a new CEO.")
    assert claims == [
        "Example was founded in 2002.",
        "Example was launched in 2003 by a new CEO.",
    ]
    assert "Example was founded by a new CEO." not in claims


def test_negation_and_ambiguous_compound_subject_are_not_rewritten():
    assert factual_texts("LinkedIn was not founded in 2002 and launched in 2003.") == [
        "LinkedIn was not founded in 2002 and launched in 2003."
    ]
    assert factual_texts("LinkedIn and PayPal were founded in 2002 and launched in 2003.") == [
        "LinkedIn and PayPal were founded in 2002 and launched in 2003."
    ]


def test_markdown_instruction_bullet_is_not_sent_to_retrieval():
    analysis = fallback_analyze("- Use the official website to check prices.")
    assert analysis.factual_claims == []
    assert analysis.candidates[0].claim_type == INSTRUCTION


@pytest.mark.parametrize(
    "content",
    [
        'Prelude {not JSON}\n{"domain":"general","claims":[]}\nTrailing {text}',
        '```json\n{"domain":"general","claims":[]}\n```',
        '{"domain":"general","claims":[{"claim_text":"The symbol is {brace}."}]}',
    ],
)
def test_json_decoder_accepts_complete_json_without_greedy_brace_capture(content):
    assert isinstance(_extract_json(content)["claims"], list)


def test_incomplete_outer_json_does_not_salvage_individual_claim_as_analysis():
    assert _extract_json('{"domain":"general","claims":[{"claim_text":"Google was founded in 1998."}') is None


class FakeLLM:
    def __init__(self, content, finish_reason="stop"):
        self.calls = []
        self.content = content
        self.finish_reason = finish_reason

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            status="success", draft_response=self.content, finish_reason=self.finish_reason
        )


def test_analyzer_sends_json_contract_as_system_message_in_one_provider_call(monkeypatch):
    monkeypatch.setattr(analyzer_module, "ANALYZER_MAX_TOKENS", 4096)
    llm = FakeLLM('{"domain":"general","claims":[{"claim_text":"Google was founded in 1998."}]}')
    analysis = asyncio.run(ClaimAnalyzer(llm_service=llm).analyze("Google was founded in 1998."))
    assert analysis.source == "llm"
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["conversation_history"][0]["role"] == "system"
    assert "Return ONLY a JSON object" in call["conversation_history"][0]["content"]
    assert "Google was founded in 1998." in call["user_query"]
    assert call["max_tokens"] == 4096
    assert call["temperature"] == 0.0


def test_truncated_provider_json_reports_length_and_falls_back_once():
    llm = FakeLLM('{"domain":"general","claims":[', finish_reason="length")
    analysis = asyncio.run(ClaimAnalyzer(llm_service=llm).analyze(LINKEDIN_DRAFT))
    assert analysis.source == "fallback"
    assert analysis.llm_error == "incomplete LLM JSON (finish_reason=length)"
    assert len(llm.calls) == 1
    assert "LinkedIn was co-founded by Allen Blue." in [c.claim_text for c in analysis.factual_claims]


def test_explicit_token_budget_environment_override_is_preserved(monkeypatch):
    monkeypatch.setenv("HALLUCIGUARD_CLAIM_ANALYZER_MAX_TOKENS", "1200")
    assert analyzer_module._env_int("HALLUCIGUARD_CLAIM_ANALYZER_MAX_TOKENS", 4096) == 1200


def test_list_name_with_period_does_not_add_duplicate_punctuation():
    claims = factual_texts("Microsoft's co-founders were:\n- Bill Gates.")
    assert claims == ["Microsoft was co-founded by Bill Gates."]


def test_prompt_requires_self_contained_claims_without_external_entity_guesses():
    draft = (
        "The founding team included Allen Blue and Eric Ly. "
        "These individuals are credited as co-founders of LinkedIn."
    )
    llm = FakeLLM('{"domain":"general","claims":[]}')
    asyncio.run(ClaimAnalyzer(llm_service=llm, enabled=True).analyze(draft))
    request = llm.calls[0]
    prompt = " ".join(request["user_query"].split())
    assert "one unambiguous explicit antecedent" in prompt
    assert "never names from the user question or outside knowledge" in prompt
    assert "ambiguous referent" in prompt
    assert "do not discard it or guess" in prompt
    assert "identical explicit referents, relationship and ALL qualifiers" in prompt
    assert "Do not omit an unresolved pronoun assertion" in prompt
    assert "unambiguous explicit draft antecedents" in request["conversation_history"][0]["content"]


def test_coercion_preserves_primary_and_credited_as_distinct_qualifiers():
    original = "The company's primary founder is Reid Hoffman."
    _, candidates = analyzer_module._coerce_candidates({
        "domain": "general",
        "claims": [
            {
                "claim_id": "c1", "claim_text": "Reid Hoffman is the primary founder of LinkedIn.",
                "original_sentence": original, "claim_type": FACTUAL_CLAIM,
            },
            {
                "claim_id": "c2", "claim_text": "Reid Hoffman is credited as a co-founder of LinkedIn.",
                "original_sentence": "Reid Hoffman is credited as a co-founder of LinkedIn.",
                "claim_type": FACTUAL_CLAIM,
            },
            {
                "claim_id": "c3", "claim_text": "Reid Hoffman is a co-founder of LinkedIn.",
                "original_sentence": "Reid Hoffman is a co-founder of LinkedIn.",
                "claim_type": FACTUAL_CLAIM,
            },
        ],
    })
    assert [c.claim_text for c in candidates] == [
        "Reid Hoffman is the primary founder of LinkedIn.",
        "Reid Hoffman is credited as a co-founder of LinkedIn.",
        "Reid Hoffman is a co-founder of LinkedIn.",
    ]
    assert candidates[0].original_sentence == original
    assert all(c.is_factual for c in candidates)


def test_coercion_does_not_guess_or_drop_an_unresolved_pronoun_assertion():
    assertion = "These individuals are credited as the co-founders of LinkedIn."
    _, candidates = analyzer_module._coerce_candidates({
        "claims": [{
            "claim_id": "c8", "claim_text": assertion,
            "original_sentence": assertion, "claim_type": FACTUAL_CLAIM,
            "reason": "ambiguous referent", "search_queries": ["LinkedIn credited co-founders"],
        }],
    })
    assert len(candidates) == 1
    assert candidates[0].claim_text == assertion
    assert candidates[0].claim_type == FACTUAL_CLAIM
    assert candidates[0].reason == "ambiguous referent"
    assert candidates[0].search_queries == ["LinkedIn credited co-founders"]


def test_prompt_and_coercion_keep_negation_dates_and_attribution():
    assertion = "Allen Blue was not credited as a co-founder of LinkedIn in 2003."
    llm = FakeLLM(
        '{"domain":"general","claims":[{"claim_text":'
        '"Allen Blue was not credited as a co-founder of LinkedIn in 2003.",'
        '"claim_type":"FACTUAL_CLAIM"}]}'
    )
    analysis = asyncio.run(ClaimAnalyzer(llm_service=llm, enabled=True).analyze(assertion))
    assert analysis.factual_claims[0].claim_text == assertion
    prompt = " ".join(llm.calls[0]["user_query"].split())
    assert "attribution, negation, numbers, dates, temporal scope" in prompt
    assert 'Do not weaken "primary founder" to "founder"' in prompt


LIVE_CREATOR_NAMES = [
    "Reid Hoffman", "Allen Blue", "Konstantin Guericke", "Eric Ly", "Jean\u2011Luc Vaillant",
]
LIVE_LIST_SENTENCE = (
    "LinkedIn was founded by Reid Hoffman, Allen Blue, Konstantin Guericke, "
    "Eric Ly, and Jean\u2011Luc Vaillant."
)
LIVE_ATTRIBUTION_SENTENCE = "Each of them is recognized as a co\u2011founder of the company."
LIVE_LIST_DRAFT = f"{LIVE_LIST_SENTENCE} {LIVE_ATTRIBUTION_SENTENCE}"


@pytest.mark.parametrize("hosted", [False, True])
def test_live_compound_creator_list_and_attribution_become_distinct_source_bound_claims(hosted):
    if hosted:
        llm = FakeLLM(json.dumps({"domain": "general", "claims": [
            {"claim_id": "c1", "claim_text": LIVE_LIST_SENTENCE,
             "original_sentence": LIVE_LIST_SENTENCE, "claim_type": FACTUAL_CLAIM},
            {"claim_id": "c2", "claim_text": LIVE_ATTRIBUTION_SENTENCE,
             "original_sentence": LIVE_ATTRIBUTION_SENTENCE, "claim_type": FACTUAL_CLAIM},
        ]}))
        analysis = asyncio.run(ClaimAnalyzer(llm_service=llm).analyze(
            LIVE_LIST_DRAFT, user_query="Who founded a different company?"
        ))
        assert analysis.source == "llm"
        assert len(llm.calls) == 1
    else:
        analysis = fallback_analyze(LIVE_LIST_DRAFT, user_query="Who founded a different company?")
    factual = analysis.factual_claims
    assert [c.claim_text for c in factual] == [
        *[f"LinkedIn was founded by {name}." for name in LIVE_CREATOR_NAMES],
        *[f"{name} is recognized as a co\u2011founder of LinkedIn." for name in LIVE_CREATOR_NAMES],
    ]
    assert all(c.original_sentence == LIVE_LIST_SENTENCE for c in factual[:5])
    assert all(c.original_sentence == LIVE_ATTRIBUTION_SENTENCE for c in factual[5:])
    assert all(c.search_queries == [c.claim_text] for c in factual)
    assert len({c.claim_id for c in analysis.candidates}) == len(analysis.candidates)


@pytest.mark.parametrize("hosted", [False, True])
def test_creator_list_preserves_distributable_modifier_unicode_and_date(hosted):
    original = "LinkedIn was originally co\u2011founded by Reid Hoffman, Allen Blue and Jean\u2011Luc Vaillant in 2002."
    if hosted:
        _, candidates = analyzer_module._coerce_candidates({"claims": [{
            "claim_id": "c9", "claim_text": original, "original_sentence": original,
        }]}, original)
    else:
        candidates = fallback_analyze(original).factual_claims
    assert [c.claim_text for c in candidates] == [
        f"LinkedIn was originally co\u2011founded by {name} in 2002."
        for name in ("Reid Hoffman", "Allen Blue", "Jean\u2011Luc Vaillant")
    ]
    assert all(c.original_sentence == original for c in candidates)


@pytest.mark.parametrize("original", [
    "LinkedIn was not founded by Reid Hoffman, Allen Blue and Eric Ly.",
    "LinkedIn was founded by Reid Hoffman, Allen Blue or Eric Ly.",
    "LinkedIn was founded only by Reid Hoffman, Allen Blue and Eric Ly.",
    "LinkedIn was solely founded by Reid Hoffman, Allen Blue and Eric Ly.",
    "LinkedIn and PayPal were founded by Reid Hoffman, Allen Blue and Eric Ly.",
    "The company was founded by Reid Hoffman, Allen Blue and Eric Ly.",
    "LinkedIn was founded by Reid Hoffman, The Engineering Team and Eric Ly.",
    "LinkedIn was founded by Reid Hoffman,, Allen Blue and Eric Ly.",
])
def test_unsafe_creator_lists_remain_factual_and_unmodified_in_both_paths(original):
    assert factual_texts(original) == [original]
    _, candidates = analyzer_module._coerce_candidates({"claims": [{"claim_text": original}]})
    assert [c.claim_text for c in candidates] == [original]


@pytest.mark.parametrize("draft", [
    "Each of them is recognized as a co-founder of the company.",
    "LinkedIn and PayPal were founded by Reid Hoffman and Allen Blue. "
    "Each of them is recognized as a co-founder of the company.",
    "LinkedIn was not founded by Reid Hoffman and Allen Blue. "
    "Each of them is recognized as a co-founder of the company.",
    "LinkedIn was founded by Reid Hoffman or Allen Blue. "
    "Each of them is recognized as a co-founder of the company.",
    "LinkedIn was founded by Reid Hoffman and Allen Blue. Another company was mentioned. "
    "Each of them is recognized as a co-founder of the company.",
    "LinkedIn was founded by Reid Hoffman and Allen Blue in partnership with PayPal. "
    "Each of them is recognized as a co-founder of the company.",
    "LinkedIn was founded by Reid Hoffman and Allen Blue. "
    "Each of them is recognized as a co-founder of Microsoft.",
])
def test_ambiguous_list_coreference_is_not_guessed_from_question_or_other_sentences(draft):
    assertion = draft.split(". ")[-1]
    _, candidates = analyzer_module._coerce_candidates({"claims": [{
        "claim_text": assertion, "original_sentence": assertion,
    }]}, draft)
    assert [c.claim_text for c in candidates] == [assertion]
    assert assertion in factual_texts(draft, user_query="Who founded LinkedIn?")


def test_bounded_each_attribution_resolution_preserves_negation_and_date():
    assertion = "Each of them was not credited as a co-founder of the company in 2003."
    draft = f"LinkedIn was founded by Reid Hoffman and Allen Blue in 2002. {assertion}"
    _, candidates = analyzer_module._coerce_candidates({"claims": [{
        "claim_text": assertion, "original_sentence": assertion,
    }]}, draft)
    assert [c.claim_text for c in candidates] == [
        "Reid Hoffman was not credited as a co-founder of LinkedIn in 2003.",
        "Allen Blue was not credited as a co-founder of LinkedIn in 2003.",
    ]
    assert all(c.original_sentence == assertion for c in candidates)


def test_hosted_atomic_expansion_retains_nonfactual_records_and_unique_duplicate_ids():
    original = "LinkedIn was founded by Reid Hoffman, Allen Blue and Eric Ly."
    _, candidates = analyzer_module._coerce_candidates({"claims": [
        {"claim_id": "c1", "claim_text": original},
        {"claim_id": "c1-1", "claim_text": "LinkedIn launched in 2003."},
        {"claim_id": "c1-1", "claim_text": "Use its official website.", "claim_type": INSTRUCTION},
    ]})
    assert len(candidates) == 5
    assert len({c.claim_id for c in candidates}) == 5
    assert candidates[-1].claim_type == INSTRUCTION
    assert candidates[-1].claim_text == "Use its official website."


def test_hosted_reference_without_verbatim_source_alignment_is_not_rewritten():
    changed_assertion = "Each of them is credited as a co-founder of the company."
    _, candidates = analyzer_module._coerce_candidates({"claims": [{
        "claim_text": changed_assertion, "original_sentence": LIVE_ATTRIBUTION_SENTENCE,
    }]}, LIVE_LIST_DRAFT)
    assert [c.claim_text for c in candidates] == [changed_assertion]
