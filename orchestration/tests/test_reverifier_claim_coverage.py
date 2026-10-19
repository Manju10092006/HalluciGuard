"""Offline re-verification contracts; no retrieval services or model weights."""
from types import SimpleNamespace

import pytest

from orchestration import graph
from services import claim_analyzer


def _state(text):
    return {"user_query": "Tell me about LinkedIn.", "domain": "general",
            "request_id": "coverage-test", "trace": [],
            "correction_result": {"corrected_text": text}}


@pytest.mark.parametrize("query", ["Tell me about LinkedIn.", "Please explain LinkedIn.",
                                  "Describe LinkedIn."])
def test_request_verbs_are_not_required_topics_in_a_corrected_answer(query):
    assert graph._answer_addresses_query(query, "LinkedIn was founded in 2002.")
    assert not graph._answer_addresses_query(query, "Microsoft was founded in 1975.")


def test_removing_request_verbs_does_not_allow_wrong_relationship_correction():
    assert not graph._answer_addresses_query(
        "Please tell me who founded Microsoft.",
        "Snehith is a product manager at Microsoft.")


@pytest.mark.parametrize("answer", [
    "Gates and Allen established Microsoft on April 4, 1975.",
    "Microsoft was established by Gates and Allen in 1975.",
])
def test_independently_verified_establishment_can_answer_founding_query(answer):
    assert graph._answer_addresses_query("Who founded Microsoft?", answer)


@pytest.mark.parametrize("answer", [
    "Gates established Microsoft AI in 2024.",
    "Gates established Microsoft headquarters in Washington.",
    "Gates established another company and worked at Microsoft.",
    "Snehith is a product manager at Microsoft.",
])
def test_establishment_alias_requires_exact_queried_organization_object(answer):
    assert not graph._answer_addresses_query("Who founded Microsoft?", answer)


@pytest.mark.asyncio
async def test_real_established_wording_is_not_rejected_as_off_topic_after_verification(monkeypatch):
    candidate = "Gates and Allen established Microsoft on April 4, 1975."
    _install(monkeypatch, [candidate])
    state = _state(candidate)
    state["user_query"] = "Who founded Microsoft?"
    result = (await graph._reverifier_node(state))["reverification_result"]
    assert result["passed"] is True
    assert result["failure_category"] is None


def _install(monkeypatch, texts, transform=None, fail=False):
    observed = []
    analysis = claim_analyzer.ClaimAnalysis(
        domain="general", source="llm", candidates=[
            claim_analyzer.ClaimCandidate(str(i), text, text)
            for i, text in enumerate(texts, 1)])

    class Analyzer:
        async def analyze(self, candidate, query, domain):
            observed.append((candidate, query, domain))
            return analysis

    class Pipeline:
        async def verify(self, payload):
            observed.append(payload)
            if fail:
                raise RuntimeError("isolated persistence-free verifier failure")
            reports = [{"claim_id": c.claim_id, "claim_text": c.text,
                        "verdict": "verified", "support_score": 0.9, "evidence": [{"source":"fixture", "snippet":c.text, "entailment_label":"entailment"}]}
                       for c in payload.suspicious_claims]
            if transform:
                reports = transform(reports)
            return {"query_id": payload.query_id, "domain": payload.domain,
                    "claim_evidence": reports, "overall_evidence_confidence": 0.9}

    monkeypatch.setattr(claim_analyzer, "ClaimAnalyzer", Analyzer)
    monkeypatch.setattr(graph, "_get_verifier_imports",
                        lambda: (Pipeline, SimpleNamespace, SimpleNamespace))
    return observed


@pytest.mark.asyncio
async def test_all_corrected_claims_are_submitted_without_five_claim_cap(monkeypatch):
    texts = ["LinkedIn was founded in 2002.", "LinkedIn launched in 2003."] + [
        f"Person {i} is a LinkedIn co-founder." for i in range(1, 7)]
    observed = _install(monkeypatch, texts)
    candidate = " ".join(texts)
    out = await graph._reverifier_node(_state(candidate))
    assert observed[0] == (candidate, "Tell me about LinkedIn.", "general")
    assert [c.text for c in observed[1].suspicious_claims] == texts
    assert len(observed[1].suspicious_claims) == 8
    result = out["reverification_result"]
    assert result["passed"] is True
    assert result["claim_coverage"]["complete"] is True
    assert result["claim_analysis"]["factual_count"] == 8
    assert out["route"] == "judge"  # Final acceptance remains the Judge's decision.


@pytest.mark.asyncio
@pytest.mark.parametrize("transform,key", [
    (lambda rows: rows[:1], "missing_claim_ids"),
    (lambda rows: rows + [rows[0]], "duplicate_claim_ids"),
    (lambda rows: rows + [{**rows[0], "claim_id": "other"}], "unexpected_claim_ids"),
    (lambda rows: [{**rows[0], "claim_text": "An unrelated supported fact."}] + rows[1:],
     "mismatched_claim_ids"),
])
async def test_supported_subset_or_mismatched_reports_cannot_release_candidate(monkeypatch, transform, key):
    texts = ["LinkedIn was founded in 2002.", "LinkedIn launched in 2003."]
    _install(monkeypatch, texts, transform)
    result = (await graph._reverifier_node(_state(" ".join(texts))))["reverification_result"]
    assert result["passed"] is False
    assert result["failure_category"] == "INCOMPLETE_CLAIM_COVERAGE"
    assert result["claim_coverage"]["complete"] is False
    assert result["claim_coverage"][key]


@pytest.mark.asyncio
async def test_unresolved_assertion_remains_unverified_not_accepted(monkeypatch):
    texts = ["These individuals founded the company."]
    _install(monkeypatch, texts, lambda rows: [{**rows[0], "verdict": "unverified"}])
    result = (await graph._reverifier_node(_state(texts[0])))["reverification_result"]
    assert result["passed"] is False
    assert result["claim_analysis"]["candidates"][0]["claim_text"] == texts[0]


@pytest.mark.asyncio
async def test_verifier_exception_is_not_a_successful_reverification(monkeypatch):
    _install(monkeypatch, ["LinkedIn launched in 2003."], fail=True)
    out = await graph._reverifier_node(_state("LinkedIn launched in 2003."))
    assert out["reverification_result"]["passed"] is False
    assert out["reverification_result"]["failure_category"] == "VERIFIER_FAILURE"
    assert graph._reverifier_route(out) == "human_escalation"


@pytest.mark.asyncio
async def test_empty_extraction_fails_closed_without_calling_verifier(monkeypatch):
    observed = _install(monkeypatch, [])
    out = await graph._reverifier_node(_state(""))
    assert len(observed) == 1
    assert out["reverification_result"]["passed"] is False
    assert out["reverification_result"]["failure_category"] == "CLAIM_DECOMPOSITION_FAILED"


@pytest.mark.asyncio
async def test_analyzer_exception_uses_visible_network_free_fallback(monkeypatch):
    observed = _install(monkeypatch, [])

    class BrokenAnalyzer:
        async def analyze(self, *args):
            raise RuntimeError("isolated analyzer failure")

    monkeypatch.setattr(claim_analyzer, "ClaimAnalyzer", BrokenAnalyzer)
    candidate = "LinkedIn was founded in 2002 and launched in 2003. LinkedIn has members."
    out = await graph._reverifier_node(_state(candidate))
    submitted = [c.text for c in observed[0].suspicious_claims]
    assert "LinkedIn was founded in 2002." in submitted
    assert "LinkedIn was launched in 2003." in submitted
    assert out["reverification_result"]["claim_analysis"]["source"] == "fallback"
    assert out["reverification_result"]["passed"] is True


@pytest.mark.asyncio
async def test_nonempty_extraction_cannot_omit_false_second_sentence(monkeypatch):
    first = "LinkedIn was founded in 2002."
    omitted = "LinkedIn launched in 1800."
    observed = _install(monkeypatch, [first], lambda rows: [
        {**row, "verdict": "contradicted" if row["claim_text"] == omitted else "verified"}
        for row in rows])
    out = await graph._reverifier_node(_state(f"{first} {omitted}"))
    assert [c.text for c in observed[1].suspicious_claims] == [first, omitted]
    result = out["reverification_result"]
    assert result["passed"] is False
    assert result["failure_category"] == "REMAINING_CONTRADICTION"
    assert result["source_coverage_backstop"]["added_count"] == 1
    assert result["source_coverage_backstop"]["added_claims"][0]["original_sentence"] == omitted


@pytest.mark.asyncio
async def test_backstop_also_preserves_adjunct_omitted_from_one_sentence(monkeypatch):
    observed = _install(monkeypatch, ["LinkedIn was founded in 2002."])
    candidate = "LinkedIn was founded in 2002 and launched in 1800."
    out = await graph._reverifier_node(_state(candidate))
    texts = [c.text for c in observed[1].suspicious_claims]
    assert "LinkedIn was launched in 1800." in texts
    assert out["reverification_result"]["source_coverage_backstop"]["added_count"] == 1
    assert out["reverification_result"]["claim_coverage"]["requested_count"] == 2
