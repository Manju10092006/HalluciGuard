"""Offline source/score presentation tests; no production inference or retrieval."""
from copy import deepcopy
import socket

import pytest

from scripts import run_seven_agent_walkthrough as walkthrough


CLAIM = "LinkedIn was founded in 2002."


def passage(title="LinkedIn", snippet="LinkedIn was founded in December 2002.", **extra):
    return {
        "source": "wikipedia", "source_id": "PAGEID:970755", "title": title,
        "url": "https://en.wikipedia.org/wiki/LinkedIn", "snippet": snippet,
        "relevance_score": 0.0, **extra,
    }


def component(name, inp, out, **extra):
    return {"kind": "component", "name": name, "node": "verifier", "status": "RETURNED",
            "seconds": 0.01, "input": inp, "output": out, **extra}


def recording(*observations, report=None):
    report = report or {"claim_id": "c1", "claim_text": CLAIM, "verdict": "unverified",
                        "explanation": "No decision-grade passage.", "evidence": [],
                        "retrieved_documents": 5, "reranked_documents": 2,
                        "retrieval_trace": {"generated_queries": []}}
    events = [{"kind": "node", "name": "verifier", "status": "RETURNED", "seconds": 0.1,
               "input": {"state": {}}, "output": {"verifier": {"claim_evidence": [report]}}},
              *observations,
              {"kind": "node", "name": "judge", "status": "RETURNED", "seconds": 0.01,
               "input": {"state": {}}, "output": {"judge": {"decision": "ABSTAIN"}}}]
    for index, event in enumerate(events, 1):
        event["sequence"] = index
    return {"query": "Who founded LinkedIn?", "draft": None, "events": events,
            "runner_status": "completed", "seconds": 0.11,
            "result": {"draft_response": CLAIM, "final_response": "Verification incomplete.",
                       "judge_decision": "ABSTAIN", "terminal_status": "human_review",
                       "verification_status": "human_review_required", "trace": []}}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("Source-score formatting must not make a network call")
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)


def test_retrieved_links_are_shown_even_without_admitted_evidence():
    doc = passage()
    run = recording(component("N8NRetrievalClient.retrieve_evidence",
                              {"claim": CLAIM.rstrip("."), "queries": ["LinkedIn founder 2002"]},
                              {"success": True, "passages": [doc]}))
    rendered = walkthrough.format_readable_walkthrough(run)
    assert "https://en.wikipedia.org/wiki/LinkedIn" in rendered
    assert "PAGEID:970755" in rendered
    assert "LinkedIn founder 2002" in rendered
    assert "[c1]" in rendered
    assert "None — no passage passed the decision gate." in rendered
    assert "retrieval alone is not a verdict" in rendered


def test_repair_and_reverification_show_actual_claims_sources_and_coverage():
    run = recording()
    evidence = {"snippet": "Company was founded by Alice.",
                "url": "https://example.test/history"}
    run["events"].extend([
        {"sequence": 3, "kind": "node", "name": "corrector", "status": "RETURNED", "seconds": 0.1,
         "input": {"state": {"correction_request": {
             "claims_to_correct": [{"claim_text": "Bob founded Company."}],
             "evidence": [evidence]}}},
         "output": {"correction_result": {"status": "completed", "corrected_text": "Alice founded Company."}}},
        {"sequence": 4, "kind": "node", "name": "reverifier", "status": "RETURNED", "seconds": 0.1,
         "input": {"state": {}}, "output": {"reverification_result": {
             "status": "completed", "passed": True,
             "claim_coverage": {"complete": True, "requested_count": 1, "reported_count": 1},
             "verifier_result": {"claim_reports": [{"claim_id": "rev-1", "claim_text": "Alice founded Company.",
                                   "verdict": "verified", "evidence": [evidence]}]}}}},
    ])
    rendered = walkthrough.format_readable_walkthrough(run)
    assert "Claim requiring repair" in rendered and "Bob founded Company." in rendered
    assert "Evidence supplied for repair" in rendered
    assert "[rev-1] Alice founded Company." in rendered
    assert "1 reports for 1 extracted claims" in rendered
    assert "Re-verification source" in rendered and evidence["url"] in rendered


def test_per_agent_exports_do_not_mislabel_a_normal_query_as_dell(tmp_path):
    run = recording()
    run.update({"case": "QUERY_RUN", "mode": "real_generation", "git": {}, "started_utc": "fixture"})
    for event in run["events"]:
        event.setdefault("node", event["name"] if event["kind"] == "node" else "verifier")
    walkthrough.validation.export_reports(tmp_path, [run])
    assert (tmp_path / "01_ALL_7_AGENTS_END_TO_END.txt").exists()
    assert (tmp_path / "04_VERIFIER_AGENT.txt").exists()
    assert (tmp_path / "10_NORMAL_LIVE_QUERY.txt").exists()
    assert not (tmp_path / "10_DELL_G15_LIVE_QUERY.txt").exists()
    assert "Who founded LinkedIn?" in (tmp_path / "SUMMARY.txt").read_text(encoding="utf-8")
    verifier_txt = (tmp_path / "04_VERIFIER_AGENT.txt").read_text(encoding="utf-8")
    assert "PURPOSE:" in verifier_txt and "STATUS:" in verifier_txt
    assert "3. [AGENT] VERIFIER AGENT" in verifier_txt
    assert "raw/QUERY_RUN_execution.json" in verifier_txt
    assert "4. [AGENT] JUDGE AGENT" not in verifier_txt
    assert '"verifier": {' not in verifier_txt


def test_standalone_validation_case_labels_match_the_supplied_mode():
    assert walkthrough.validation.requested_cases("Who founded LinkedIn?", None) == [
        ("Who founded LinkedIn?", None, "QUERY_RUN")]
    assert walkthrough.validation.requested_cases("Who created Rust?", "An incorrect creator.") == [
        ("Who created Rust?", "An incorrect creator.", "SUPPLIED_DRAFT_RUN")]
    cases = walkthrough.validation.requested_cases(None, None, suite=True)
    assert [case[2] for case in cases] == ["TEST_A_DELL_G15", "TEST_B_MICROSOFT"]
    assert cases[0][1] is None
    assert cases[1][1] == "Snehith is the founder of Microsoft."


def test_actual_bge_model_rank_and_neural_scores_are_separate():
    doc = passage(relevance_score=0.281764239)
    run = recording(
        component("CrossEncoderReranker.rerank", {"claim": CLAIM.rstrip(".")}, [doc],
                  diagnostics={"model": "BAAI/bge-reranker-large", "inference_executed": True,
                               "status": "executed", "degraded": False}),
        component("NLIEngine.batch_classify", {"claim": CLAIM.rstrip("."), "evidences": [doc["snippet"]]},
                  [{"label": "neutral", "entailment_score": 0.002624,
                    "contradiction_score": 0.000052, "neutral_score": 0.997324}],
                  diagnostics={"model": "cross-encoder/nli-deberta-v3-base", "inference_executed": True,
                               "status": "executed", "degraded": False}),
    )
    rendered = walkthrough.format_readable_walkthrough(run)
    assert "BAAI/bge-reranker-large" in rendered
    assert "BGE relevance score" in rendered and "0.2818" in rendered
    assert "cross-encoder/nli-deberta-v3-base" in rendered
    assert "Neural neutral" in rendered and "0.9973" in rendered
    assert "observed NLI output, not the final verdict" in rendered
    assert "token survival is not exposed" in rendered
    assert "NLI evidence link" in rendered


def test_different_claim_does_not_receive_another_claims_sources_or_scores():
    run = recording(component("CrossEncoderReranker.rerank", {"claim": "Microsoft was founded in 1975"},
                              [passage(title="Wrong claim source", url="https://example.invalid/wrong-claim")],
                              diagnostics={"inference_executed": True}))
    rendered = walkthrough.format_readable_walkthrough(run)
    assert "https://example.invalid/wrong-claim" not in rendered
    assert "Wrong claim source" not in rendered
    assert "BGE relevance score" not in rendered


def test_explicit_claim_id_mismatch_prevents_attribution():
    report = {"claim_id": "c1", "claim_text": CLAIM}
    observation = component("CrossEncoderReranker.rerank", {"claim_id": "c2", "claim": CLAIM}, [])
    assert walkthrough.claim_component_events(report, [observation]) == []


def test_exported_decision_score_is_not_called_a_neural_probability():
    doc = passage(bge_score=0.92, nli_entailment=0.95, nli_contradiction=0.01,
                  nli_neutral=0.82, entailment_label="entailment", classification="SUPPORTING")
    report = {"claim_id": "c1", "claim_text": CLAIM, "verdict": "verified", "evidence": [doc]}
    rendered = walkthrough.format_readable_walkthrough(recording(report=report))
    assert "reported entailment score=0.9500" in rendered
    assert "Exported decision scores may include relation-rule adjustments" in rendered
    assert "Neural entailment" not in rendered
    assert "https://en.wikipedia.org/wiki/LinkedIn" in rendered
    assert "Reported BGE score" in rendered


def test_backend_class_name_is_not_misreported_as_dense_execution():
    run = recording(component("HybridRetriever.retrieve", {"query": CLAIM.rstrip(".")}, [passage()]))
    rendered = walkthrough.format_readable_walkthrough(run)
    assert "HybridRetriever class name is not proof of dense/fusion execution" in rendered
    assert "Actual retrieval route" not in rendered
    assert "Dense retriever execution" not in rendered


def test_degraded_nli_does_not_claim_neural_execution_or_probability():
    run = recording(component("NLIEngine.batch_classify", {"claim": CLAIM, "evidences": ["Evidence"]},
                              [{"label": "neutral", "entailment_score": 0.0,
                                "contradiction_score": None, "neutral_score": 1.0}],
                              diagnostics={"inference_executed": False, "degraded": True}))
    rendered = walkthrough.format_readable_walkthrough(run)
    assert "No inference executed; degraded fallback" in rendered
    assert "Observed NLI entailment" in rendered and "0.0000" in rendered
    assert "Observed NLI contradiction" not in rendered
    assert "Neural entailment" not in rendered


def test_rejection_reason_is_taken_from_actual_trace_without_mutating_input():
    report = {"claim_id": "c1", "claim_text": CLAIM, "verdict": "unverified", "evidence": [],
              "retrieval_trace": {"generated_queries": [], "evidence_flow": {"nli_outputs": [
                  {"selected_for_decision": False, "source_id": "source-7", "rejection_reason": "RELATION_NOT_GROUNDED"}]}}}
    run = recording(component("N8NRetrievalClient.retrieve_evidence", {"claim": CLAIM, "queries": ["founder"]},
                              {"passages": []}), report=report)
    before = deepcopy(run)
    rendered = walkthrough.format_readable_walkthrough(run)
    assert "RELATION_NOT_GROUNDED" in rendered
    assert "source-7" in rendered
    assert run == before


def test_missing_source_link_is_reported_not_synthesized():
    run = recording(component("N8NRetrievalClient.retrieve_evidence", {"claim": CLAIM},
                              {"passages": [passage(url=None)]}))
    rendered = walkthrough.format_readable_walkthrough(run)
    assert "Source link" in rendered and "Not exposed by this record" in rendered
    assert "https://" not in rendered


def test_empty_n8n_is_not_reported_as_empty_merged_retrieval():
    report = {"claim_id": "c1", "claim_text": CLAIM, "evidence": [], "verdict": "unverified",
              "retrieval_trace": {"subclaim_execution": [{"parent_claim_id": "c1",
                  "counts": {"retrieved": 17, "relevance_selected": 8, "nli_selected": 3}}]}}
    run = recording(component("N8NRetrievalClient.retrieve_evidence", {"claim": CLAIM},
                              {"success": True, "passages": []}), report=report)
    rendered = walkthrough.format_readable_walkthrough(run)
    assert "n8n candidates for this claim" in rendered and "0 passage(s)" in rendered
    assert "Merged retrieval candidates" in rendered and "17" in rendered
    assert "Relevance-selected candidates" in rendered and "NLI-selected passages" in rendered
    assert "this is not the merged retrieval total" in rendered


def test_small_measured_scores_are_distinct_from_zero_and_unavailable():
    assert walkthrough.number(0.00000083) == "8.3e-07"
    assert walkthrough.number(0.0) == "0.0000"
    assert walkthrough.number(None) == "N/A"
    assert walkthrough.number(False) == "N/A"


@pytest.mark.parametrize("status,degraded,expected", [
    ("degraded", True, "FAILED"), ("unavailable", False, "FAILED"),
    ("completed", False, "FIRED"),
])
def test_detector_failure_is_not_counted_as_healthy_triage(status, degraded, expected):
    run = recording()
    run["events"].insert(0, {"kind": "node", "name": "detector", "node": "detector",
        "sequence": 0, "status": "RETURNED", "seconds": 0.02, "input": {"state": {}},
        "output": {"detector_result": {"status": status, "detector_degraded": degraded,
            "grounded": False, "inference_executed": False, "probability_available": False},
            "trace": [{"node": "detector", "status": "completed"}]}})
    assert walkthrough.validation.agent_status(run, "DETECTOR", ("detector", "grounded_detector")) == expected
    rendered = walkthrough.format_readable_walkthrough(run)
    if expected == "FAILED":
        assert "Detector unavailable or failed" in rendered
        assert "Evidence-free triage only" not in rendered
        assert "Failed agents" in rendered and "DETECTOR" in rendered
    else:
        assert "Evidence-free triage only" in rendered


@pytest.mark.parametrize("parent_claim_id,should_show", [("c1", True), ("c2", False)])
def test_object_anchored_ranking_uses_explicit_parent_binding(parent_claim_id, should_show):
    doc = passage(title="Bound ranked source", relevance_score=0.7321)
    report = {"claim_id": "c1", "claim_text": CLAIM, "verdict": "unverified", "evidence": [],
              "retrieval_trace": {"subclaim_execution": [{
                  "parent_claim_id": parent_claim_id,
                  "claim_text": CLAIM,
                  "retrieval_query": "LinkedIn founded by",
                  "backend_execution": {"route": "hybrid_rrf"},
                  "reranker_execution": {"model": "BAAI/bge-reranker-large",
                      "inference_executed": True, "status": "executed"},
                  "reranked_passages": [doc],
              }]}}
    rendered = walkthrough.format_readable_walkthrough(recording(
        component("CrossEncoderReranker.rerank", {"claim": "LinkedIn founded by"}, [doc]),
        report=report))
    assert ("Bound ranked source" in rendered) is should_show
    assert ("0.7321" in rendered) is should_show
    assert ("Ranking query" in rendered) is should_show
