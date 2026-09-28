from orchestration.graph import _build_agent_outcomes


def test_conditional_agents_have_explicit_not_required_outcomes():
    state = {
        "trace": [
            {"node": "base_llm", "status": "completed"},
            {"node": "detector", "status": "completed"},
            {"node": "claim_analyzer", "status": "completed"},
            {"node": "verifier", "status": "completed"},
            {"node": "judge", "status": "completed"},
        ],
        "judge_decision": "ACCEPT",
        "judge_result": {"decision": "ACCEPT"},
    }

    outcomes = _build_agent_outcomes(
        state, memory_result={"status": "skipped", "reason": "no facts"}
    )

    assert set(outcomes) == {
        "base_llm",
        "detector",
        "claim_analyzer",
        "verifier",
        "judge",
        "corrector",
        "reverifier",
        "memory",
    }
    assert outcomes["corrector"]["status"] == "not_required"
    assert outcomes["corrector"]["executed"] is False
    assert outcomes["corrector"]["reason"]
    assert outcomes["reverifier"]["status"] == "not_required"
    assert outcomes["memory"]["executed"] is True


def test_executed_correction_and_reverification_keep_real_results():
    state = {
        "trace": [
            {"node": "corrector", "status": "completed"},
            {"node": "reverifier", "status": "completed"},
        ],
        "correction_result": {"status": "completed", "corrected_text": "truth"},
        "reverification_result": {"status": "completed", "passed": True},
    }

    outcomes = _build_agent_outcomes(state, memory_result={"status": "stored"})

    assert outcomes["corrector"]["executed"] is True
    assert outcomes["corrector"]["result"]["corrected_text"] == "truth"
    assert outcomes["reverifier"]["executed"] is True
    assert outcomes["reverifier"]["result"]["passed"] is True
