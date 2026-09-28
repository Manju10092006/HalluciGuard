"""Retrieval health must reflect the complete fallback chain."""

from api.pipeline import append_retrieval_health_stage


def _last_stage(failures: list[str], retrieved: int):
    stages = []
    append_retrieval_health_stage(stages, failures, retrieved)
    return stages[-1] if stages else None


def test_successful_fallback_is_authoritative() -> None:
    stage = _last_stage(["n8n:HTTP 404"], retrieved=8)

    assert stage is not None
    assert stage.status == "completed"
    assert "Fallback retrieval succeeded" in stage.details
    assert "n8n:HTTP 404" in stage.details


def test_all_failed_retrieval_is_failed() -> None:
    stage = _last_stage(["n8n:HTTP 404", "general:timeout"], retrieved=0)

    assert stage is not None
    assert stage.status == "failed"
    assert "All retrieval paths failed" in stage.details


def test_empty_retrieval_without_provider_error_is_degraded() -> None:
    stage = _last_stage([], retrieved=0)

    assert stage is not None
    assert stage.status == "degraded"


def test_healthy_retrieval_adds_no_extra_stage() -> None:
    stages = []
    append_retrieval_health_stage(stages, [], total_retrieved=3)

    assert stages == []
