from pathlib import Path

import pytest

from scripts.trace_pipeline_execution import isolate_runtime


def test_diagnostic_memory_never_uses_existing_store(tmp_path, monkeypatch):
    for key in ("KG_PERSISTENCE_PATH", "CACHE_DB_PATH", "VECTOR_STORE_PATH", "PATTERN_DB_PATH", "TRUST_DB_PATH"):
        monkeypatch.setenv(key, "old-sensitive-store")
    for key in ("CACHE_ENABLED", "VERIFIER_CACHE_ENABLED", "ALWAYS_VERIFY",
                "ALLOW_DETECTOR_FAST_PATH", "ALLOW_MODEL_DOWNLOADS", "MOCK_MODE"):
        monkeypatch.setenv(key, "old")
    store = isolate_runtime(tmp_path / "trace.json")
    import os
    assert store.exists()
    for key in ("KG_PERSISTENCE_PATH", "CACHE_DB_PATH", "VECTOR_STORE_PATH", "PATTERN_DB_PATH", "TRUST_DB_PATH"):
        assert Path(os.environ[key]).parent == store
    assert os.environ["ALWAYS_VERIFY"] == "true"
    assert os.environ["ALLOW_DETECTOR_FAST_PATH"] == "false"
    with pytest.raises(FileExistsError):
        isolate_runtime(tmp_path / "trace.json")
