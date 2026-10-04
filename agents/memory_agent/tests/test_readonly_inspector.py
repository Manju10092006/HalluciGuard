"""Isolated saved-file fixtures; no Memory agents, APIs, or models are used."""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import datetime, timezone

import pytest

from scripts.inspect_memory import ROOT, _cache_key, inspect_memory, main


NOW = datetime(2026, 10, 4, tzinfo=timezone.utc)
TEXT = "Microsoft was founded by Bill Gates and Paul Allen."


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _create_cache(path, text=TEXT, verdict="verified", confidence=0.8, expiry="2099-01-01T00:00:00"):
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE verification_cache (cache_key TEXT PRIMARY KEY, domain TEXT, claim_text TEXT, verdict TEXT, confidence REAL, expires_at TEXT, created_at TEXT, source_count INTEGER, hit_count INTEGER)")
        connection.execute("INSERT INTO verification_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (_cache_key("general", text), "general", text, verdict, confidence, expiry, "2026-01-01T00:00:00", 1, 7))


@pytest.fixture
def runtime(tmp_path):
    graph = {"nodes": [
        {"entity_id": "fact-node", "entity_type": "fact", "name": "fact-1", "confidence": 0.8,
         "properties": {"domain": "general", "verdict": "verified", "claim_text": TEXT}},
        {"entity_id": "claim-node", "entity_type": "claim", "name": TEXT, "confidence": 0.8,
         "properties": {"domain": "general", "verdict": "verified", "confidence": 0.8, "fact_id": "fact-1"}},
    ], "edges": [{"source_id": "claim-node", "target_id": "fact-node", "relation": "derived_from"}]}
    _write_json(tmp_path / "knowledge_graph.json", graph)
    _write_json(tmp_path / "vector_store" / "entries.json", [{"entry_id": "fact-1", "text": TEXT,
                "metadata": {"fact_id": "fact-1", "domain": "general", "verdict": "verified", "confidence": 0.8,
                             "api_key": "dummy-secret-DO-NOT-DUMP"}, "embedding": [111.123456789]}])
    _create_cache(tmp_path / "memory-cache.db")
    for name, table in (("patterns.db", "patterns"), ("source-trust.db", "source_trust")):
        with closing(sqlite3.connect(tmp_path / name)) as connection, connection:
            connection.execute(f"CREATE TABLE {table} (id INTEGER)")
    return tmp_path


def _hashes(directory):
    return {path.relative_to(directory): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def test_normal_consistent_saved_fact(runtime):
    report, status = inspect_memory(runtime_dir=runtime, now=NOW)
    assert status == 0
    assert "Fact ID: fact-1" in report
    assert "FACT graph node: fact-node" in report
    assert "CLAIM graph node: claim-node" in report
    assert "VECTOR entry: fact-1" in report
    assert "CACHE key:" in report
    assert "No mismatch detected in the displayed sample" in report
    assert TEXT not in report
    assert "dummy-secret-DO-NOT-DUMP" not in report
    assert "111.123456789" not in report


def test_opt_in_text_is_local_display_only(runtime):
    report, status = inspect_memory(runtime_dir=runtime, show_text=True, now=NOW)
    assert status == 0
    assert TEXT in report
    assert "Local-only" in report
    assert "dummy-secret-DO-NOT-DUMP" not in report


def test_stale_claim_and_fact_top_level_quarantine_is_visible(runtime):
    graph_path = runtime / "knowledge_graph.json"
    graph = json.loads(graph_path.read_text())
    graph["nodes"][0]["properties"].update(verdict="unverified", confidence=0)
    _write_json(graph_path, graph)
    vector_path = runtime / "vector_store" / "entries.json"
    vector = json.loads(vector_path.read_text())
    vector[0]["metadata"].update(verdict="unverified", confidence=0)
    _write_json(vector_path, vector)
    with closing(sqlite3.connect(runtime / "memory-cache.db")) as connection, connection:
        connection.execute("UPDATE verification_cache SET verdict='unverified', confidence=0")
    before = _hashes(runtime)
    report, status = inspect_memory(runtime_dir=runtime, now=NOW)
    assert status == 1
    assert "MISMATCH across graph, claim, vector, or cache verdict/confidence" in report
    assert "Top-level confidence: 0.8; property confidence: 0" in report
    assert _hashes(runtime) == before


def test_missing_directory_is_not_created(tmp_path):
    missing = tmp_path / "does-not-exist"
    report, status = inspect_memory(runtime_dir=missing)
    assert status == 1
    assert report.count("Status: MISSING") == 5
    assert not missing.exists()


def test_missing_database_is_not_created(runtime):
    cache = runtime / "memory-cache.db"
    cache.unlink()
    report, status = inspect_memory(runtime_dir=runtime, now=NOW)
    assert status == 1
    assert "verification_cache: file missing" in report
    assert not cache.exists()


@pytest.mark.parametrize("store, payload", [
    ("knowledge_graph.json", {"nodes": {}, "edges": []}),
    ("knowledge_graph.json", {"nodes": [], "edges": "bad"}),
    ("vector_store/entries.json", {"entries": []}),
])
def test_malformed_json_schema_visible(runtime, store, payload):
    _write_json(runtime / store, payload)
    report, status = inspect_memory(runtime_dir=runtime, now=NOW)
    assert status == 1
    assert "MALFORMED_SCHEMA" in report


def test_malformed_json_does_not_echo_sensitive_contents(runtime):
    (runtime / "knowledge_graph.json").write_text('{"dummy-token-do-not-echo": ', encoding="utf-8")
    report, status = inspect_memory(runtime_dir=runtime, now=NOW)
    assert status == 1
    assert "JSONDecodeError" in report
    assert "dummy-token-do-not-echo" not in report


def test_malformed_vector_metadata_cannot_look_consistent(runtime):
    _write_json(runtime / "vector_store" / "entries.json", [{"entry_id": "fact-1", "text": TEXT, "metadata": "bad"}])
    report, status = inspect_memory(runtime_dir=runtime, now=NOW)
    assert status == 1
    assert "malformed record/metadata omitted" in report
    assert "vector entry absent or unreadable" in report


def test_invalid_confidence_does_not_become_zero(runtime):
    graph_path = runtime / "knowledge_graph.json"
    graph = json.loads(graph_path.read_text())
    graph["nodes"][1]["confidence"] = float("nan")
    _write_json(graph_path, graph)
    report, status = inspect_memory(runtime_dir=runtime, now=NOW)
    assert status == 1
    assert "CLAIM top-level verdict/confidence missing or invalid" in report
    assert "Top-level confidence: nan" in report


def test_malformed_sqlite_is_read_error(runtime):
    (runtime / "memory-cache.db").write_bytes(b"dummy-secret-broken-sqlite")
    report, status = inspect_memory(runtime_dir=runtime, now=NOW)
    assert status == 1
    assert "DatabaseError" in report
    assert "dummy-secret-broken-sqlite" not in report


def test_unknown_cache_schema_is_visible(runtime):
    with closing(sqlite3.connect(runtime / "memory-cache.db")) as connection, connection:
        connection.execute("DROP TABLE verification_cache")
        connection.execute("CREATE TABLE other_cache (id INTEGER)")
    report, status = inspect_memory(runtime_dir=runtime, now=NOW)
    assert status == 1
    assert "verification_cache: expected table missing" in report


def test_expired_cache_is_reported_and_not_removed(runtime):
    with closing(sqlite3.connect(runtime / "memory-cache.db")) as connection, connection:
        connection.execute("UPDATE verification_cache SET expires_at='2000-01-01T00:00:00'")
    before = _hashes(runtime)
    report, status = inspect_memory(runtime_dir=runtime, now=NOW)
    assert status == 0
    assert "EXPIRED (retained; not removed)" in report
    assert _hashes(runtime) == before
    with closing(sqlite3.connect(runtime / "memory-cache.db")) as connection, connection:
        assert connection.execute("SELECT hit_count FROM verification_cache").fetchone()[0] == 7


def test_invalid_cache_expiry_is_not_active(runtime):
    with closing(sqlite3.connect(runtime / "memory-cache.db")) as connection, connection:
        connection.execute("UPDATE verification_cache SET expires_at='not-a-date'")
    report, status = inspect_memory(runtime_dir=runtime, now=NOW)
    assert status == 1
    assert "INVALID_EXPIRY" in report


def test_fact_filter_excludes_other_fact_and_cache(runtime):
    graph_path = runtime / "knowledge_graph.json"
    graph = json.loads(graph_path.read_text())
    graph["nodes"].append({"entity_id": "other-node", "entity_type": "fact", "name": "fact-2", "confidence": 0.3,
                           "properties": {"domain": "general", "verdict": "unverified", "claim_text": "Other claim."}})
    _write_json(graph_path, graph)
    report, status = inspect_memory(runtime_dir=runtime, fact_id="fact-1", now=NOW)
    assert status == 0
    assert "Fact ID: fact-1" in report
    assert "Fact ID: fact-2" not in report
    assert "other-node" not in report
    assert "CACHE SAMPLE" not in report


def test_unknown_fact_filter_is_explicit(runtime):
    report, status = inspect_memory(runtime_dir=runtime, fact_id="missing-id", now=NOW)
    assert status == 1
    assert "requested fact ID not found" in report
    assert "Fact ID: fact-1" not in report


@pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5])
def test_inspection_limit_is_bounded(runtime, limit):
    with pytest.raises(ValueError, match="limit"):
        inspect_memory(runtime_dir=runtime, limit=limit)


@pytest.mark.parametrize("limit", ["0", "1001", "bad"])
def test_cli_rejects_invalid_limit(limit):
    with pytest.raises(SystemExit) as exc:
        main(["--limit", limit])
    assert exc.value.code == 2


def test_read_only_database_and_sidecars_are_byte_identical(runtime):
    (runtime / "memory-cache.db-wal").write_bytes(b"")
    (runtime / "memory-cache.db-shm").write_bytes(b"existing-sidecar-not-to-modify")
    before = _hashes(runtime)
    report, status = inspect_memory(runtime_dir=runtime, show_text=True, now=NOW)
    assert status == 0
    assert "stored hit_count=7" in report
    assert _hashes(runtime) == before


def test_nonempty_wal_is_not_silently_ignored(runtime):
    (runtime / "memory-cache.db-wal").write_bytes(b"uncheckpointed-data")
    before = _hashes(runtime)
    report, status = inspect_memory(runtime_dir=runtime, now=NOW)
    assert status == 1
    assert "WAL_SNAPSHOT_REQUIRED" in report
    assert "non-empty WAL" in report
    assert _hashes(runtime) == before


def test_production_names_and_explicit_data_directory(runtime):
    (runtime / "memory-cache.db").rename(runtime / "verification_cache.db")
    (runtime / "source-trust.db").rename(runtime / "source_trust.db")
    report, status = inspect_memory(data_dir=runtime, now=NOW)
    assert status == 0
    assert "Mode: production data directory" in report
    assert "verification_cache.db" in report
    assert "source_trust.db" in report


def test_cli_runs_without_project_agent_or_model_imports(runtime):
    before = _hashes(runtime)
    result = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "inspect_memory.py"),
                             "--runtime-dir", str(runtime), "--show-text"],
                            check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert TEXT in result.stdout
    assert not result.stderr
    assert _hashes(runtime) == before


def test_configuration_locations_are_mutually_exclusive(runtime):
    with pytest.raises(ValueError, match="not both"):
        inspect_memory(runtime_dir=runtime, data_dir=runtime)


def test_default_path_does_not_depend_on_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report, _ = inspect_memory(limit=1)
    assert f"Directory: {(ROOT / 'data').resolve()}" in report
    assert list(tmp_path.iterdir()) == []
