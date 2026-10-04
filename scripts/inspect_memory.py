"""Inspect saved Memory files without initializing agents, models, or stores.

Uses only the Python standard library. No environment files are loaded. SQLite
is opened read-only and immutable to avoid even WAL/SHM sidecar writes; a
non-empty WAL requires a safely captured, checkpointed snapshot instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
CACHE_COLUMNS = (
    "cache_key, domain, claim_text, verdict, confidence, expires_at, "
    "created_at, source_count, hit_count"
)
MAX_JSON_BYTES = 100 * 1024 * 1024


def _display(value: object, maximum: int = 180) -> str:
    if value is None:
        return "N/A"
    if not isinstance(value, (str, int, float, bool)):
        return "MALFORMED"
    text = " ".join(str(value).split())
    return text if len(text) <= maximum else text[:maximum] + " [display truncated]"


def _confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0 <= number <= 1 else None


def _cache_key(domain: str, text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(f"{domain}:{normalized}".encode("utf-8")).hexdigest()


def _load_json(path: Path, kind: str) -> tuple[object, str, list[str]]:
    if not path.is_file():
        return None, "MISSING", [f"{kind}: file missing; state cannot be verified"]
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            return None, "TOO_LARGE", [f"{kind}: exceeds 100 MiB inspection bound"]
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if kind == "knowledge_graph":
            valid = isinstance(value, dict) and isinstance(value.get("nodes"), list) and isinstance(value.get("edges"), list)
        else:
            valid = isinstance(value, list)
        if not valid:
            return None, "MALFORMED_SCHEMA", [f"{kind}: unexpected saved JSON schema"]
        return value, "READ", []
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        return None, "READ_ERROR", [f"{kind}: {type(exc).__name__}; raw error omitted"]


def _open_database(path: Path) -> sqlite3.Connection:
    # Read-only SQLite can otherwise create/update -shm files. Immutable mode
    # avoids that but ignores WAL, so explicitly refuse uncheckpointed WAL.
    wal = Path(str(path) + "-wal")
    if wal.is_file() and wal.stat().st_size:
        raise RuntimeError("WAL_SNAPSHOT_REQUIRED")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only = ON")
    connection.row_factory = sqlite3.Row
    return connection


def _read_database(path: Path, table: str, limit: int, keys: list[str] | None = None) -> tuple[int | None, list[dict], str, list[str]]:
    # Identifiers below are fixed, never interpolated from user input.
    if table not in {"verification_cache", "patterns", "source_trust"}:
        raise ValueError("unsupported table")
    if not path.is_file():
        return None, [], "MISSING", [f"{table}: file missing; state cannot be verified"]
    connection = None
    try:
        connection = _open_database(path)
        schema = connection.execute("SELECT type FROM sqlite_master WHERE name = ?", (table,)).fetchone()
        if schema is None or schema["type"] != "table":
            return None, [], "MALFORMED_SCHEMA", [f"{table}: expected table missing"]
        count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        rows = []
        if table == "verification_cache":
            if keys is None:
                rows = [dict(row) for row in connection.execute(f"SELECT {CACHE_COLUMNS} FROM verification_cache ORDER BY cache_key LIMIT ?", (limit,))]
            else:
                for key in keys[:limit]:
                    rows.extend(dict(row) for row in connection.execute(f"SELECT {CACHE_COLUMNS} FROM verification_cache WHERE cache_key = ?", (key,)))
        return count, rows, "READ", []
    except RuntimeError:
        return None, [], "WAL_SNAPSHOT_REQUIRED", [f"{table}: non-empty WAL; use a safely captured checkpointed snapshot, not a live writable store"]
    except (sqlite3.Error, OSError, ValueError) as exc:
        return None, [], "READ_ERROR", [f"{table}: {type(exc).__name__}; raw error omitted"]
    finally:
        if connection is not None:
            connection.close()


def _expiry(value: object, now: datetime) -> str:
    if not isinstance(value, str):
        return "INVALID_EXPIRY"
    try:
        expires = datetime.fromisoformat(value)
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return "EXPIRED (retained; not removed)" if expires < now else "ACTIVE"
    except ValueError:
        return "INVALID_EXPIRY"


def inspect_memory(*, runtime_dir: Path | None = None, data_dir: Path | None = None, limit: int = 25,
                   fact_id: str | None = None, show_text: bool = False,
                   now: datetime | None = None) -> tuple[str, int]:
    """Return a local human-readable inspection and 0/1 diagnostic exit status.

    A zero means the displayed sample has no detected warnings, not factual
    verification or distributed consistency. Missing files are not created.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer from 1 through 1000")
    if runtime_dir is not None and data_dir is not None:
        raise ValueError("choose runtime_dir or data_dir, not both")
    base = Path(runtime_dir or data_dir or ROOT / "data").resolve()
    diagnostic = runtime_dir is not None
    paths = {
        "knowledge_graph": base / "knowledge_graph.json",
        "vector_entries": base / "vector_store" / "entries.json",
        "verification_cache": base / ("memory-cache.db" if diagnostic else "verification_cache.db"),
        "patterns": base / "patterns.db",
        "source_trust": base / ("source-trust.db" if diagnostic else "source_trust.db"),
    }
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    lines = ["HALLUCIGUARD MEMORY - READ-ONLY SAVED-FILE INSPECTION", "=" * 72,
             f"Directory: {base}", "Mode: isolated diagnostic runtime" if diagnostic else "Mode: production data directory",
             "No agent/API/model initialization, writes, cache hits, TTL deletion, or .env loading.",
             "Local-only: stored claim text may be private. Text is hidden unless --show-text is used.",
             "This is a bounded saved-file sample, not a transactional or factual verification.",
             "Evidence snippets, citation URLs, request IDs, and embeddings are NOT dumped.", ""]
    graph, graph_status, warnings = _load_json(paths["knowledge_graph"], "knowledge_graph")
    vectors, vector_status, vector_warnings = _load_json(paths["vector_entries"], "vector_entries")
    warnings.extend(vector_warnings)
    nodes, edges, vector_rows = [], [], []
    if graph is not None:
        for node in graph["nodes"]:
            if isinstance(node, dict) and isinstance(node.get("entity_id"), str) and isinstance(node.get("name"), str) and isinstance(node.get("entity_type"), str) and isinstance(node.get("properties"), dict):
                nodes.append(node)
            else:
                warnings.append("knowledge_graph: malformed node omitted; sample cannot establish complete consistency")
        for edge in graph["edges"]:
            if isinstance(edge, dict) and all(isinstance(edge.get(key), str) for key in ("source_id", "target_id", "relation")):
                edges.append(edge)
            else:
                warnings.append("knowledge_graph: malformed edge omitted; links cannot be fully verified")
    if vectors is not None:
        for row in vectors:
            if isinstance(row, dict) and isinstance(row.get("entry_id"), str) and isinstance(row.get("text"), str) and isinstance(row.get("metadata"), dict):
                vector_rows.append(row)
            else:
                warnings.append("vector_entries: malformed record/metadata omitted; sample cannot establish complete consistency")
    facts = [node for node in nodes if node["entity_type"] == "fact"]
    claims = [node for node in nodes if node["entity_type"] == "claim"]
    ids = sorted({node["name"] for node in facts} | {row["entry_id"] for row in vector_rows} |
                 {node["properties"]["fact_id"] for node in claims if isinstance(node["properties"].get("fact_id"), str)})
    if fact_id is not None:
        selected = [fact_id] if fact_id in ids else []
        if not selected:
            warnings.append("requested fact ID not found in readable graph/vector records")
    else:
        selected = ids[:limit]
    # Exact lookups avoid incorrectly calling a fact's cache absent just because
    # its hashed key falls outside the independent bounded cache sample.
    cache_keys = []
    for identity in selected:
        fact = next((node for node in facts if node["name"] == identity), None)
        vector = next((row for row in vector_rows if row["entry_id"] == identity), None)
        properties = fact["properties"] if fact else vector["metadata"] if vector else {}
        text = properties.get("claim_text") if fact else vector["text"] if vector else None
        domain = properties.get("domain")
        if isinstance(text, str) and isinstance(domain, str):
            cache_keys.append(_cache_key(domain, text))
    cache_count, cache_rows, cache_status, cache_warnings = _read_database(paths["verification_cache"], "verification_cache", limit, cache_keys if fact_id else None)
    warnings.extend(cache_warnings)
    if fact_id is None and cache_keys and cache_status == "READ":
        _, relevant, _, relevant_warnings = _read_database(paths["verification_cache"], "verification_cache", limit, cache_keys)
        warnings.extend(relevant_warnings)
        by_key = {row["cache_key"]: row for row in cache_rows}
        by_key.update({row["cache_key"]: row for row in relevant})
        cache_rows = list(by_key.values())
    counts = {"knowledge_graph": (len(nodes), graph_status), "vector_entries": (len(vector_rows), vector_status),
              "verification_cache": (cache_count, cache_status)}
    for table in ("patterns", "source_trust"):
        count, _, status, messages = _read_database(paths[table], table, limit)
        counts[table] = (count, status)
        warnings.extend(messages)
    lines.append("STORES")
    for name, path in paths.items():
        count, status = counts[name]
        lines.extend([f"{name}: {path}", f"  Status: {status}; readable record count: {count if count is not None and status == 'READ' else 'N/A'}"])
    lines.extend([f"Graph facts: {len(facts)}; linked-claim nodes: {len(claims)}; readable edges: {len(edges)}",
                  f"Fact identities: {len(ids)}; displayed: {len(selected)} (limit {limit})", "", "FACT REPRESENTATIONS"])
    for identity in selected:
        lines.append(f"Fact ID: {_display(identity)}")
        related_facts = [node for node in facts if node["name"] == identity]
        related_vectors = [row for row in vector_rows if row["entry_id"] == identity]
        fact_node_ids = {node["entity_id"] for node in related_facts}
        linked_ids = {edge["source_id"] for edge in edges if edge["relation"] == "derived_from" and edge["target_id"] in fact_node_ids}
        related_claims = [node for node in claims if node["properties"].get("fact_id") == identity or node["entity_id"] in linked_ids]
        states: list[tuple[str, object, object]] = []
        if not related_facts:
            warnings.append(f"fact {_display(identity)}: FACT node absent or unreadable")
        if not related_claims:
            warnings.append(f"fact {_display(identity)}: linked CLAIM node absent or unreadable")
        if not related_vectors:
            warnings.append(f"fact {_display(identity)}: vector entry absent or unreadable")
        if len(related_facts) > 1 or len(related_vectors) > 1:
            warnings.append(f"fact {_display(identity)}: duplicate fact/vector identity")
        text_domain = []
        for node in related_facts + related_claims:
            label = "FACT" if node["entity_type"] == "fact" else "CLAIM"
            props = node["properties"]
            lines.append(f"  {label} graph node: {_display(node['entity_id'])}; domain: {_display(props.get('domain'))}; verdict: {_display(props.get('verdict'))}")
            lines.append(f"    Top-level confidence: {_display(node.get('confidence'))}; property confidence: {_display(props.get('confidence'))}")
            states.append((f"{label} top-level", props.get("verdict"), node.get("confidence")))
            if "confidence" in props or label == "CLAIM":
                states.append((f"{label} properties", props.get("verdict"), props.get("confidence")))
            if label == "CLAIM" and props.get("fact_id") != identity:
                warnings.append(f"fact {_display(identity)}: linked CLAIM fact_id missing or conflicting")
            if label == "FACT":
                text_domain.append((props.get("claim_text"), props.get("domain")))
                if show_text:
                    lines.append(f"    Stored claim text: {_display(props.get('claim_text'), 2000)}")
        for row in related_vectors:
            props = row["metadata"]
            lines.append(f"  VECTOR entry: {_display(row['entry_id'])}; domain: {_display(props.get('domain'))}; verdict: {_display(props.get('verdict'))}; confidence: {_display(props.get('confidence'))}")
            states.append(("VECTOR", props.get("verdict"), props.get("confidence")))
            if props.get("fact_id") != identity:
                warnings.append(f"fact {_display(identity)}: vector metadata fact_id missing or conflicting")
            text_domain.append((row["text"], props.get("domain")))
            if show_text:
                lines.append(f"    Stored claim text: {_display(row['text'], 2000)}")
        if len(set((text, domain) for text, domain in text_domain if isinstance(text, str) and isinstance(domain, str))) > 1:
            warnings.append(f"fact {_display(identity)}: graph/vector text or domain mismatch")
        keys = {_cache_key(domain, text) for text, domain in text_domain if isinstance(text, str) and isinstance(domain, str)}
        relevant_cache = [row for row in cache_rows if row["cache_key"] in keys]
        if not relevant_cache:
            warnings.append(f"fact {_display(identity)}: exact cache representation absent or unreadable (expiry/eviction is possible)")
        for row in relevant_cache:
            expiry = _expiry(row["expires_at"], now)
            lines.append(f"  CACHE key: {_display(row['cache_key'])}; verdict: {_display(row['verdict'])}; confidence: {_display(row['confidence'])}; {expiry}")
            states.append(("CACHE", row["verdict"], row["confidence"]))
            if expiry == "INVALID_EXPIRY":
                warnings.append(f"fact {_display(identity)}: invalid cache expiry")
            if not isinstance(row["domain"], str) or not isinstance(row["claim_text"], str) or _cache_key(row["domain"], row["claim_text"]) != row["cache_key"]:
                warnings.append(f"fact {_display(identity)}: cache key does not match saved domain/text")
        valid_states = []
        for label, verdict, confidence in states:
            numeric = _confidence(confidence)
            if not isinstance(verdict, str) or not verdict or numeric is None:
                warnings.append(f"fact {_display(identity)}: {label} verdict/confidence missing or invalid")
            else:
                valid_states.append((verdict, numeric))
        if len(set(valid_states)) > 1:
            warnings.append(f"fact {_display(identity)}: MISMATCH across graph, claim, vector, or cache verdict/confidence")
        lines.append("")
    if fact_id is None:
        lines.extend(["CACHE SAMPLE (saved rows; inspection does not increment hit counters)"])
        for row in cache_rows[:limit]:
            lines.append(f"  {_display(row['cache_key'])}: domain={_display(row['domain'])}; verdict={_display(row['verdict'])}; confidence={_display(row['confidence'])}; {_expiry(row['expires_at'], now)}; stored hit_count={_display(row['hit_count'])}")
            if show_text:
                lines.append(f"    Stored claim text: {_display(row['claim_text'], 2000)}")
    lines.append("WARNINGS / LIMITATIONS")
    # Deduplicate warnings while preserving meaningful diagnostic order.
    warnings = list(dict.fromkeys(warnings))
    lines.extend(f"  - {message}" for message in warnings)
    if not warnings:
        lines.append("  No mismatch detected in the displayed sample. This is not proof of factual truth or whole-store consistency.")
    lines.extend(["  JSON and SQLite are read independently; concurrent updates can produce a mixed-time view.",
                  "  FACT property confidence may be absent on a newly stored fact; top-level confidence is checked.",
                  "  Current store_fact persists source links/IDs where supplied, not full citation passages/URLs or request IDs.",
                  "  Missing stores, malformed records, expired cache, and unavailable readings are never repaired here.",
                  f"Inspection exit status: {1 if warnings else 0}"])
    return "\n".join(lines) + "\n", 1 if warnings else 0


def _bounded_limit(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must be an integer") from exc
    if not 1 <= result <= 1000:
        raise argparse.ArgumentTypeError("limit must be from 1 through 1000")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--runtime-dir", type=Path, help="isolated validation runtime directory")
    location.add_argument("--data-dir", type=Path, help="production data directory; default repository/data; .env overrides are NOT loaded")
    parser.add_argument("--limit", type=_bounded_limit, default=25)
    parser.add_argument("--fact-id", help="exact saved fact UUID")
    parser.add_argument("--show-text", action="store_true", help="display potentially private stored claim text locally")
    args = parser.parse_args(argv)
    try:
        report, status = inspect_memory(**vars(args))
    except (OSError, ValueError) as exc:
        print(f"Memory inspection could not resolve its input: {type(exc).__name__}; raw error omitted")
        return 1
    print(report, end="")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
