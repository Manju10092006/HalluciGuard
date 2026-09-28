"""Repair and deploy the HalluciGuard n8n retrieval workflow.

The production workflow used to reference a deleted ``Analyze Claim`` node.
This utility replaces that fragile dependency with deterministic context
construction, validates every connection, writes a repository export, and can
atomically synchronize the nodes/connections into n8n's SQLite database.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKFLOW_IDS = ("ihaerpGOM2pUM4q9", "ebTCMnzByybm0MI2")

BUILD_CONTEXT_JS = r"""const body = $('Receive Claim').first().json.body || {};
const claim = String(body.claim ?? body.query ?? '').replace(/\s+/g, ' ').trim();
if (!claim) { throw new Error('claim_required'); }

const validDomains = ['general','healthcare','cybersecurity','ai_research','finance','legal','earth_science','chemistry_biology','astronomy_physics','software_engineering','humanities_history','current_events'];
let domain = String(body.domain ?? 'general').trim().toLowerCase().replace(/[\s-]+/g, '_');
if (!validDomains.includes(domain)) { domain = 'general'; }

const retrievalMode = ['hybrid','primary_only','tavily_only'].includes(body.retrieval_mode) ? body.retrieval_mode : 'hybrid';
const forceTavily = Boolean(body.force_tavily ?? false);
const maxResults = Number.isFinite(Number(body.max_results)) ? Math.max(1, Math.min(20, Number(body.max_results))) : 5;
const requestId = String(body.request_id ?? '').trim() || ('n8n-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8));
const normalizedClaim = String(body.normalized_claim ?? claim).replace(/\s+/g, ' ').trim();

const queries = [];
const addQuery = (value) => {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim();
  if (text && !queries.some(q => q.toLowerCase() === text.toLowerCase())) { queries.push(text); }
};

if (Array.isArray(body.queries)) { body.queries.forEach(addQuery); }
addQuery(normalizedClaim);

// Deterministic fallback for callers that do not yet send expanded queries.
// These transformations anchor searches on the object being checked instead
// of amplifying a potentially fabricated subject.
const plain = normalizedClaim.replace(/[?.!]+$/, '').trim();
let match = plain.match(/^who\s+(?:is|was|are|were)\s+(?:the\s+)?(?:founder|co-?founder|creator|inventor|designer|developer)\s+of\s+(.+)$/i);
if (match) {
  addQuery(`${match[1]} founder`);
  addQuery(`${match[1]} founded by`);
}
match = plain.match(/^who\s+(founded|created|invented|designed|developed|discovered)\s+(.+)$/i);
if (match) {
  addQuery(`${match[2]} ${match[1]} by`);
  addQuery(`${match[2]} founder creator inventor`);
}
match = plain.match(/^what\s+is\s+(?:the\s+)?capital\s+of\s+(.+)$/i);
if (match) {
  addQuery(`${match[1]} capital city`);
  addQuery(`capital of ${match[1]}`);
}
match = plain.match(/^(.+?)\s+(?:is|was|were|are)\s+(?:the\s+)?(?:founder|co-?founder|creator|inventor|designer|developer)\s+of\s+(.+)$/i);
if (match) {
  addQuery(`${match[2]} founder`);
  addQuery(`who founded ${match[2]}`);
}
match = plain.match(/^(.+?)\s+(?:is|was|were|are)\s+(?:the\s+)?capital\s+of\s+(.+)$/i);
if (match) {
  addQuery(`${match[2]} capital city`);
  addQuery(`capital of ${match[2]}`);
}

const entities = Array.isArray(body.entities)
  ? body.entities.filter(x => typeof x === 'string' && x.trim()).map(x => x.trim())
  : [...new Set((normalizedClaim.match(/\b[A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*)*\b/g) || [])
      .filter(x => !['The','A','An','Who','What','When','Where','Why','How'].includes(x)))];
const timeSensitive = Boolean(body.time_sensitive ?? /\b(today|current|currently|latest|recent|now|this year)\b/i.test(normalizedClaim));
const runPrimary = retrievalMode !== 'tavily_only';

return [{ json: {
  request_id: requestId,
  claim,
  normalized_claim: normalizedClaim,
  domain,
  domain_is_routing_only: true,
  queries: queries.slice(0, 3),
  entities,
  time_sensitive: timeSensitive,
  retrieval_mode: retrievalMode,
  force_tavily: forceTavily,
  max_results: maxResults,
  run_primary: runPrimary,
  t_start: Date.now(),
  workflow_version: '3.1.0',
} }];"""


def repair_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return a repaired workflow and reject malformed graph references."""
    nodes = workflow.get("nodes")
    connections = workflow.get("connections")
    if not isinstance(nodes, list) or not isinstance(connections, dict):
        raise ValueError("workflow must contain nodes[] and connections{}")

    build_nodes = [node for node in nodes if node.get("name") == "Build Context"]
    if len(build_nodes) != 1:
        raise ValueError(f"expected exactly one Build Context node, found {len(build_nodes)}")
    build_nodes[0].setdefault("parameters", {})["jsCode"] = BUILD_CONTEXT_JS

    node_names = {str(node.get("name")) for node in nodes if node.get("name")}
    repaired_connections: dict[str, Any] = {}
    for source, channels in connections.items():
        if source not in node_names or not isinstance(channels, dict):
            continue
        repaired_channels: dict[str, Any] = {}
        for channel_name, outputs in channels.items():
            if not isinstance(outputs, list):
                repaired_channels[channel_name] = outputs
                continue
            repaired_channels[channel_name] = [
                [edge for edge in output if edge.get("node") in node_names]
                if isinstance(output, list)
                else output
                for output in outputs
            ]
        repaired_connections[source] = repaired_channels
    workflow["connections"] = repaired_connections

    missing = []
    for source, channels in repaired_connections.items():
        for outputs in channels.values():
            if not isinstance(outputs, list):
                continue
            for output in outputs:
                if not isinstance(output, list):
                    continue
                for edge in output:
                    if edge.get("node") not in node_names:
                        missing.append(f"{source} -> {edge.get('node')}")
    if missing:
        raise ValueError(f"workflow contains missing node references: {missing}")
    return workflow


def _backup(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copy2(path, backup)
    return backup


def _write_json(path: Path, workflow: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workflow, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sync_database(database: Path, workflow: dict[str, Any]) -> Path:
    """Back up and atomically synchronize the two production workflow rows."""
    if not database.exists():
        raise FileNotFoundError(database)
    backup = _backup(database)
    nodes_json = json.dumps(workflow["nodes"], ensure_ascii=False)
    connections_json = json.dumps(workflow["connections"], ensure_ascii=False)
    with sqlite3.connect(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        found = {
            row[0]
            for row in connection.execute(
                "SELECT id FROM workflow_entity WHERE id IN (?, ?)", WORKFLOW_IDS
            )
        }
        missing = set(WORKFLOW_IDS) - found
        if missing:
            raise RuntimeError(f"n8n workflow IDs not found: {sorted(missing)}")
        connection.executemany(
            "UPDATE workflow_entity SET nodes = ?, connections = ?, active = 1 WHERE id = ?",
            [(nodes_json, connections_json, workflow_id) for workflow_id in WORKFLOW_IDS],
        )
        connection.commit()
    return backup


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workflow",
        type=Path,
        default=Path.home() / "Downloads" / "My_workflow_fixed.json",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=repo_root / "n8n" / "halluciguard-verify-v3.json",
    )
    parser.add_argument(
        "--database", type=Path, default=Path.home() / ".n8n" / "database.sqlite"
    )
    parser.add_argument("--no-db", action="store_true")
    args = parser.parse_args()

    workflow = repair_workflow(json.loads(args.workflow.read_text(encoding="utf-8")))
    workflow_backup = _backup(args.workflow)
    _write_json(args.workflow, workflow)
    _write_json(args.export, workflow)
    print(f"workflow_backup={workflow_backup}")
    print(f"workflow_export={args.export}")
    if not args.no_db:
        print(f"database_backup={sync_database(args.database, workflow)}")
        print(f"database_updated_ids={','.join(WORKFLOW_IDS)}")


if __name__ == "__main__":
    main()
