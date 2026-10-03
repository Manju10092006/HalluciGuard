"""Independent evaluation data: deterministic audits, source components and locks.

No labels are inferred, examples deleted, or evaluation rows rebalanced here.
Hashes attest supplied bytes; complete historical exposure needs an external,
checkpoint-linked provenance declaration, not a freshly randomised split.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import unicodedata

from .data_roles import RoleLabels, require_rows_role

LABELS = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")
ROLE_FILES = {"training": "train.jsonl", "development": "dev.jsonl", "locked_test": "locked_test.jsonl"}
NORMALIZATION = "nfkc-casefold-collapse-unicode-whitespace-v1; punctuation/digits/negation retained"
UNICODE_VERSION = unicodedata.unidata_version
NEAR_METHOD = "exhaustive shared-claim-character-5gram candidates; claim AND evidence set-Jaccard >= threshold; exact pairs reported separately"


class GateBlocked(ValueError):
    """A missing prerequisite or contamination must never become a metric."""


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    if not Path(path).is_file():
        raise GateBlocked(f"missing artifact: {path}")
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def validate_rows(rows: list[dict], *, nonempty=True) -> None:
    if nonempty and not rows:
        raise GateBlocked("empty dataset split")
    for row in rows:
        if not isinstance(row, dict):
            raise GateBlocked("dataset rows must be objects")
        for key in ("id", "source_id", "claim", "evidence"):
            if not isinstance(row.get(key), str) or not normalize(row[key]):
                raise GateBlocked(f"missing/non-string {key} in dataset row")
        for key in ("id", "source_id", "response_id", "document_id"):
            if key in row and (not isinstance(row[key], str) or not row[key] or row[key] != row[key].strip()):
                raise GateBlocked(f"invalid or padded provenance identifier: {key}")
        if row.get("label") not in LABELS or type(row.get("label_id")) is not int or row["label_id"] != LABELS.index(row["label"]):
            raise GateBlocked("label/count integrity violation: preserve the exact three-class mapping")
        response_id(row)  # Require response provenance; never invent a response ID.


def response_id(row: dict) -> str:
    if isinstance(row.get("response_id"), str) and row["response_id"].strip():
        return row["response_id"]
    # RAGTruth converter writes raw response ID followed by real span offsets.
    match = re.fullmatch(r"(.+):\d+-\d+(?:#\d+)?", row["id"])
    if match:
        return match.group(1)
    raise GateBlocked("missing response_id (only documented RAGTruth span IDs can derive it)")


def read_rows(path: Path) -> list[dict]:
    file_hash(path)
    try:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (ValueError, UnicodeError) as exc:
        raise GateBlocked("invalid JSONL artifact") from exc
    validate_rows(rows)
    return rows


def _keys(row):
    claim, evidence = normalize(row["claim"]), normalize(row["evidence"])
    return {"row_id": row["id"], "claim": claim, "evidence": evidence,
            "pair": canonical([claim, evidence]).decode(), "source_group": row["source_id"],
            "document_group": str(row.get("document_id") or row["source_id"]),
            "response_group": response_id(row),
            "source_response_group": canonical([row["source_id"], response_id(row)]).decode()}


def _shingles(text: str) -> frozenset[str]:
    text = normalize(text)
    return frozenset(text[i:i + 5] for i in range(max(1, len(text) - 4)))


def _similarity(left, right):
    return len(left & right) / len(left | right) if left | right else 1.0


def audit_splits(splits: dict[str, list[dict]], *, near_threshold=0.85, max_candidates=10_000_000) -> dict:
    """Complete exact/near audit. Budget exhaustion blocks, never reports zero."""
    if not math.isfinite(near_threshold) or not 0 < near_threshold <= 1 or max_candidates < 1:
        raise ValueError("invalid near-duplicate threshold/budget")
    records, indexes = [], {k: defaultdict(list) for k in _keys({"id": "x", "source_id": "s", "claim": "c", "evidence": "e", "response_id": "r"})}
    for role, rows in sorted(splits.items()):
        validate_rows(rows)
        for row in sorted(rows, key=lambda r: (r["id"], digest(r))):
            ref = {"role": role, "id": row["id"], "ordinal": len(records)}
            records.append((ref, row))
            for key, value in _keys(row).items():
                indexes[key][value].append(ref)
    duplicates = {}
    for key, index in indexes.items():
        groups = [{"key_sha256": digest(value), "rows": refs, "cross_split": len({r['role'] for r in refs}) > 1}
                  for value, refs in sorted(index.items()) if len(refs) > 1]
        duplicates[key] = {"groups": groups, "duplicate_groups": len(groups),
                           "extra_rows": sum(len(g["rows"]) - 1 for g in groups),
                           "cross_split_groups": sum(g["cross_split"] for g in groups)}
    candidates, claim_sets, evidence_sets, inverted = 0, [], [], defaultdict(list)
    near = []
    for i, (ref, row) in enumerate(records):
        c, e = _shingles(row["claim"]), _shingles(row["evidence"])
        prior = set(itertools.chain.from_iterable(inverted[s] for s in c))
        candidates += len(prior)
        if candidates > max_candidates:
            raise GateBlocked("near-duplicate candidate budget exhausted: audit incomplete; increase budget explicitly")
        for j in sorted(prior):
            other_ref, other = records[j]
            if normalize(row["claim"]) == normalize(other["claim"]) and normalize(row["evidence"]) == normalize(other["evidence"]):
                continue
            cs = _similarity(c, claim_sets[j])
            if cs < near_threshold:
                continue
            es = _similarity(e, evidence_sets[j])
            if es >= near_threshold:
                near.append({"left": other_ref, "right": ref, "claim_similarity": cs, "evidence_similarity": es,
                             "cross_split": other_ref["role"] != ref["role"]})
        claim_sets.append(c)
        evidence_sets.append(e)
        for s in c:
            inverted[s].append(i)
    stats = {}
    for role, rows in sorted(splits.items()):
        refs = [ref for ref, _ in records if ref["role"] == role]
        stats[role] = {"total_rows": len(rows), "labels": {label: sum(r['label'] == label for r in rows) for label in LABELS},
                       "unique_source_groups": len({r['source_id'] for r in rows}),
                       "unique_document_groups": len({_keys(r)['document_group'] for r in rows}),
                       "document_id_fallback": "source_id when explicit document_id is absent",
                       "unique_response_groups": len({response_id(r) for r in rows}),
                       "unique_source_response_groups": len({_keys(r)['source_response_group'] for r in rows}),
                       "exact_duplicate_extra_rows": {key: sum(max(0, sum(ref['role'] == role for ref in g['rows']) - 1) for g in value['groups']) for key, value in duplicates.items()},
                       "near_duplicate_pairs_within_split": sum(p['left']['role'] == role == p['right']['role'] for p in near),
                       "exclusions": [], "exclusion_count": 0, "row_ids": sorted(ref['id'] for ref in refs)}
    return {"normalization": NORMALIZATION, "splits": stats, "exact": duplicates,
            "near": {"method": NEAR_METHOD, "threshold": near_threshold, "candidate_pairs_checked": candidates,
                     "pair_count": len(near), "cross_split_pair_count": sum(p['cross_split'] for p in near), "pairs": near, "complete": True},
            "cross_split_contamination": any(v['cross_split_groups'] for v in duplicates.values()) or any(p['cross_split'] for p in near)}


def require_isolation(audit: dict) -> None:
    if audit["cross_split_contamination"]:
        raise GateBlocked("train/dev/locked-test overlap: exact/group/near-duplicate contamination")
    if any(len(s["row_ids"]) != len(set(s["row_ids"])) for s in audit["splits"].values()):
        raise GateBlocked("duplicate row IDs are not stable split identities")


def grouped_split(rows: list[dict], *, seed=42, fractions=(0.8, 0.1, 0.1), near_threshold=0.85, max_candidates=10_000_000) -> dict:
    """Split connected source/response/document/content components, not random rows."""
    validate_rows(rows)
    if len(fractions) != 3 or any(not math.isfinite(f) or f <= 0 for f in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError("three positive fractions summing to one are required")
    rows = sorted(rows, key=lambda r: r["id"])
    if len({r['id'] for r in rows}) != len(rows):
        raise GateBlocked("duplicate row IDs")
    audit = audit_splits({"pool": rows}, near_threshold=near_threshold, max_candidates=max_candidates)
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(b)] = find(a)
    for result in audit['exact'].values():
        for group in result['groups']:
            ordinals = [r['ordinal'] for r in group['rows']]
            for i in ordinals[1:]:
                union(ordinals[0], i)
    for pair in audit['near']['pairs']:
        union(pair['left']['ordinal'], pair['right']['ordinal'])
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append(row)
    blocks = sorted(components.values(), key=lambda b: digest([seed, sorted(r['id'] for r in b)]))
    if len(blocks) < 3:
        raise GateBlocked("fewer than three independent source/content components; do not force a split")
    ntrain = min(len(blocks) - 2, max(1, int(len(blocks) * fractions[0])))
    ndev = min(len(blocks) - ntrain - 1, max(1, int(len(blocks) * fractions[1])))
    ends = (blocks[:ntrain], blocks[ntrain:ntrain + ndev], blocks[ntrain + ndev:])
    return {role: sorted([r for b in subset for r in b], key=lambda r: r['id']) for role, subset in zip(ROLE_FILES, ends)}


def _write_new(path: Path, value) -> None:
    with path.open("xb") as f:
        f.write(canonical(value) + b"\n")


def lock_splits(splits: dict, output: Path, *, dataset_name: str, dataset_version: str, git_sha: str,
                created_at: str | None = None, seed: int | None = None, source_artifacts=(), exposure_record: Path | None = None,
                near_threshold=0.85, max_candidates=10_000_000, provenance_kind="real") -> str:
    """Lock all roles before evaluation. Never overwrite a previous bundle."""
    if set(splits) != set(ROLE_FILES) or not dataset_name or not dataset_version or not git_sha:
        raise GateBlocked("all three explicit roles and dataset/version/git identity are required")
    for role, rows in splits.items():
        require_rows_role(rows, role)
    audit = audit_splits(splits, near_threshold=near_threshold, max_candidates=max_candidates)
    require_isolation(audit)
    sources = [{"path": str(Path(p).resolve()), "sha256": file_hash(Path(p))} for p in source_artifacts]
    exposure = None
    if exposure_record is not None:
        exposure = {"path": str(Path(exposure_record).resolve()), "sha256": file_hash(Path(exposure_record))}
    if provenance_kind not in ("real", "synthetic_fixture"):
        raise ValueError("unknown provenance kind")
    if Path(output).exists():
        raise FileExistsError("immutable output bundle already exists")
    Path(output).mkdir(parents=True)
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    manifests = {}
    for role, filename in ROLE_FILES.items():
        rows = [{**r, "data_role": role} for r in sorted(splits[role], key=lambda r: r['id'])]
        with (Path(output) / filename).open("xb") as f:
            for row in rows:
                f.write(canonical(row) + b"\n")
        m = {"schema_version": 1, "dataset_name": dataset_name, "dataset_version": dataset_version,
             "created_at": timestamp, "git_sha": git_sha, "role": role, "locked": role == "locked_test",
             "file": filename, "file_sha256": file_hash(Path(output) / filename),
             "row_ids": [r['id'] for r in rows], "source_ids": sorted({r['source_id'] for r in rows}),
             "response_ids": sorted({response_id(r) for r in rows}), "row_content_sha256": digest(rows),
             "label_counts": audit['splits'][role]['labels'], "summary": audit['splits'][role],
             "source_artifacts": sources, "normalization": NORMALIZATION, "split_seed": seed,
             "normalization_unicode_version": UNICODE_VERSION,
             "provenance_kind": provenance_kind, "contamination_summary": {"cross_split_contamination": False, "near_threshold": near_threshold}}
        name = role + ".manifest.json"
        _write_new(Path(output) / name, m)
        manifests[role] = {"path": name, "sha256": file_hash(Path(output) / name)}
    bundle = {"schema_version": 1, "created_at": timestamp, "dataset_name": dataset_name,
              "dataset_version": dataset_version, "git_sha": git_sha, "provenance_kind": provenance_kind,
              "split_manifests": manifests, "audit": audit, "exposure_record": exposure,
              "independence": "requires complete checkpoint-linked historical training/calibration audit",
              "exclusions": [], "exclusion_count": 0, "normalization": NORMALIZATION}
    bundle['normalization_unicode_version'] = UNICODE_VERSION
    bundle['near_candidate_budget'] = max_candidates
    _write_new(Path(output) / "gate-manifest.json", bundle)
    return file_hash(Path(output) / "gate-manifest.json")


def verify_bundle(output: Path, expected_hash: str) -> tuple[dict, dict]:
    """Caller pins the external manifest hash; a self-reported hash is insufficient."""
    root = Path(output)
    if not expected_hash or file_hash(root / "gate-manifest.json") != expected_hash:
        raise GateBlocked("locked bundle manifest hash mismatch")
    bundle = json.loads((root / "gate-manifest.json").read_text(encoding="utf-8"))
    if bundle.get('normalization') != NORMALIZATION or bundle.get('normalization_unicode_version') != UNICODE_VERSION or set(bundle.get('split_manifests', {})) != set(ROLE_FILES):
        raise GateBlocked("unsupported manifest schema/roles/normalization")
    splits = {}
    for role, filename in ROLE_FILES.items():
        ref = bundle['split_manifests'][role]
        if ref['path'] != role + '.manifest.json' or file_hash(root / ref['path']) != ref['sha256']:
            raise GateBlocked("split manifest mutation")
        manifest = json.loads((root / ref['path']).read_text(encoding="utf-8"))
        if manifest.get('role') != role or manifest.get('locked') != (role == 'locked_test') or manifest.get('file') != filename:
            raise GateBlocked("split role/file mutation")
        if file_hash(root / filename) != manifest['file_sha256']:
            raise GateBlocked("locked dataset artifact mutation")
        rows = read_rows(root / filename)
        if any(r.get('data_role') != role for r in rows) or [r['id'] for r in rows] != manifest['row_ids'] or digest(rows) != manifest['row_content_sha256']:
            raise GateBlocked("locked row/role/identity mutation")
        if {label: sum(r['label'] == label for r in rows) for label in LABELS} != manifest['label_counts']:
            raise GateBlocked("locked label count mutation")
        splits[role] = rows
    actual = audit_splits(splits, near_threshold=bundle['audit']['near']['threshold'],
                          max_candidates=bundle['near_candidate_budget'])
    require_isolation(actual)
    if actual != bundle['audit']:
        raise GateBlocked("contamination summary mutation")
    return bundle, splits


def verify_checkpoint_exposure(bundle: dict, locked_rows: list[dict], checkpoint_sha256: str) -> dict:
    ref = bundle.get('exposure_record')
    if ref is None:
        raise GateBlocked("complete historical training/calibration exposure record missing for frozen checkpoint")
    if file_hash(Path(ref['path'])) != ref['sha256']:
        raise GateBlocked("historical exposure declaration mutation")
    record = json.loads(Path(ref['path']).read_text(encoding='utf-8'))
    if record.get('checkpoint_sha256') != checkpoint_sha256 or record.get('coverage') != 'complete' or not record.get('attested_by'):
        raise GateBlocked("checkpoint-linked complete exposure attestation required")
    historical = {}
    for role in ('training', 'development'):
        refs = record.get(role + '_artifacts', [])
        if not refs:
            raise GateBlocked("historical exposure artifact references missing")
        rows = []
        for artifact in refs:
            if file_hash(Path(artifact['path'])) != artifact['sha256']:
                raise GateBlocked("historical exposure artifact mutation")
            rows.extend(read_rows(Path(artifact['path'])))
        historical[role] = rows
    audit = audit_splits({**historical, 'locked_test': locked_rows}, near_threshold=bundle['audit']['near']['threshold'],
                         max_candidates=bundle['near_candidate_budget'])
    # Historical train/dev leakage also invalidates independent calibration.
    crosses = any(g['cross_split'] and any(r['role'] == 'locked_test' for r in g['rows']) for v in audit['exact'].values() for g in v['groups'])
    crosses |= any(p['cross_split'] and 'locked_test' in (p['left']['role'], p['right']['role']) for p in audit['near']['pairs'])
    if crosses:
        raise GateBlocked("frozen checkpoint exposure overlaps locked test")
    if audit['cross_split_contamination']:
        raise GateBlocked("historical training/development contamination invalidates independent calibration")
    return {"declaration_sha256": ref['sha256'], "historical_audit": audit,
            "qualification": "complete historical membership is a supplied provenance attestation; hashes cannot independently prove completeness"}


def labels_for_role(root: Path, expected_hash: str, role: str) -> RoleLabels:
    _, splits = verify_bundle(root, expected_hash)
    if role not in ROLE_FILES:
        raise GateBlocked("unknown split role")
    return RoleLabels([r['label_id'] for r in splits[role]], role)


def verify_training_directory(root: Path) -> None:
    """Standard training cannot rename/copy a managed locked split into dev."""
    path = Path(root) / 'gate-manifest.json'
    if path.exists():
        _, splits = verify_bundle(root, file_hash(path))
        require_rows_role(splits['training'], 'training')
        require_rows_role(splits['development'], 'development')
