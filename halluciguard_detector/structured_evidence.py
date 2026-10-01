"""Bounded, deterministic conversion of retrieval records to model candidates.

This is representation, not fact checking. Missing and null fields never imply
contradiction. Source identifiers are retained in metadata, not invented text.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

MAX_DOCUMENTS = 64
MAX_FIELDS = 64
MAX_CHARACTERS = 8000
_WRAPPERS = ("evidence", "passages", "documents", "context")
_SOURCE_KEYS = ("source_id", "id", "url", "source")
_TEXT_KEYS = ("snippet", "text", "content")


def normalize_evidence(value: Any) -> tuple[list[str], dict[str, Any]]:
    """Return candidate strings and an honest account of representation loss."""
    records: list[str] = []
    sources: list[str | None] = []
    meta: dict[str, Any] = {
        "source_ids": sources, "document_truncated": False,
        "field_truncated": False, "character_truncated": False,
        "null_fields": 0, "malformed_structured": 0,
    }

    def fields(obj: Mapping[str, Any], prefix: str = "", depth: int = 0) -> list[str]:
        if depth >= 8:
            meta["field_truncated"] = True
            return []
        out: list[str] = []
        for key in sorted(obj, key=str):
            if key in _SOURCE_KEYS or key in _WRAPPERS:
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            item = obj[key]
            if item is None:
                meta["null_fields"] += 1
                continue
            if isinstance(item, Mapping):
                out.extend(fields(item, path, depth + 1))
            elif isinstance(item, (list, tuple)):
                for index, child in enumerate(item):
                    if isinstance(child, Mapping):
                        out.extend(fields(child, f"{path}[{index}]", depth + 1))
                    elif child is None:
                        meta["null_fields"] += 1
                    elif isinstance(child, (str, int, float, bool)):
                        out.append(f"{path}[{index}]: {child}")
                    if len(out) >= MAX_FIELDS:
                        meta["field_truncated"] = True
                        return out[:MAX_FIELDS]
            elif isinstance(item, (str, int, float, bool)):
                if key in _TEXT_KEYS and not prefix:
                    out.append(str(item))
                else:
                    out.append(f"{path}: {item}")
            if len(out) >= MAX_FIELDS:
                meta["field_truncated"] = True
                return out[:MAX_FIELDS]
        return out

    def add(text: str, source: str | None) -> None:
        text = text.strip()
        if not text:
            return
        if len(records) >= MAX_DOCUMENTS:
            meta["document_truncated"] = True
            return
        if len(text) > MAX_CHARACTERS:
            meta["character_truncated"] = True
            text = text[:MAX_CHARACTERS]
        records.append(text)
        sources.append(source)

    def visit(item: Any, inherited: str | None = None, inherited_fields: list[str] | None = None, depth: int = 0) -> None:
        inherited_fields = inherited_fields or []
        if depth >= 8:
            meta["field_truncated"] = True
            return
        if item is None:
            return
        if isinstance(item, str):
            stripped = item.strip()
            if stripped.startswith(("{", "[")):
                try:
                    visit(json.loads(stripped), inherited, inherited_fields, depth + 1)
                    return
                except (ValueError, TypeError):
                    meta["malformed_structured"] += 1
            add("; ".join([*inherited_fields, item]), inherited)
        elif isinstance(item, Mapping):
            source = next((str(item[k]) for k in _SOURCE_KEYS if item.get(k) is not None), inherited)
            own = fields(item)
            wrapper = next((key for key in _WRAPPERS if key in item), None)
            if wrapper is not None:
                visit(item[wrapper], source, [*inherited_fields, *own], depth + 1)
            else:
                add("; ".join([*inherited_fields, *own]), source)
        elif isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
            for child in item:
                visit(child, inherited, inherited_fields, depth + 1)
        else:
            meta["malformed_structured"] += 1

    visit(value)
    meta["normalized_count"] = len(records)
    meta["degraded"] = any(meta[key] for key in (
        "document_truncated", "field_truncated", "character_truncated", "malformed_structured"
    ))
    return records, meta
