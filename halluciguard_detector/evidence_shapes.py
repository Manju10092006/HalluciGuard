"""Bounded, deterministic evidence normalization for grounded Detector inputs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedEvidence:
    documents: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    document_sources: list[str | None] = field(default_factory=list)
    null_fields: int = 0
    malformed_records: int = 0
    truncated_records: int = 0
    character_truncations: int = 0
    field_truncations: int = 0
    document_truncations: int = 0
    depth_truncations: int = 0

    def diagnostics(self) -> dict[str, Any]:
        return {
            "route": "lexical_only",
            "dense_executed": False,
            "degraded": self.malformed_records > 0 or self.truncated_records > 0,
            "normalization": {
                "documents": len(self.documents), "source_ids": self.source_ids,
                "document_sources": self.document_sources,
                "null_fields": self.null_fields,
                "malformed_records": self.malformed_records,
                "truncated_records": self.truncated_records,
                "character_truncations": self.character_truncations,
                "field_truncations": self.field_truncations,
                "document_truncations": self.document_truncations,
                "depth_truncations": self.depth_truncations,
            },
        }


def normalize_evidence(value: Any, *, max_documents: int = 32,
                       max_chars: int = 8000, max_fields: int = 128,
                       max_depth: int = 6) -> NormalizedEvidence:
    """Convert supported JSON/text shapes without inventing facts for nulls.

    Structured truncation stops at whole fields. Prose truncation prefers a
    sentence or word boundary but retains a bounded prefix if none exists.
    Lexical selection and tokenization may further reduce these documents.
    """
    out = NormalizedEvidence()

    def append_document(text: str, source_id: str | None = None) -> None:
        if len(out.documents) >= max_documents:
            out.document_truncations += 1
            out.truncated_records += 1
            return
        out.documents.append(text)
        out.document_sources.append(source_id)
        if source_id is not None:
            out.source_ids.append(source_id)

    def add_text(text: str, source_id: str | None = None) -> None:
        text = text.strip()
        if not text:
            return
        if len(text) > max_chars:
            boundary = max(text.rfind(".", 0, max_chars), text.rfind("!", 0, max_chars),
                           text.rfind("?", 0, max_chars))
            out.truncated_records += 1
            out.character_truncations += 1
            if boundary >= 0:
                text = text[:boundary + 1]
            else:
                word_boundary = text.rfind(" ", 0, max_chars + 1)
                text = text[:word_boundary] if word_boundary > 0 else text[:max_chars]
        if text:
            append_document(text, source_id)

    def structured(record: dict[str, Any]) -> None:
        source = record.get("source_id")
        source_id = str(source) if source is not None else None
        lines: list[str] = []
        size = 0

        def visit(path: str, item: Any, depth: int) -> None:
            nonlocal size
            if depth > max_depth:
                out.depth_truncations += 1
                out.truncated_records += 1
                return
            if len(lines) >= max_fields:
                out.field_truncations += 1
                out.truncated_records += 1
                return
            if item is None:
                out.null_fields += 1
                return
            if isinstance(item, dict):
                for key in sorted(item, key=str):
                    visit(f"{path}.{key}" if path else str(key), item[key], depth + 1)
                return
            if isinstance(item, list):
                for index, child in enumerate(item):
                    visit(f"{path}[{index}]", child, depth + 1)
                return
            if not isinstance(item, (str, int, float, bool)):
                out.malformed_records += 1
                return
            rendered = str(item).strip()
            if not rendered:
                return
            line = f"{path}: {rendered}" if path else rendered
            if size + len(line) + 1 > max_chars:
                out.character_truncations += 1
                out.truncated_records += 1
                return
            lines.append(line)
            size += len(line) + 1

        # Titles and publication dates can be factual evidence, not merely
        # provenance (e.g. a claim about when a document was published).
        factual_fields = {key: item for key, item in record.items()
                          if key not in {"source_id", "source", "url"}}
        visit("", factual_fields, 0)
        if lines:
            append_document("\n".join(lines), source_id)

    def consume(item: Any) -> None:
        if isinstance(item, str):
            text = item.strip()
            if text.startswith(("{", "[")):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    out.malformed_records += 1
                    return
                if isinstance(parsed, (dict, list)):
                    consume(parsed)
                    return
            add_text(text)
        elif isinstance(item, list):
            for child in item[:max_documents]:
                consume(child)
            if len(item) > max_documents:
                out.document_truncations += len(item) - max_documents
                out.truncated_records += 1
        elif isinstance(item, dict):
            for wrapper in ("evidence", "passages", "documents", "context"):
                if wrapper in item and isinstance(item[wrapper], (list, dict)):
                    before = len(out.documents)
                    consume(item[wrapper])
                    parent_source = item.get("source_id")
                    if parent_source is not None:
                        for index in range(before, len(out.documents)):
                            if out.document_sources[index] is None:
                                source_id = str(parent_source)
                                out.document_sources[index] = source_id
                                out.source_ids.append(source_id)
                    siblings = {key: value for key, value in item.items()
                                if key not in {wrapper, "source", "source_id", "url"}}
                    if siblings:
                        structured({**siblings, "source_id": parent_source})
                    return
            text_key = next((k for k in ("snippet", "text", "content") if isinstance(item.get(k), str)), None)
            metadata_keys = {"snippet", "text", "content", "title", "source", "source_id", "url", "publication_date"}
            if text_key and set(item).issubset(metadata_keys):
                source_id = str(item["source_id"]) if item.get("source_id") is not None else None
                add_text(item[text_key], source_id)
                factual_metadata = {key: item[key] for key in ("title", "publication_date") if key in item}
                if factual_metadata:
                    structured({**factual_metadata, "source_id": source_id})
            else:
                structured(item)
        elif item is not None:
            out.malformed_records += 1

    consume(value)
    return out
