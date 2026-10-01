"""Bounded evidence-to-context adapter for the existing HaluEval input format."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreparedEvidence:
    context: str = ""
    selected: list[dict[str, str | None]] = field(default_factory=list)
    normalized_count: int = 0
    malformed_records: int = 0
    null_fields: int = 0
    truncated_fields: int = 0
    truncated_documents: int = 0
    truncated_characters: int = 0

    def diagnostics(self) -> dict:
        return {
            "normalized_count": self.normalized_count,
            "selected": self.selected,
            "selected_context": self.context,
            "malformed_records": self.malformed_records,
            "null_fields": self.null_fields,
            "truncated_fields": self.truncated_fields,
            "truncated_documents": self.truncated_documents,
            "truncated_characters": self.truncated_characters,
            "degraded": bool(self.malformed_records or self.truncated_fields
                             or self.truncated_documents or self.truncated_characters),
        }


def prepare_evidence(value: Any, *, max_context_chars: int = 1500,
                     max_documents: int = 16, max_fields: int = 64,
                     max_depth: int = 5) -> PreparedEvidence:
    """Render factual field paths and source IDs; never infer facts from absent/null fields.

    The returned context is precisely the context supplied to the canonical
    HaluEval formatter. Tokenization may truncate it further and is measured
    separately by inference diagnostics.
    """
    result = PreparedEvidence()
    documents: list[tuple[str, str | None, bool]] = []

    def append(text: str, source: str | None, structured: bool = False) -> None:
        if not text.strip():
            return
        if len(documents) >= max_documents:
            result.truncated_documents += 1
            return
        documents.append((text.strip(), source, structured))

    def record(item: dict, inherited_source: str | None = None) -> None:
        source = item.get("source_id", inherited_source)
        source_id = str(source) if source is not None else None
        lines: list[str] = []

        def walk(path: str, value: Any, depth: int) -> None:
            if depth > max_depth or len(lines) >= max_fields:
                result.truncated_fields += 1
                return
            if value is None:
                result.null_fields += 1
            elif isinstance(value, dict):
                for key in sorted(value, key=str):
                    if key not in {"source_id", "url", "source"}:
                        walk(f"{path}.{key}" if path else str(key), value[key], depth + 1)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(f"{path}[{index}]", child, depth + 1)
            elif isinstance(value, (str, int, float, bool)):
                if str(value).strip():
                    lines.append(f"{path}: {value}" if path else str(value))
            else:
                result.malformed_records += 1

        walk("", item, 0)
        append("\n".join(lines), source_id, structured=True)

    def consume(item: Any, inherited_source: str | None = None) -> None:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped.startswith(("{", "[")):
                try:
                    decoded = json.loads(stripped)
                except json.JSONDecodeError:
                    result.malformed_records += 1
                    return
                if isinstance(decoded, (dict, list)):
                    consume(decoded, inherited_source)
                    return
            append(stripped, inherited_source)
        elif isinstance(item, list):
            for child in item:
                consume(child, inherited_source)
        elif isinstance(item, dict):
            source = item.get("source_id", inherited_source)
            source_id = str(source) if source is not None else None
            wrapper = next((key for key in ("evidence", "passages", "documents", "context")
                            if isinstance(item.get(key), (dict, list))), None)
            if wrapper:
                consume(item[wrapper], source_id)
                siblings = {k: v for k, v in item.items() if k != wrapper}
                if any(k not in {"source_id", "source", "url"} for k in siblings):
                    record(siblings, source_id)
            else:
                record(item, source_id)
        elif item is not None:
            result.malformed_records += 1

    consume(value)
    result.normalized_count = len(documents)
    remaining = max_context_chars
    parts: list[str] = []
    for index, (text, source, structured) in enumerate(documents):
        separator = "\n" if parts else ""
        if remaining <= len(separator):
            result.truncated_documents += len(documents) - index
            break
        space = remaining - len(separator)
        if structured and len(text) > space:
            selected_lines: list[str] = []
            used = 0
            for line in text.splitlines():
                line_size = len(line) + (1 if selected_lines else 0)
                if used + line_size > space:
                    result.truncated_fields += 1
                    continue
                selected_lines.append(line)
                used += line_size
            selected_text = "\n".join(selected_lines)
        else:
            selected_text = text[:space]
        if len(text) > space:
            result.truncated_characters += len(text) - len(selected_text)
        if not selected_text:
            result.truncated_documents += 1
            continue
        parts.append(separator + selected_text)
        result.selected.append({"source_id": source, "text": selected_text})
        remaining -= len(separator) + len(selected_text)
    result.context = "".join(parts)
    return result
