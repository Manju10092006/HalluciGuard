"""
HaluEval Dataset Formatter and Constants — Staging Runtime.

Single canonical source of truth for detector input formatting.
"""

from typing import Optional

# Label constants — must match model config id2label exactly
NO_HALLUCINATION = 0
HALLUCINATION = 1


def format_detector_input(query: str, response: str, context: Optional[str] = None) -> str:
    """Canonical formatter for detector input.

    Used by BOTH training and inference — single source of truth.

    Args:
        query:    User query or question.
        response: LLM response / answer to classify.
        context:  Optional reference context (e.g., knowledge base, retrieved doc).
                  If None or empty, the example is contextless (production path).

    Returns:
        Formatted input string for the classifier.
    """
    parts = []
    if query and query.strip():
        parts.append(f"Query: {query.strip()}")
    if context and context.strip():
        ctx = context.strip()
        if len(ctx) > 1500:
            ctx = ctx[:1500] + "..."
        parts.append(f"Context: {ctx}")
    if response and response.strip():
        parts.append(f"Answer: {response.strip()}")
    return "\n".join(parts)


__all__ = [
    "NO_HALLUCINATION",
    "HALLUCINATION",
    "format_detector_input",
]
