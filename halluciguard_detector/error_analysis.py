"""Categorise CONTRADICTED false negatives so the next bottleneck is visible.

A single recall number says "the detector misses contradictions". It does not
say *which* ones, and the fix is completely different for each kind:

* the evidence snippet was never about the claim (retrieval),
* the claim carries two facts and one snippet cannot settle it (decomposition),
* the claim and evidence disagree in a specific slot the model ignores (NLI),
* the model saw a clean contradiction and still said "not enough" (capacity).

The categories below are assigned by **deterministic heuristics**, and the
report says so. They are triage labels for deciding what to work on next, not
verified ground truth; a human reading the listed examples is expected to
overrule some of them.

Two of the categories exist because an earlier version of this file reported
them as ``RETRIEVAL`` and that was wrong. Roughly 12% of RAGTruth test rows
come from structured Yelp-style sources whose ``source_info`` is a JSON
document; the sentence splitter cannot segment those, so the model is handed a
raw JSON record as its "evidence sentence" and the low lexical overlap got
billed to retrieval. One-word claims such as ``"Somali-Canadian"`` produced the
same mislabel from the other direction, because Jaccard overlap punishes a
short claim for being short. ``UNSEGMENTED_SOURCE`` and coverage-based
``RETRIEVAL`` separate those from genuine retriever misses.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .text import has_entity_conflict, numeric_consistency, shared_relation

#: Domain-generic words carry no topic information, so sharing one does not
#: mean the evidence is about the same thing. Used only by the retrieval gate:
#: a claim and a passage both saying "company" is not evidence of relevance.
GENERIC_TERMS = frozenset(
    {
        "company", "companies", "business", "businesses", "million", "billion",
        "thousand", "users", "user", "customers", "customer", "people",
        "person", "years", "year", "months", "month", "weeks", "week", "days",
        "day", "hours", "hour", "minutes", "minute", "service", "services",
        "product", "products", "market", "markets", "system", "systems",
        "number", "numbers", "time", "times", "part", "parts", "group",
        "member", "members", "team", "teams", "manager", "managers", "staff",
        "made", "make", "makes", "new", "more", "most", "less", "least",
        "than", "then", "also", "including", "include", "includes", "based",
        "used", "use", "uses", "using", "one", "two", "three", "first",
        "second", "third", "last", "next", "other", "others", "many", "most",
        "such", "same", "different", "total", "overall", "about", "after",
        "before", "during", "since", "until", "while", "where", "when",
        "which", "who", "whom", "whose", "what", "there", "here", "only",
        "even", "still", "already", "always", "never", "often", "usually",
    }
)

#: A claim is treated as off-topic only when *none* of its non-generic content
#: words appear in the selected evidence. Two earlier versions of this rule
#: were wrong in instructive ways. Jaccard overlap punished one-word claims
#: like "Somali-Canadian" for being short, and a plain coverage threshold
#: reclassified "Apple acquired Company A" against "Apple has partnered with
#: companies" as a retrieval miss, when the passage was plainly about the same
#: subject and merely said something else. Sharing a non-generic term is
#: treated as on-topic, and the different-relation case is left to RELATION.
RETRIEVAL_COVERAGE_FLOOR = 0.34

#: Below this Jaccard the evidence is certainly about a different subject.
#: Kept as a secondary, stricter signal rather than the primary test.
RETRIEVAL_JACCARD_FLOOR = 0.08

_NEGATION = re.compile(
    r"\b(?:not|never|no longer|isn't|wasn't|hasn't|haven't|does not|did not|"
    r"cannot|can't|without|neither|nor|refused to|failed to)\b",
    re.IGNORECASE,
)
_TEMPORAL = re.compile(
    r"\b(?:as of|since|until|by then|at the time|previously|formerly|"
    r"originally|initially|later|eventually|subsequently)\b",
    re.IGNORECASE,
)
_COORDINATOR = re.compile(
    r"\b(?:and|but|while|whereas|although|though|however|plus)\b", re.IGNORECASE
)
_TOKEN = re.compile(r"[a-z0-9]+")

CATEGORIES = (
    "UNSEGMENTED_SOURCE",
    "RETRIEVAL",
    "CLAIM_DECOMPOSITION",
    "DATE",
    "NUMBER",
    "ENTITY",
    "NEGATION",
    "RELATION",
    "TEMPORAL",
    "MULTI_HOP",
    "NLI_MODEL",
    "OTHER",
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if len(t) > 2}


def _looks_unsegmented(evidence: str) -> bool:
    """True when the evidence side is a structured record, not prose.

    RAGTruth's Yelp-derived sources are JSON documents. Splitting them into
    sentences yields nothing usable, so what reaches the model is the raw
    record, sometimes truncated mid-object by a character limit.

    The rule is deliberately dumb: a record opens with ``{`` or ``[``. An
    earlier version also tried to parse the text, and required several
    ``"key":`` pairs before believing it -- which both over- and
    under-counted. Parsing rejected truncated records (still records), and the
    pair count matched ordinary prose that quoted a field, labelling plain
    reviews as "unsegmented". Measured on the natural test set this rule
    selects 12.0% of all rows and 36.3% of CONTRADICTED rows.
    """
    head = evidence.strip()
    return bool(head) and head[0] in "{["


def _jaccard(claim: str, evidence: str) -> float:
    left = _content_tokens(claim)
    right = _content_tokens(evidence)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _is_compound(claim: str) -> bool:
    """True when a claim reads as two or more coordinated clauses.

    A coordinator alone is not enough -- plenty of single-clause claims contain
    "and". The test is whether the text on *both* sides of a coordinator carries
    real content, so "Tesla was founded in 2003 and acquired by Google in 2018"
    is compound while "The judge and the verifier disagree" is not (only one
    content word on the left).

    Compound claims are reported separately because one selected snippet cannot
    settle two facts, so a missed contradiction there is a decomposition problem
    rather than an NLI problem.
    """
    parts = [part for part in _COORDINATOR.split(claim) if part.strip()]
    if len(parts) < 2:
        return False
    return all(len(_content_tokens(part)) >= 2 for part in parts)


def categorise(claim: str, evidence: str) -> Dict[str, Any]:
    """Assign one failure category to a CONTRADICTED false negative.

    The first matching rule wins, so the order encodes "which explanation is
    more fundamental": if the evidence is off-topic nothing about the claim's
    internal structure can be the cause.
    """
    evidence = (evidence or "").strip()
    claim = (claim or "").strip()
    overlap = _jaccard(claim, evidence)
    claim_words = _content_tokens(claim)
    specific = claim_words - GENERIC_TERMS
    # Fall back to all content words for a claim made entirely of generic
    # terms ("the company was sold"), which must not be declared off-topic
    # just because nothing distinctive is left to compare.
    distinctive = specific or claim_words
    covered = distinctive & _content_tokens(evidence)
    coverage = (len(covered) / len(distinctive)) if distinctive else 0.0

    if not evidence:
        return {"category": "RETRIEVAL", "reason": "no evidence was selected"}

    # A source that never went through sentence segmentation. Checked first
    # because it explains the rest: when the "evidence sentence" is a raw JSON
    # record, no amount of claim/evidence reasoning applies, and calling it a
    # retrieval miss hides a dataset-preparation defect behind a plausible
    # story. About 12% of RAGTruth test rows come from structured (Yelp-style)
    # sources.
    if _looks_unsegmented(evidence):
        return {
            "category": "UNSEGMENTED_SOURCE",
            "reason": (
                "the evidence side is a raw structured record, not a sentence; "
                "the sentence-level evidence pipeline did not segment this source"
            ),
        }

    if not covered:
        return {
            "category": "RETRIEVAL",
            "reason": (
                "evidence shares no distinctive term with the claim "
                f"(jaccard={overlap:.3f}, coverage={coverage:.0%})"
            ),
        }

    clause_like = _is_compound(claim)
    if clause_like:
        return {
            "category": "CLAIM_DECOMPOSITION",
            "reason": "claim reads as two or more coordinated clauses",
        }

    numeric = numeric_consistency(claim, evidence)
    if numeric:
        kind = "NUMBER" if any("conflicting" in item and "year" not in item for item in numeric) else "DATE"
        return {"category": kind, "reason": "; ".join(numeric)}

    # Negation is checked on term overlap with the claim rather than on
    # ``shared_relation``: "Safari runs on Windows" against "Safari is a
    # browser for macOS and iOS, not Windows" shares no relation word at all,
    # yet the evidence plainly denies the claim. Requiring a shared predicate
    # here would hide exactly the cases the category exists to find.
    shared_terms = _content_tokens(claim) & _content_tokens(evidence)
    if _NEGATION.search(evidence) and len(shared_terms) >= 2:
        return {
            "category": "NEGATION",
            "reason": "evidence negates terms the claim also asserts",
        }

    if has_entity_conflict(claim, evidence):
        return {
            "category": "ENTITY",
            "reason": "named entity swapped on an otherwise shared relation",
        }

    if _TEMPORAL.search(evidence) and not _TEMPORAL.search(claim):
        return {
            "category": "TEMPORAL",
            "reason": "evidence is explicitly time-scoped and the claim is not",
        }

    if shared_relation(claim, evidence):
        return {
            "category": "NLI_MODEL",
            "reason": (
                "the selected evidence is comparable to the claim and states a "
                "different fact, yet the model did not predict CONTRADICTED"
            ),
        }

    return {
        "category": "RELATION",
        "reason": "claim and evidence share a subject but no relation word",
    }


def analyse(
    rows: Sequence[dict],
    logits: Any,
    labels: Any,
    *,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Categorise every CONTRADICTED false negative in one split.

    ``rows``, ``logits`` and ``labels`` must be in the same order. Only true
    CONTRADICTED rows that were *not* predicted CONTRADICTED are categorised;
    false positives are reported as a count so the asymmetry is visible.
    """
    import numpy as np

    from .calibration import class_probabilities

    probabilities = class_probabilities(np.asarray(logits), temperature)
    predicted = probabilities.argmax(axis=1)
    labels = np.asarray(labels).astype(int)

    counts: Counter = Counter()
    false_negatives: List[Dict[str, Any]] = []
    false_positives: List[Dict[str, Any]] = []
    true_positive = 0
    for index, row in enumerate(rows):
        truth = int(labels[index])
        guess = int(predicted[index])
        if truth == 1 and guess == 1:
            true_positive += 1
            continue
        if truth == 1:
            verdict = categorise(str(row.get("claim", "")), str(row.get("evidence", "")))
            counts[verdict["category"]] += 1
            false_negatives.append(
                {
                    "id": row.get("id"),
                    "claim": row.get("claim"),
                    "evidence": str(row.get("evidence", ""))[:400],
                    "evidence_shape": row.get("evidence_shape"),
                    "predicted": ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")[guess],
                    "p_contradicted": round(float(probabilities[index, 1]), 6),
                    **verdict,
                }
            )
        elif guess == 1:
            false_positives.append(
                {
                    "id": row.get("id"),
                    "claim": row.get("claim"),
                    "evidence": str(row.get("evidence", ""))[:400],
                    "true": ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")[truth],
                    "p_contradicted": round(float(probabilities[index, 1]), 6),
                }
            )
    support = int((labels == 1).sum())
    return {
        "rows": len(rows),
        "contradicted_support": support,
        "contradicted_true_positives": true_positive,
        "contradicted_false_negatives": len(false_negatives),
        "contradicted_recall": round(true_positive / support, 6) if support else None,
        "false_positives": len(false_positives),
        "category_counts": {
            category: counts.get(category, 0)
            for category in CATEGORIES
            if counts.get(category, 0)
        },
        "method": (
            "Deterministic first-match rules over the claim and the selected "
            "evidence, using the same comparability guards the runtime uses. "
            "These are triage labels, not verified ground truth."
        ),
        "false_negative_examples": false_negatives[:40],
        "false_positive_examples": false_positives[:20],
    }


def main(argv: List[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="metric")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("reports/detector_v2_error_analysis.json"))
    args = parser.parse_args(argv)

    import torch

    from .calibration import load_calibration
    from .experiments import score_rows
    from .training import load_rows

    rows = load_rows(Path(args.data_dir) / f"{args.split}.jsonl")
    scored = score_rows(
        args.checkpoint,
        rows,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=args.batch_size,
    )
    calibration = load_calibration(Path(args.checkpoint) / "calibration.json")
    report = analyse(
        rows, scored["logits"], scored["labels"], temperature=float(calibration.get("temperature", 1.0))
    )
    report["checkpoint"] = str(args.checkpoint)
    report["split"] = args.split
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("examples")}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
