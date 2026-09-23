"""
scripts / build_p5.py
─────────────────────
Build the P5 ADVERSARIAL / hard-negative dataset (handoff spec §15).

Why this exists
    A strong Base LLM answers easy factual questions correctly, so natural P0
    generation yields almost no CONTRADICTED claims -> a one-class, untrainable
    set. P5 fixes this by taking KNOWN-CORRECT claims and deliberately
    corrupting them into hallucinations with KNOWN ground truth. The corrupted
    claim is guaranteed-false (label=1); its untouched twin is the matched
    true control (label=0). This yields balanced, style-matched pairs that test
    whether the detector responds to factual RISK rather than writing style.

Corruption operators (each records what it changed)
    number_swap   : change a number to a nearby but wrong value
    year_shift    : shift a 4-digit year by ±1..3
    entity_swap   : replace a known entity with a plausible wrong one
    negation      : insert / remove a negation
    one_char_num  : change a single digit (subtle numeric error)

Design guarantees (spec §16, §42)
    - Splitting is BY SOURCE CLAIM: a true claim and its corrupted twin never
      straddle a split, and no source claim appears in two splits.
    - Original claim text is preserved for the control; corruption is applied
      to a copy. Immutable claim IDs are assigned fresh per output claim.
    - Every corruption is auditable (operator + before/after recorded).

Sources
    --from-p0 <file>   : mine SUPPORTED claims from a built P0 file as the
                         true-claim pool (preferred: uses real Base-LLM text).
    (built-in seed pool is used if --from-p0 is omitted.)

Usage
    python halluciguard_detector/scripts/build_p5.py \
        --from-p0 halluciguard_detector/data/raw/P0_all.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("build_p5")

DATA_DIR = _ROOT / "data" / "processed"
RAW_DIR = _ROOT / "data" / "raw"
MANIFEST_DIR = _ROOT / "data" / "manifests"

_NUM_RE = re.compile(r"\b\d[\d,]*\b")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")

# Plausible wrong-entity substitutions for entity_swap. Kept small and
# well-known so the corrupted claim reads naturally but is clearly false.
_ENTITY_SWAPS: List[Tuple[str, str]] = [
    ("James Gosling", "Dennis Ritchie"),
    ("Guido van Rossum", "Bjarne Stroustrup"),
    ("Canberra", "Sydney"),
    ("William Shakespeare", "Christopher Marlowe"),
    ("Shakespeare", "Marlowe"),
    ("Leonardo da Vinci", "Michelangelo"),
    ("Jupiter", "Saturn"),
    ("Neil Armstrong", "Buzz Aldrin"),
    ("Albert Einstein", "Isaac Newton"),
    ("Einstein", "Newton"),
    ("Portuguese", "Spanish"),
    ("Sun Microsystems", "Bell Labs"),
    ("Pacific", "Atlantic"),
    ("Paris", "Lyon"),
    ("Oxygen", "Osmium"),
]


@dataclass
class TrueClaim:
    text: str
    query: str
    source_id: str


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Corruption operators. Each returns (corrupted_text, detail) or None if the
# operator does not apply to this claim.
# ─────────────────────────────────────────────────────────────────────────────
def op_number_swap(text: str, rng: random.Random) -> Optional[Tuple[str, str]]:
    nums = [m for m in _NUM_RE.finditer(text) if not _YEAR_RE.fullmatch(m.group())]
    if not nums:
        return None
    m = rng.choice(nums)
    orig = m.group()
    val = int(orig.replace(",", ""))
    delta = rng.choice([v for v in (-100, -10, -3, 3, 10, 100) if val + v != val])
    new_val = max(0, val + delta)
    if new_val == val:
        new_val = val + 7
    new = text[:m.start()] + str(new_val) + text[m.end():]
    return new, f"number_swap {orig}->{new_val}"


def op_year_shift(text: str, rng: random.Random) -> Optional[Tuple[str, str]]:
    ms = list(_YEAR_RE.finditer(text))
    if not ms:
        return None
    m = rng.choice(ms)
    year = int(m.group())
    shift = rng.choice([-3, -2, -1, 1, 2, 3])
    new_year = year + shift
    new = text[:m.start()] + str(new_year) + text[m.end():]
    return new, f"year_shift {year}->{new_year}"


def op_one_char_num(text: str, rng: random.Random) -> Optional[Tuple[str, str]]:
    idxs = [i for i, ch in enumerate(text) if ch.isdigit()]
    if not idxs:
        return None
    i = rng.choice(idxs)
    orig = text[i]
    choices = [d for d in "0123456789" if d != orig]
    new_ch = rng.choice(choices)
    new = text[:i] + new_ch + text[i + 1:]
    return new, f"one_char_num pos{i} {orig}->{new_ch}"


def op_entity_swap(text: str, rng: random.Random) -> Optional[Tuple[str, str]]:
    applicable = [(a, b) for a, b in _ENTITY_SWAPS if a in text]
    if not applicable:
        return None
    a, b = rng.choice(applicable)
    new = text.replace(a, b, 1)
    return new, f"entity_swap {a}->{b}"


def op_negation(text: str, rng: random.Random) -> Optional[Tuple[str, str]]:
    # Remove an existing negation if present, else insert one after the first
    # auxiliary/verb. Keep it readable.
    for neg in (" was not ", " did not ", " is not ", " does not ", " were not "):
        if neg in text:
            new = text.replace(neg, neg.replace(" not ", " "), 1)
            return new, f"negation removed '{neg.strip()}'"
    for aux in (" was ", " is ", " were ", " are ", " created ", " released ", " founded "):
        if aux in text:
            replacement = aux.rstrip() + " not " if aux.strip() in ("was", "is", "were", "are") \
                else " did not " + aux.strip().rstrip("ed").rstrip() + " "
            # simpler, robust: insert "not" for be-verbs; prefix "did not" otherwise
            if aux.strip() in ("was", "is", "were", "are"):
                new = text.replace(aux, f"{aux.rstrip()} not ", 1)
            else:
                new = text.replace(aux, f" did not {aux.strip()} ", 1)
            return new, f"negation inserted at '{aux.strip()}'"
    return None


_OPERATORS = [op_entity_swap, op_year_shift, op_number_swap, op_negation, op_one_char_num]


def corrupt(text: str, rng: random.Random) -> Optional[Tuple[str, str]]:
    """Apply the first applicable operator (order randomized) to get a
    guaranteed-false variant. Returns (corrupted_text, operator_detail)."""
    ops = _OPERATORS[:]
    rng.shuffle(ops)
    for op in ops:
        result = op(text, rng)
        if result and result[0] != text:
            return result
    return None


# ─────────────────────────────────────────────────────────────────────────────
# True-claim sources
# ─────────────────────────────────────────────────────────────────────────────
def _builtin_true_claims() -> List[TrueClaim]:
    return [
        TrueClaim("Java was created by James Gosling at Sun Microsystems in 1995.", "Who created Java?", "s01"),
        TrueClaim("The capital of Australia is Canberra.", "Capital of Australia?", "s02"),
        TrueClaim("World War II ended in 1945.", "When did WWII end?", "s03"),
        TrueClaim("Hamlet was written by William Shakespeare around 1600.", "Who wrote Hamlet?", "s04"),
        TrueClaim("The Mona Lisa was painted by Leonardo da Vinci.", "Who painted the Mona Lisa?", "s05"),
        TrueClaim("Jupiter is the largest planet in the solar system.", "Largest planet?", "s06"),
        TrueClaim("Neil Armstrong was the first person to walk on the Moon in 1969.", "First on the Moon?", "s07"),
        TrueClaim("Water boils at 100 degrees Celsius at sea level.", "Boiling point of water?", "s08"),
        TrueClaim("Python was created by Guido van Rossum.", "Who created Python?", "s09"),
        TrueClaim("The Pacific Ocean is the largest ocean on Earth.", "Largest ocean?", "s10"),
    ]


def true_claims_from_p0(path: str) -> List[TrueClaim]:
    """Mine SUPPORTED claims from a built P0 file to use as the true pool."""
    rows = json.load(open(path, encoding="utf-8"))
    out: List[TrueClaim] = []
    for s in rows:
        for c in s.get("claims", []):
            if c.get("label_semantic") == "SUPPORTED" and c.get("text"):
                out.append(TrueClaim(text=c["text"], query=s.get("query", ""),
                                     source_id=f'{s.get("qid","q")}_{c["claim_id"]}'))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────
def build_pairs(true_claims: List[TrueClaim], rng: random.Random) -> List[dict]:
    """For each true claim, emit the true control (label 0) and, if a corruption
    applies, its guaranteed-false adversarial twin (label 1)."""
    samples: List[dict] = []
    for tc in true_claims:
        corrupted = corrupt(tc.text, rng)
        pair_id = f"P5_{tc.source_id}"
        # true control
        control = {
            "sample_id": f"{pair_id}_true",
            "source_id": tc.source_id,
            "query": tc.query,
            "answer": tc.text,
            "claims": [{
                "claim_id": "claim_001",
                "text": tc.text,
                "label_semantic": "SUPPORTED",
                "label": 0,
                "adversarial": False,
                "operator": None,
            }],
        }
        samples.append(control)
        if corrupted:
            new_text, detail = corrupted
            adv = {
                "sample_id": f"{pair_id}_adv",
                "source_id": tc.source_id,          # SAME source -> same split
                "query": tc.query,
                "answer": new_text,
                "claims": [{
                    "claim_id": "claim_001",
                    "text": new_text,
                    "label_semantic": "CONTRADICTED",
                    "label": 1,
                    "adversarial": True,
                    "operator": detail,
                    "original_text": tc.text,
                }],
            }
            samples.append(adv)
        else:
            logger.warning("  [no-op] no corruption applied to: %s", tc.text[:70])
    return samples


def split_by_source(samples: List[dict], seed: int, ratios=(0.70, 0.15, 0.15)):
    """Split BY source_id so a true claim and its adversarial twin stay together
    and no source appears in two splits (spec §16)."""
    by_src: Dict[str, List[dict]] = {}
    for s in samples:
        by_src.setdefault(s["source_id"], []).append(s)
    srcs = sorted(by_src.keys())
    random.Random(seed).shuffle(srcs)
    n = len(srcs)
    n_train, n_val = int(ratios[0] * n), int(ratios[1] * n)
    train_s, val_s = set(srcs[:n_train]), set(srcs[n_train:n_train + n_val])
    train, val, test = [], [], []
    for src in srcs:
        bucket = train if src in train_s else val if src in val_s else test
        bucket.extend(by_src[src])
    a, b, c = ({s["source_id"] for s in train},
               {s["source_id"] for s in val},
               {s["source_id"] for s in test})
    assert not (a & b) and not (a & c) and not (b & c), "LEAKAGE: source in multiple splits"
    return train, val, test


def _count(samples, label=None):
    return sum(1 for s in samples for c in s["claims"] if label is None or c["label"] == label)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build P5 adversarial hard-negative dataset")
    ap.add_argument("--from-p0", default=None, help="built P0 json to mine SUPPORTED claims from")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="P5")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.from_p0:
        true_claims = true_claims_from_p0(args.from_p0)
        src_desc = args.from_p0
    else:
        true_claims = _builtin_true_claims()
        src_desc = "builtin"
    if not true_claims:
        ap.error("no true claims to corrupt")
    logger.info("Building P5 from %d true claims (source=%s) ...", len(true_claims), src_desc)

    samples = build_pairs(true_claims, rng)
    train, val, test = split_by_source(samples, args.seed)

    write_json(DATA_DIR / f"{args.out_prefix}_train.json", train)
    write_json(DATA_DIR / f"{args.out_prefix}_val.json", val)
    write_json(DATA_DIR / f"{args.out_prefix}_test.json", test)
    write_json(RAW_DIR / f"{args.out_prefix}_all.json", samples)

    manifest = {
        "dataset": args.out_prefix,
        "seed_source": src_desc,
        "generation_method": "deterministic adversarial corruption (spec §15)",
        "operators": [o.__name__ for o in _OPERATORS],
        "n_true_claims": len(true_claims),
        "n_samples": len(samples),
        "label_counts": {"supported_0": _count(samples, 0), "contradicted_1": _count(samples, 1)},
        "splits": {
            "train": {"samples": len(train), "pos": _count(train, 1), "neg": _count(train, 0)},
            "val": {"samples": len(val), "pos": _count(val, 1), "neg": _count(val, 0)},
            "test": {"samples": len(test), "pos": _count(test, 1), "neg": _count(test, 0)},
        },
    }
    write_json(MANIFEST_DIR / f"{args.out_prefix}_manifest.json", manifest)

    logger.info("=" * 66)
    logger.info("  P5 ADVERSARIAL BUILD COMPLETE")
    logger.info("=" * 66)
    logger.info("  true claims: %d  ->  %d samples", len(true_claims), len(samples))
    logger.info("  labels: SUPPORTED(0)=%d  CONTRADICTED(1)=%d",
                manifest["label_counts"]["supported_0"], manifest["label_counts"]["contradicted_1"])
    for name, st in manifest["splits"].items():
        logger.info("  %-6s: %d samples (pos=%d neg=%d)", name, st["samples"], st["pos"], st["neg"])
    logger.info("  manifest -> %s", MANIFEST_DIR / f"{args.out_prefix}_manifest.json")
    logger.info("=" * 66)


if __name__ == "__main__":
    main()
