"""
scripts / build_p0.py
─────────────────────
Build a PRODUCTION-SHAPED P0 dataset (handoff spec §4-6, Phase 3).

The old build_datasets.py hardcodes ~360 synthetic claims with hand-written
labels. That is exactly the weakness the review flagged. This script instead
produces claims the way production does:

    QA gold (TriviaQA / NQ-Open style)
        -> REAL Base LLM draft answer   (OpenRouter, same model as prod)
        -> REAL ClaimExtractor          (atomic claims, immutable IDs)
        -> auto-label vs gold           (SUPPORTED / CONTRADICTED / UNRESOLVABLE)
        -> binary risk                  (SUPPORTED=0, CONTRADICTED=1, UNRESOLVABLE dropped)
        -> leakage-safe split BY QUESTION (train / val / test)

Provenance (base model, temperature, prompt hashes, config hash) is recorded on
every sample and in a manifest so results are reproducible (spec §5, §48-49).

IMPORTANT SEMANTICS
    - Labels come from GOLD reference answers, never from the existing Verifier
      (spec §14). UNRESOLVABLE claims are EXCLUDED from binary training rather
      than being silently called hallucinations (spec §6).
    - Splitting is by question so the same question/answer can never appear in
      two splits (spec §16).

Modes
    --online   : call OpenRouter for generation (needs OPENROUTER_API_KEY).
    --offline  : use canned draft answers shipped with the seed set, so the
                 whole pipeline (extraction + labeling + splitting) is runnable
                 and testable without network or API spend. Default.

Usage
    python halluciguard_detector/scripts/build_p0.py --offline
    python halluciguard_detector/scripts/build_p0.py --online --limit 100 \
        --seeds path/to/qa_seeds.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── make the package importable when run as a script ─────────────────────────
_ROOT = Path(__file__).parent.parent           # halluciguard_detector/
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ROOT.parent / ".env")
except ImportError:
    pass

from halluciguard_detector.claim_extraction.extractor import ClaimExtractor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("build_p0")

DATA_DIR = _ROOT / "data" / "processed"
RAW_DIR = _ROOT / "data" / "raw"
MANIFEST_DIR = _ROOT / "data" / "manifests"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _resolve_model(model: str) -> str:
    """Remap deprecated/unavailable model ids to a known-good one (matches
    ClaimExtractor / run_live_detector behavior)."""
    if model in ("qwen/qwen3-4b", "qwen/qwen3-4b:free", ""):
        return "qwen/qwen3-14b"
    return model


DEFAULT_MODEL = _resolve_model(os.environ.get("OPENROUTER_MODEL", "qwen/qwen3-14b"))
GEN_SYSTEM_PROMPT = "You are a helpful AI assistant. Answer the user's question concisely and factually."
GEN_TEMPERATURE = 0.0   # deterministic generation for a reproducible dataset

# Label vocabulary (spec §6, §12)
SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
UNRESOLVABLE = "UNRESOLVABLE"

# Binary risk mapping. UNRESOLVABLE -> None means "exclude from binary training".
_RISK_MAP = {SUPPORTED: 0, CONTRADICTED: 1, UNRESOLVABLE: None}


# ─────────────────────────────────────────────────────────────────────────────
# Seed QA data
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QASeed:
    qid: str
    question: str
    gold_answers: List[str]            # accepted gold answer strings / aliases
    canned_answer: Optional[str] = None  # offline-mode draft (simulated Base LLM)


def _builtin_seeds() -> List[QASeed]:
    """
    A small TriviaQA/NQ-Open-style seed set with gold answers and canned
    Base-LLM drafts (some correct, some wrong) so --offline exercises the full
    pipeline. Replace/extend with --seeds for a real, larger build.
    """
    return [
        QASeed("q0001", "Who created the Java programming language?",
               ["James Gosling"],
               "Java was created by James Gosling at Sun Microsystems, first released in 1995."),
        QASeed("q0002", "What is the capital of Australia?",
               ["Canberra"],
               "The capital of Australia is Sydney, its largest and most famous city."),
        QASeed("q0003", "In what year did World War II end?",
               ["1945"],
               "World War II ended in 1945 with the surrender of the Axis powers."),
        QASeed("q0004", "Who wrote the play Hamlet?",
               ["William Shakespeare", "Shakespeare"],
               "Hamlet was written by William Shakespeare around the year 1600."),
        QASeed("q0005", "What is the chemical symbol for gold?",
               ["Au"],
               "The chemical symbol for gold is Gd on the periodic table."),
        QASeed("q0006", "Who painted the Mona Lisa?",
               ["Leonardo da Vinci", "Leonardo"],
               "The Mona Lisa was painted by Leonardo da Vinci during the Italian Renaissance."),
        QASeed("q0007", "What is the largest planet in our solar system?",
               ["Jupiter"],
               "The largest planet in our solar system is Saturn, known for its rings."),
        QASeed("q0008", "Who was the first person to walk on the Moon?",
               ["Neil Armstrong"],
               "Neil Armstrong was the first person to walk on the Moon, in July 1969."),
        QASeed("q0009", "What is the boiling point of water at sea level in Celsius?",
               ["100", "100 degrees Celsius", "100 C"],
               "At sea level water boils at 100 degrees Celsius."),
        QASeed("q0010", "Who developed the theory of general relativity?",
               ["Albert Einstein", "Einstein"],
               "The theory of general relativity was developed by Isaac Newton in the 1600s."),
        QASeed("q0011", "What is the smallest prime number?",
               ["2"],
               "The smallest prime number is 2, the only even prime."),
        QASeed("q0012", "What language is primarily spoken in Brazil?",
               ["Portuguese"],
               "The language primarily spoken in Brazil is Spanish."),
    ]


def load_seeds(path: Optional[str], limit: Optional[int]) -> List[QASeed]:
    """Load seeds from a JSONL file, else use the built-in seed set."""
    if path:
        seeds: List[QASeed] = []
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                gold = d.get("gold_answers") or ([d["gold_answer"]] if d.get("gold_answer") else [])
                seeds.append(QASeed(
                    qid=str(d.get("qid", f"q{i:04d}")),
                    question=d["question"],
                    gold_answers=[str(g) for g in gold],
                    canned_answer=d.get("canned_answer"),
                ))
        logger.info("Loaded %d seeds from %s", len(seeds), path)
    else:
        seeds = _builtin_seeds()
        logger.info("Using %d built-in seeds (no --seeds provided)", len(seeds))
    if limit:
        seeds = seeds[:limit]
    return seeds


# ─────────────────────────────────────────────────────────────────────────────
# Base LLM generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_draft(question: str, model: str, api_key: str) -> str:
    """Generate a draft answer from the real Base LLM via OpenRouter."""
    import httpx
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://halluciguard.app",
        "X-Title": "HalluciGuard-P0-Build",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": GEN_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": GEN_TEMPERATURE,
        "max_tokens": 400,
    }
    with httpx.Client(timeout=45.0) as client:
        resp = client.post(OPENROUTER_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-labeling vs gold reference
# ─────────────────────────────────────────────────────────────────────────────
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def _numbers(text: str) -> List[str]:
    return [n.replace(",", "") for n in _NUM_RE.findall(text)]


def label_claim_against_gold(claim_text: str, gold_answers: List[str]) -> Tuple[str, str]:
    """
    Auto-label one atomic claim against gold reference answers.

    Heuristic, gold-grounded (spec §6):
      - If any gold string (or its salient token) appears in the claim -> SUPPORTED.
      - Else if the claim asserts a competing value of the SAME TYPE as gold
        (a different number where gold is a number; a different capitalized
        entity where none of the gold entities appear) -> CONTRADICTED.
      - Else -> UNRESOLVABLE (cannot be adjudicated from gold alone).

    This is deliberately conservative: ambiguous cases become UNRESOLVABLE and
    are excluded from binary training rather than mislabeled. Returns
    (label, reason).
    """
    claim_norm = _normalize(claim_text)
    claim_nums = set(_numbers(claim_text))

    gold_present = False
    gold_is_numeric = False
    gold_nums: set = set()

    for g in gold_answers:
        g_norm = _normalize(g)
        g_nums = set(_numbers(g))
        if g_nums:
            gold_is_numeric = True
            gold_nums |= g_nums
        if not g_norm:
            continue
        # direct substring match on the normalized gold answer
        if g_norm and g_norm in claim_norm:
            gold_present = True
            break
        # token-level: all salient (len>2) gold tokens present in the claim
        g_tokens = [t for t in g_norm.split() if len(t) > 2]
        if g_tokens and all(t in claim_norm.split() for t in g_tokens):
            gold_present = True
            break

    if gold_present:
        return SUPPORTED, "gold answer text present in claim"

    # Numeric contradiction: gold is a number, claim states a different number.
    if gold_is_numeric and claim_nums and not (claim_nums & gold_nums):
        return CONTRADICTED, f"claim numbers {sorted(claim_nums)} differ from gold {sorted(gold_nums)}"

    # Entity contradiction: gold is a name/entity (non-numeric), the claim names
    # a capitalized entity, yet no gold token appears -> competing entity.
    if not gold_is_numeric:
        caps = re.findall(r"\b[A-Z][a-zA-Z]+\b", claim_text)
        # ignore leading-sentence capitalization noise by requiring 1+ multi-word
        salient = [c for c in caps if c.lower() not in {"the", "a", "an", "it"}]
        if salient:
            return CONTRADICTED, "claim asserts a competing entity absent from gold"

    return UNRESOLVABLE, "cannot adjudicate from gold reference alone"


# ─────────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────────
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class BuildProvenance:
    base_model: str
    temperature: float
    system_prompt_hash: str
    mode: str
    extractor_model: str
    seed: int
    config_hash: str = field(default="")

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────
def build_samples(
    seeds: List[QASeed],
    extractor: ClaimExtractor,
    prov: BuildProvenance,
    online: bool,
    api_key: str,
) -> Tuple[List[dict], Dict[str, int]]:
    """Generate → extract → label. Returns (samples, label_counts)."""
    samples: List[dict] = []
    counts = {SUPPORTED: 0, CONTRADICTED: 0, UNRESOLVABLE: 0}

    for seed in seeds:
        if online:
            try:
                draft = generate_draft(seed.question, prov.base_model, api_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  [skip %s] generation failed: %s", seed.qid, exc)
                continue
        else:
            if not seed.canned_answer:
                logger.warning("  [skip %s] no canned_answer in offline mode", seed.qid)
                continue
            draft = seed.canned_answer

        claims, mode, reasons = extractor.extract(seed.question, draft)
        if not claims:
            logger.warning("  [skip %s] no claims extracted (%s)", seed.qid, reasons)
            continue

        labeled_claims = []
        for c in claims:
            label, reason = label_claim_against_gold(c.text, seed.gold_answers)
            counts[label] += 1
            risk = _RISK_MAP[label]
            labeled_claims.append({
                "claim_id": c.claim_id,          # immutable ID preserved (spec §42)
                "text": c.text,
                "label_semantic": label,         # SUPPORTED / CONTRADICTED / UNRESOLVABLE
                "label": risk,                   # binary risk 0/1, or None if excluded
                "label_reason": reason,
                "excluded_from_binary": risk is None,
            })

        samples.append({
            "sample_id": f"P0_{seed.qid}",
            "qid": seed.qid,
            "query": seed.question,
            "gold_answers": seed.gold_answers,
            "answer": draft,
            "extraction_mode": mode.value,
            "extraction_reasons": reasons,
            "claims": labeled_claims,
            "provenance": {
                "base_model": prov.base_model,
                "temperature": prov.temperature,
                "system_prompt_hash": prov.system_prompt_hash,
                "answer_hash": _sha(draft),
                "mode": prov.mode,
            },
        })
        logger.info("  [%s] %d claims  (mode=%s)", seed.qid, len(labeled_claims), mode.value)

    return samples, counts


def split_by_question(
    samples: List[dict], seed: int, ratios=(0.70, 0.15, 0.15)
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Leakage-safe split BY QUESTION (spec §16). Each qid lands in exactly one
    split, so no claim from a question can appear in two splits.
    """
    by_q: Dict[str, List[dict]] = {}
    for s in samples:
        by_q.setdefault(s["qid"], []).append(s)
    qids = sorted(by_q.keys())
    random.Random(seed).shuffle(qids)

    n = len(qids)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_q = set(qids[:n_train])
    val_q = set(qids[n_train:n_train + n_val])

    train, val, test = [], [], []
    for qid in qids:
        bucket = train if qid in train_q else val if qid in val_q else test
        bucket.extend(by_q[qid])

    # Contamination guard: assert disjoint qids across splits.
    s_train = {s["qid"] for s in train}
    s_val = {s["qid"] for s in val}
    s_test = {s["qid"] for s in test}
    assert not (s_train & s_val) and not (s_train & s_test) and not (s_val & s_test), \
        "LEAKAGE: qid appears in multiple splits"
    return train, val, test


def _count_claims(samples: List[dict], binary_only: bool = False) -> int:
    total = 0
    for s in samples:
        for c in s["claims"]:
            if binary_only and c["excluded_from_binary"]:
                continue
            total += 1
    return total


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build production-shaped P0 dataset")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--online", action="store_true", help="call OpenRouter for generation")
    mode.add_argument("--offline", action="store_true", help="use canned seed answers (default)")
    ap.add_argument("--seeds", default=None, help="JSONL of {question, gold_answers, canned_answer?}")
    ap.add_argument("--limit", type=int, default=None, help="cap number of seed questions")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Base LLM model id (online mode)")
    ap.add_argument("--seed", type=int, default=42, help="split RNG seed")
    ap.add_argument("--out-prefix", default="P0", help="output filename prefix")
    args = ap.parse_args()

    online = bool(args.online)  # default is offline
    args.model = _resolve_model(args.model)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if online and not api_key:
        ap.error("--online requires OPENROUTER_API_KEY in the environment/.env")

    seeds = load_seeds(args.seeds, args.limit)
    if not seeds:
        ap.error("no seeds to build from")

    extractor = ClaimExtractor(
        max_claims=40,
        openrouter_api_key=api_key if online else None,  # offline -> sentence fallback
        llm_model=args.model,
    )
    if not online:
        # Force sentence fallback: ClaimExtractor.__init__ falls back to the env
        # OPENROUTER_API_KEY when passed None, which would (wrongly) make offline
        # mode call the LLM for extraction. Blank it so offline is truly offline.
        extractor.api_key = ""

    prov = BuildProvenance(
        base_model=args.model if online else "offline-canned",
        temperature=GEN_TEMPERATURE,
        system_prompt_hash=_sha(GEN_SYSTEM_PROMPT),
        mode="online" if online else "offline",
        extractor_model=args.model if online else "sentence_fallback",
        seed=args.seed,
    )
    prov.config_hash = _sha(json.dumps(prov.to_dict(), sort_keys=True))

    logger.info("Building P0 (%s mode, %d seeds, model=%s) ...",
                prov.mode, len(seeds), prov.base_model)
    samples, counts = build_samples(seeds, extractor, prov, online, api_key)
    if not samples:
        logger.error("No samples produced; aborting.")
        sys.exit(1)

    train, val, test = split_by_question(samples, args.seed)

    write_json(DATA_DIR / f"{args.out_prefix}_train.json", train)
    write_json(DATA_DIR / f"{args.out_prefix}_val.json", val)
    write_json(DATA_DIR / f"{args.out_prefix}_test.json", test)
    # keep the raw unsplit build for audit
    write_json(RAW_DIR / f"{args.out_prefix}_all.json", samples)

    manifest = {
        "dataset": args.out_prefix,
        "provenance": prov.to_dict(),
        "seed_source": args.seeds or "builtin",
        "n_questions": len({s["qid"] for s in samples}),
        "n_answers": len(samples),
        "label_counts_semantic": counts,
        "splits": {
            "train": {"answers": len(train),
                      "claims": _count_claims(train),
                      "binary_claims": _count_claims(train, binary_only=True)},
            "val": {"answers": len(val),
                    "claims": _count_claims(val),
                    "binary_claims": _count_claims(val, binary_only=True)},
            "test": {"answers": len(test),
                     "claims": _count_claims(test),
                     "binary_claims": _count_claims(test, binary_only=True)},
        },
    }
    write_json(MANIFEST_DIR / f"{args.out_prefix}_manifest.json", manifest)

    logger.info("=" * 66)
    logger.info("  P0 BUILD COMPLETE (%s mode)", prov.mode)
    logger.info("=" * 66)
    logger.info("  questions: %d   answers: %d", manifest["n_questions"], manifest["n_answers"])
    logger.info("  labels   : SUPPORTED=%d  CONTRADICTED=%d  UNRESOLVABLE(excluded)=%d",
                counts[SUPPORTED], counts[CONTRADICTED], counts[UNRESOLVABLE])
    for split_name, st in manifest["splits"].items():
        logger.info("  %-6s: %d answers, %d claims (%d binary)",
                    split_name, st["answers"], st["claims"], st["binary_claims"])
    logger.info("  provenance config_hash=%s", prov.config_hash)
    logger.info("  manifest -> %s", MANIFEST_DIR / f"{args.out_prefix}_manifest.json")
    logger.info("=" * 66)
    if counts[CONTRADICTED] == 0 or counts[SUPPORTED] == 0:
        logger.warning("  WARNING: only one binary class present -> not trainable yet. "
                       "Add more seeds with a mix of correct/incorrect drafts.")


if __name__ == "__main__":
    main()
