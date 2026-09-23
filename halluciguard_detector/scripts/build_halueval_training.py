"""
scripts / build_halueval_training.py
────────────────────────────────────
Build a REAL claim-level training set from HaluEval QA for the v0.1 diagnostic
DeBERTa checkpoint.

HONEST SCOPE (per handoff spec §9): HaluEval is SUPPLEMENTARY/diagnostic data.
Training on it alone teaches "what a HaluEval hallucination looks like", not the
full HalluciGuard production distribution. This produces a v0.1 checkpoint that
yields MEANINGFUL (non-noise) scores to prove the train->calibrate loop — it is
NOT a validated production detector. P0/P1/P4/P5 remain to be built.

Each HaluEval QA row -> two examples:
    right_answer        -> label 0 (low verification risk)
    hallucinated_answer -> label 1 (high verification risk)

Leakage guard: split BY QUESTION. All examples from one question land in the
same split, so no question appears in both train and test.

Output records match the detector's scoring contract:
    {"query", "answer", "claim", "label"}
where claim == answer (HaluEval answers are single atomic statements).
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).parent.parent / "data" / "processed"
SOURCE = "pminervini/HaluEval"


def build(n_questions: int, seed: int = 42):
    ds = load_dataset(SOURCE, "qa", split="data", streaming=True)

    rows = []
    for i, row in enumerate(ds):
        if i >= n_questions:
            break
        q = (row.get("question") or "").strip()
        right = (row.get("right_answer") or "").strip()
        hallu = (row.get("hallucinated_answer") or "").strip()
        if not q or not right or not hallu:
            continue
        rows.append((q, right, hallu))

    # Split BY QUESTION to prevent leakage.
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    n_test = max(1, int(n * 0.15))
    n_dev = max(1, int(n * 0.15))
    test_q = rows[:n_test]
    dev_q = rows[n_test : n_test + n_dev]
    train_q = rows[n_test + n_dev :]

    def to_examples(question_rows):
        out = []
        for q, right, hallu in question_rows:
            out.append({"query": q, "answer": right, "claim": right, "label": 0})
            out.append({"query": q, "answer": hallu, "claim": hallu, "label": 1})
        rng.shuffle(out)
        return out

    return to_examples(train_q), to_examples(dev_q), to_examples(test_q)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=4000,
                    help="HaluEval QA questions to pull (each -> 2 examples)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, dev, test = build(args.n_questions, args.seed)

    for name, data in (("train", train), ("dev", dev), ("test", test)):
        path = OUT_DIR / f"HALUEVAL_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        pos = sum(d["label"] for d in data)
        print(f"[build] HALUEVAL_{name}.json: {len(data)} examples ({pos} pos / {len(data)-pos} neg) -> {path}")

    print("[build] done. Leakage guard: split by question, no question spans splits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
