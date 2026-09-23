"""
scripts / eval_and_dump_scores.py
─────────────────────────────────
Independent verification of a trained checkpoint:
1. Load the saved checkpoint via the REAL M2DebertaClassifier (same path the
   detector uses at inference).
2. Score the HaluEval dev + frozen test splits.
3. Report AUROC / F1 on each (test = honest generalization within HaluEval).
4. Dump per-example {raw_score, label} JSON for dev + test so the calibrator
   can be fitted on DEV and evaluated ONCE on TEST.

No fabricated scores: everything comes from the real model forward pass.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from halluciguard_detector.models.deberta import M2DebertaClassifier

DATA = Path(__file__).parent.parent / "data" / "processed"
CKPT = Path(__file__).parent.parent / "data" / "checkpoints" / "deberta-halueval-v0.1"


def score_split(clf: M2DebertaClassifier, name: str):
    rows = json.load(open(DATA / f"HALUEVAL_{name}.json", encoding="utf-8"))
    labels = [int(r["label"]) for r in rows]
    scores = []
    # Score one at a time via the exact inference contract (claim vs [Q][A]).
    for r in rows:
        s = clf.predict_claim_risk(r["query"], r["answer"], [r["claim"]])
        scores.append(float(s[0]))
    y = np.array(labels)
    p = np.array(scores)
    auroc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    f1 = f1_score(y, (p >= 0.5).astype(int), zero_division=0)
    dump = [{"raw_score": round(float(a), 6), "label": int(b)} for a, b in zip(scores, labels)]
    return auroc, f1, dump


def main() -> int:
    print(f"[eval] loading checkpoint: {CKPT}")
    clf = M2DebertaClassifier(checkpoint_path=str(CKPT))
    if not clf.is_available():
        print("[eval] ERROR: checkpoint failed to load")
        return 1
    print(f"[eval] model loaded on {clf.device}")

    for name in ("dev", "test"):
        auroc, f1, dump = score_split(clf, name)
        out = DATA / f"HALUEVAL_{name}_scores.json"
        json.dump(dump, open(out, "w", encoding="utf-8"), indent=1)
        tag = "FROZEN TEST" if name == "test" else "dev"
        print(f"[eval] {tag:12s}: AUROC={auroc:.4f}  F1={f1:.4f}  (n={len(dump)}) -> {out.name}")

    print("[eval] done. Scores dumped for calibration (fit on dev, evaluate once on test).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
