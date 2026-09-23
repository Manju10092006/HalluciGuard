"""
scripts / build_p0_dataset.py
─────────────────────────────
Build the P0 controlled-factual-QA training set from TriviaQA, using the REAL
production Base LLM and the REAL claim extractor (handoff spec §4-6).

Pipeline per question:
    TriviaQA question (+ gold answer & aliases)
        -> Base LLM (OpenRouter, same production model) generates a draft answer
        -> ClaimExtractor extracts atomic claims
        -> each claim labeled by an LLM JUDGE against the GOLD answer as trusted
           reference (SUPPORTED / CONTRADICTED / UNRESOLVABLE)
        -> risk label: SUPPORTED=0, CONTRADICTED=1, UNRESOLVABLE=dropped

WHY the gold answer (not the Verifier): TriviaQA gold answers are real dataset
ground truth. The judge only checks consistency of a claim against that gold
reference — it does NOT search, retrieve, or reuse HalluciGuard's Verifier
(which spec §14 forbids as a label source). This is noisier than human
annotation (that's P4's job) but gives genuine, non-fabricated P0 labels.

Resumable: appends one JSON line per question to <out>/P0_raw.jsonl and skips
question_ids already present. Re-run to accumulate more. Then --finalize splits
by question into train/dev/test.

Usage:
    # generate 50 questions (pilot):
    python halluciguard_detector/scripts/build_p0_dataset.py --n 50
    # split what's been accumulated:
    python halluciguard_detector/scripts/build_p0_dataset.py --finalize
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import httpx
from halluciguard_detector.claim_extraction import ClaimExtractor

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3-14b")
# The .env ships a stale/unserved model id; remap to the working reasoning model.
if BASE_MODEL in ("qwen/qwen3-4b", "qwen/qwen3-4b:free"):
    BASE_MODEL = "qwen/qwen3-14b"
JUDGE_MODEL = os.environ.get("P0_JUDGE_MODEL", BASE_MODEL)
if JUDGE_MODEL in ("qwen/qwen3-4b", "qwen/qwen3-4b:free"):
    JUDGE_MODEL = "qwen/qwen3-14b"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_PATH = OUT_DIR / "P0_raw.jsonl"
_URL = "https://openrouter.ai/api/v1/chat/completions"


def _chat(model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://halluciguard.app",
        "X-Title": "HalluciGuard-P0-Build",
    }
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=90.0) as client:
        r = client.post(_URL, headers=headers, json=payload)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        # Reasoning models (qwen3) may put the answer in content; guard None.
        return (msg.get("content") or "").strip()


def generate_answer(question: str) -> str:
    # qwen3 is a reasoning model: hidden thinking consumes max_tokens before any
    # content. Give ample headroom or content comes back empty.
    return _chat(
        BASE_MODEL,
        "You are a helpful AI assistant. Answer the user's question concisely and factually.",
        question,
        temperature=0.7,
        max_tokens=2048,
    )


JUDGE_SYSTEM = (
    "You label whether a CLAIM is consistent with a trusted GOLD ANSWER to a "
    "question. Use ONLY the gold answer as your reference — do not use outside "
    "knowledge, do not search. Respond with STRICT JSON: "
    '{"label": "SUPPORTED|CONTRADICTED|UNRESOLVABLE"}. '
    "SUPPORTED = the claim agrees with the gold answer. "
    "CONTRADICTED = the claim conflicts with the gold answer. "
    "UNRESOLVABLE = the gold answer does not address the claim."
)


def judge_claim(question: str, gold: list[str], claim: str) -> str:
    gold_str = " | ".join(gold)
    user = (
        f"[QUESTION] {question}\n[GOLD ANSWER(S)] {gold_str}\n[CLAIM] {claim}\n"
        "Return the strict JSON label now."
    )
    try:
        # Reasoning model needs room for hidden reasoning BEFORE the JSON.
        raw = _chat(JUDGE_MODEL, JUDGE_SYSTEM, user, temperature=0.0, max_tokens=2000)
        s, e = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[s : e + 1])
        label = str(data.get("label", "")).upper().strip()
        return label if label in ("SUPPORTED", "CONTRADICTED", "UNRESOLVABLE") else "UNRESOLVABLE"
    except Exception:
        return "UNRESOLVABLE"


def _done_ids() -> set[str]:
    if not RAW_PATH.exists():
        return set()
    ids = set()
    with open(RAW_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["question_id"])
            except Exception:
                pass
    return ids


def generate(n: int, seed: int, sleep: float) -> None:
    if not OPENROUTER_API_KEY:
        print("[p0] ERROR: OPENROUTER_API_KEY not set (.env).")
        return
    from datasets import load_dataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = _done_ids()
    print(f"[p0] already done: {len(done)} questions. Target this run: {n} new.")

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True)
    extractor = ClaimExtractor(max_claims=40)

    made = 0
    for row in ds:
        if made >= n:
            break
        qid = row.get("question_id", "")
        if qid in done:
            continue
        question = (row.get("question") or "").strip()
        ans = row.get("answer") or {}
        gold = [ans.get("value", "")] + list(ans.get("aliases", []))
        gold = [g for g in gold if g]
        if not question or not gold:
            continue

        try:
            answer = generate_answer(question)
            if not answer.strip():
                print(f"[p0] skip {qid}: empty answer (model returned no content)")
                continue
            claims, _mode, _reasons = extractor.extract(user_query=question, draft_answer=answer)
            claim_records = []
            for c in claims:
                label = judge_claim(question, gold, c.text)
                claim_records.append({"claim_id": c.claim_id, "text": c.text, "judge": label})
        except Exception as exc:
            print(f"[p0] skip {qid}: {exc}")
            continue

        rec = {
            "question_id": qid,
            "query": question,
            "answer": answer,
            "gold": gold,
            "base_model": BASE_MODEL,
            "judge_model": JUDGE_MODEL,
            "claims": claim_records,
        }
        with open(RAW_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        made += 1
        n_sup = sum(1 for c in claim_records if c["judge"] == "SUPPORTED")
        n_con = sum(1 for c in claim_records if c["judge"] == "CONTRADICTED")
        print(f"[p0] {made}/{n} {qid}: {len(claim_records)} claims (sup={n_sup} con={n_con})")
        if sleep:
            time.sleep(sleep)

    print(f"[p0] done. total questions in {RAW_PATH.name}: {len(_done_ids())}")


def finalize(seed: int) -> None:
    if not RAW_PATH.exists():
        print("[p0] no raw file to finalize.")
        return
    questions = []
    with open(RAW_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                questions.append(json.loads(line))
            except Exception:
                pass

    # SUPPORTED=0, CONTRADICTED=1, UNRESOLVABLE dropped (spec §6).
    def examples_for(q):
        out = []
        for c in q["claims"]:
            if c["judge"] == "SUPPORTED":
                label = 0
            elif c["judge"] == "CONTRADICTED":
                label = 1
            else:
                continue
            out.append({"query": q["query"], "answer": q["answer"], "claim": c["text"], "label": label})
        return out

    rng = random.Random(seed)
    rng.shuffle(questions)
    n = len(questions)
    n_test = max(1, int(n * 0.15))
    n_dev = max(1, int(n * 0.15))
    splits = {
        "test": questions[:n_test],
        "dev": questions[n_test : n_test + n_dev],
        "train": questions[n_test + n_dev :],
    }
    for name, qs in splits.items():
        rows = []
        for q in qs:
            rows.extend(examples_for(q))
        rng.shuffle(rows)
        path = OUT_DIR / f"P0_{name}.json"
        json.dump(rows, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        pos = sum(r["label"] for r in rows)
        print(f"[p0] P0_{name}.json: {len(rows)} examples ({pos} pos / {len(rows)-pos} neg) from {len(qs)} questions")
    print("[p0] finalized. Split BY QUESTION — no question spans splits.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="new questions to generate this run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between questions (rate limit)")
    ap.add_argument("--finalize", action="store_true", help="split accumulated raw into train/dev/test")
    args = ap.parse_args()

    if args.finalize:
        finalize(args.seed)
    else:
        generate(args.n, args.seed, args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
