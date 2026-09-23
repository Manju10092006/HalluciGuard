"""
scripts / run_live_detector.py
───────────────────────────────
Live end-to-end runner:
1. Takes a user query from command line arguments or interactive prompt.
2. Generates a draft answer from the Base LLM via OpenRouter.
3. Passes the answer into StandaloneDetector (halluciguard_detector) which
   runs the REAL DeBERTa-v3-base model over the extracted atomic claims.
4. Prints a formatted detection report with per-claim risk breakdown.

Config (all via environment / .env — no secrets in source):
    OPENROUTER_API_KEY        required
    OPENROUTER_MODEL          default: qwen/qwen3-14b
    DETECTOR_CHECKPOINT       default: microsoft/deberta-v3-base
                              (the untrained base model produces NOISE scores;
                               point this at a fine-tuned checkpoint for real
                               risk signal)

Usage:
    python halluciguard_detector/scripts/run_live_detector.py "your question"
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# UTF-8 stdout on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Load .env from the project root (parent of halluciguard_detector/)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import json
from halluciguard_detector.detector import StandaloneDetector
from halluciguard_detector.models.deberta import M2DebertaClassifier
from halluciguard_detector.calibration import Calibrator

# SECURITY: read the key from the environment only. Never hardcode it.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3-14b")
if OPENROUTER_MODEL in ("qwen/qwen3-4b", "qwen/qwen3-4b:free"):
    OPENROUTER_MODEL = "qwen/qwen3-14b"

# Default to the trained HaluEval v0.1 checkpoint if present; else the base model.
_ROOT = Path(__file__).parent.parent
_TRAINED = _ROOT / "data" / "checkpoints" / "deberta-halueval-v0.1"
DETECTOR_CHECKPOINT = os.environ.get(
    "DETECTOR_CHECKPOINT",
    str(_TRAINED) if _TRAINED.exists() else "microsoft/deberta-v3-base",
)
# Dev scores used to (re)fit the calibrator at startup. Platt = safer than the
# isotonic auto-pick on small data (isotonic overfits dev to ECE 0).
_DEV_SCORES = _ROOT / "data" / "processed" / "HALUEVAL_dev_scores.json"
CALIBRATION_METHOD = os.environ.get("CALIBRATION_METHOD", "platt")
# Frozen thresholds selected on dev (see calibration_fitted.json).
T_LOW = float(os.environ.get("DETECTOR_T_LOW", "0.5"))
T_HIGH = float(os.environ.get("DETECTOR_T_HIGH", "0.5"))


def _load_calibrator():
    """Fit a calibrator from the dumped dev scores, if available."""
    if not _DEV_SCORES.exists():
        return None
    try:
        rows = json.load(open(_DEV_SCORES, encoding="utf-8"))
        raw = [float(r["raw_score"]) for r in rows]
        y = [int(r["label"]) for r in rows]
        cal = Calibrator(method=CALIBRATION_METHOD, version="halueval-v0.1")
        cal.fit(raw, y, dataset_name="HALUEVAL_dev")
        return cal
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] calibrator fit failed ({exc}); running uncalibrated.")
        return None


def generate_llm_draft(user_query: str) -> str:
    """Generate a draft answer from the Base LLM via OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add it to your .env or environment."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://halluciguard.app",
        "X-Title": "HalluciGuard-Live-Test",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant. Answer the user's question concisely."},
            {"role": "user", "content": user_query},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def run_pipeline(user_query: str) -> None:
    print("=" * 80)
    print("                HALLUCIGUARD DETECTOR V2: LIVE PIPELINE RUN")
    print("=" * 80)
    print(f"USER QUERY: {user_query}")
    print(f"\n[Step 1] Calling Base LLM (OpenRouter: {OPENROUTER_MODEL}) ...")

    try:
        draft_answer = generate_llm_draft(user_query)
    except Exception as exc:
        print(f"\n[ERROR] LLM generation failed: {exc}")
        return
    print(f"\nBASE LLM DRAFT ANSWER:\n{draft_answer}")

    print(f"\n[Step 2] Loading REAL DeBERTa detector (checkpoint: {DETECTOR_CHECKPOINT}) ...")
    classifier = M2DebertaClassifier(checkpoint_path=DETECTOR_CHECKPOINT)
    model_ok = classifier.is_available()
    print(f"         Model loaded: {model_ok}  |  device: {classifier.device}")
    is_base = DETECTOR_CHECKPOINT.strip().lower() in ("microsoft/deberta-v3-base", "deberta-v3-base")
    if model_ok and is_base:
        print("         WARNING: this is the UNTRAINED base model. Raw scores are")
        print("                  NOISE and are intentionally reported as UNKNOWN.")

    calibrator = _load_calibrator()
    print(
        f"         Calibrator: {'fitted ('+CALIBRATION_METHOD+' on HaluEval dev)' if calibrator else 'none'}"
        f"  |  thresholds: t_low={T_LOW} t_high={T_HIGH}"
    )

    print("\n[Step 3] Running detector over extracted atomic claims ...")
    detector = StandaloneDetector(
        classifier=classifier, calibrator=calibrator, t_low=T_LOW, t_high=T_HIGH
    )
    resp = detector.detect(user_query=user_query, draft_answer=draft_answer, request_id="LIVE-RUN-001")

    print("\n" + "-" * 80)
    print("                            DETECTION RESULTS")
    print("-" * 80)
    print(f"Overall Risk Score : {resp.overall.risk_score:.4f}")
    print(f"Overall Risk Level : {resp.overall.risk_level.value}")
    print(f"Routing Decision   : {resp.routing.value}   (ALWAYS_VERIFY)")
    print(f"Extraction Mode    : {resp.extraction_mode.value}")
    print(f"Execution Status   : {resp.status.value}")
    print(f"Calibration        : {resp.calibration.method} (fitted_on={resp.calibration.fitted_on})")
    if resp.degraded_reasons:
        print(f"Degraded Reasons   : {', '.join(resp.degraded_reasons)}")
    print(
        f"Latency (ms)       : total={resp.latency_ms.total_ms:.1f}  "
        f"(extraction={resp.latency_ms.extraction_ms:.1f}, scoring={resp.latency_ms.scoring_ms:.1f})"
    )

    print(f"\nEXTRACTED ATOMIC CLAIMS & RISK TRIAGE ({len(resp.claims)} claims):")
    for idx, claim in enumerate(resp.claims, 1):
        cal = "n/a" if claim.calibrated_probability is None else f"{claim.calibrated_probability:.3f}"
        print(
            f"  {idx}. {claim.claim_id}  [{claim.risk_level.value:<9}] "
            f"[{claim.verification_hint.value:<8}]  raw={claim.raw_score:.3f}  cal={cal}"
        )
        print(f'     "{claim.text}"')
        if claim.flags:
            print(f"     flags: {claim.flags}")

    print("\nLEGACY COMPATIBILITY ADAPTER OUTPUT:")
    print(f"  {resp.to_legacy()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live hallucination detector test runner")
    parser.add_argument(
        "query",
        nargs="?",
        default="Who created the Java programming language and when was it released?",
        help="User query to test",
    )
    args = parser.parse_args()
    run_pipeline(args.query)
