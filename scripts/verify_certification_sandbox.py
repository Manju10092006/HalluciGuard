"""
HalluciGuard — Certification Guard Smoke Test (zero-dependency, runs anywhere).

Unlike the full regression suite (agents/verifier_agent/tests/
test_verifier_hardening_regression.py), this script imports **only the stdlib**
and api/certification.py, so it runs in any Python 3.8+ environment with no
torch / transformers / pydantic / pytest / network. It exercises every branch of
the fail-closed certification guards (§28/§29/§30) and prints a real PASS/FAIL
line per case.

Use it as an instant "are the guards wired correctly" check before the full
Windows live validation. It proves the guard *logic*; it does NOT prove real
model execution (that requires the live A-H run in LIVE_VALIDATION_WINDOWS.md).

    python scripts/verify_certification_sandbox.py

Exit code 0 = all guard cases passed; non-zero = at least one failed.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Make api/certification.py importable without pulling in the heavy pipeline.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERIFIER_DIR = PROJECT_ROOT / "agents" / "verifier_agent"
for _p in (str(VERIFIER_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from api.certification import (  # noqa: E402
    CertificationError,
    certification_enabled_from_env,
    enforce_detector,
    enforce_nli,
    enforce_reranker,
    enforce_retrieval_evidence,
)

_PASS = 0
_FAIL = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}{('  -> ' + detail) if detail else ''}")


def _raises(fn) -> bool:
    """True iff calling fn() raises CertificationError."""
    try:
        fn()
        return False
    except CertificationError:
        return True


def _stage_of(fn) -> str:
    try:
        fn()
    except CertificationError as exc:
        return exc.stage
    return ""


class _P:
    """Duck-typed passage with the attributes _looks_mock() inspects."""

    def __init__(self, url="", source="", title="", snippet=""):
        self.url = url
        self.source = source
        self.title = title
        self.snippet = snippet


def main() -> int:
    print("=" * 72)
    print("  CERTIFICATION GUARD SMOKE TEST (stdlib-only, in-sandbox)")
    print("=" * 72)

    # ── CertificationError shape ──────────────────────────────────────────
    print("\n[CertificationError]")
    err = CertificationError("detector", "boom")
    _check("stage attribute preserved", err.stage == "detector")
    _check("reason attribute preserved", err.reason == "boom")
    _check("message format [certification:stage] reason",
           str(err) == "[certification:detector] boom", str(err))

    # ── env flag parsing ──────────────────────────────────────────────────
    print("\n[certification_enabled_from_env]")
    for val, expect in (("true", True), ("1", True), ("yes", True), ("on", True),
                        ("TRUE", True), ("false", False), ("off", False), ("", False)):
        os.environ["CERTIFICATION_MODE"] = val
        _check(f"{val!r} -> {expect}", certification_enabled_from_env() is expect)
    os.environ.pop("CERTIFICATION_MODE", None)
    _check("unset -> False", certification_enabled_from_env() is False)

    # ── enforce_reranker ──────────────────────────────────────────────────
    print("\n[enforce_reranker]")
    _check("disabled is a no-op",
           not _raises(lambda: enforce_reranker({"status": "degraded"}, False)))
    _check("empty diag is a no-op",
           not _raises(lambda: enforce_reranker({}, True)))
    _check("executed+clean passes",
           not _raises(lambda: enforce_reranker({"status": "executed", "degraded": False}, True)))
    _check("not_run (no passages) allowed",
           not _raises(lambda: enforce_reranker({"status": "not_run"}, True)))
    _check("status=degraded FAILS",
           _raises(lambda: enforce_reranker({"status": "degraded"}, True)))
    _check("status=unavailable FAILS",
           _raises(lambda: enforce_reranker({"status": "unavailable"}, True)))
    _check("executed+degraded FAILS",
           _raises(lambda: enforce_reranker({"status": "executed", "degraded": True}, True)))
    _check("failure stage is bge_reranker",
           _stage_of(lambda: enforce_reranker({"status": "degraded"}, True)) == "bge_reranker")

    # ── enforce_nli (diag, enabled, evidence_present) ─────────────────────
    print("\n[enforce_nli]")
    good_nli = {"status": "executed", "inference_executed": True}
    _check("disabled is a no-op",
           not _raises(lambda: enforce_nli({"status": "degraded"}, False, True)))
    _check("no evidence present is a no-op",
           not _raises(lambda: enforce_nli({"status": "degraded"}, True, False)))
    _check("executed+evidence passes",
           not _raises(lambda: enforce_nli(good_nli, True, True)))
    _check("degraded+evidence FAILS",
           _raises(lambda: enforce_nli({"status": "degraded"}, True, True)))
    _check("unavailable+evidence FAILS",
           _raises(lambda: enforce_nli({"status": "unavailable"}, True, True)))
    _check("not-executed+evidence FAILS",
           _raises(lambda: enforce_nli({"status": "executed", "inference_executed": False}, True, True)))
    _check("failure stage is deberta_nli",
           _stage_of(lambda: enforce_nli({"status": "degraded"}, True, True)) == "deberta_nli")

    # ── enforce_retrieval_evidence (passages, enabled, mock_mode) ─────────
    print("\n[enforce_retrieval_evidence]")
    real = _P(url="https://en.wikipedia.org/wiki/Paris", source="wikipedia",
              title="Paris", snippet="Paris is the capital of France.")
    mock = _P(url="https://example.com/x", source="mock", title="t", snippet="s")
    malformed = _P(url="https://en.wikipedia.org/wiki/x", source="wikipedia")  # no title/snippet
    _check("disabled is a no-op",
           not _raises(lambda: enforce_retrieval_evidence([mock], False, False)))
    _check("mock_mode=True FAILS",
           _raises(lambda: enforce_retrieval_evidence([real], True, True)))
    _check("empty passages FAILS",
           _raises(lambda: enforce_retrieval_evidence([], True, False)))
    _check("all-mock FAILS",
           _raises(lambda: enforce_retrieval_evidence([mock, mock], True, False)))
    _check("all-malformed FAILS",
           _raises(lambda: enforce_retrieval_evidence([malformed], True, False)))
    _check("real evidence passes",
           not _raises(lambda: enforce_retrieval_evidence([real], True, False)))
    _check("real+mock mix passes (not ALL mock)",
           not _raises(lambda: enforce_retrieval_evidence([real, mock], True, False)))
    _check("failure stage is retrieval",
           _stage_of(lambda: enforce_retrieval_evidence([], True, False)) == "retrieval")

    # ── enforce_detector ──────────────────────────────────────────────────
    print("\n[enforce_detector]")
    real_det = {"detector_degraded": False, "detector_inference_executed": True}
    _check("disabled is a no-op",
           not _raises(lambda: enforce_detector({"detector_degraded": True}, False)))
    _check("empty result is a no-op",
           not _raises(lambda: enforce_detector({}, True)))
    _check("real inference passes",
           not _raises(lambda: enforce_detector(real_det, True)))
    _check("degraded FAILS",
           _raises(lambda: enforce_detector({"detector_degraded": True,
                                             "detector_inference_executed": True}, True)))
    _check("not-executed FAILS",
           _raises(lambda: enforce_detector({"detector_degraded": False,
                                             "detector_inference_executed": False}, True)))
    _check("failure stage is detector",
           _stage_of(lambda: enforce_detector({"detector_degraded": True}, True)) == "detector")

    # ── summary ────────────────────────────────────────────────────────────
    total = _PASS + _FAIL
    print("\n" + "=" * 72)
    print(f"  RESULT: {_PASS}/{total} cases passed, {_FAIL} failed")
    print("=" * 72)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001 - surface unexpected import/logic errors
        traceback.print_exc()
        raise SystemExit(2)
