"""Regression: the pipeline must use ONE canonical NLIEngine that exposes diagnostics().

Guards the exact live crash that motivated this fix:

    AttributeError: 'NLIEngine' object has no attribute 'diagnostics'
    at api/pipeline.py  ->  self._last_nli_diag = self.nli_engine.diagnostics()

Root cause was an implementation/import mismatch: ``nli/__init__.py`` re-exported the
robust engine (``nli.robust_entailment.NLIEngine``) while the §15 ``diagnostics()``
execution-proof contract lived only on the *other* engine (``nli.entailment.NLIEngine``).
The canonical pipeline engine (robust_entailment) now carries the same diagnostics
contract, so the pipeline binds one engine that both makes robust decisions AND proves
its own execution.

These tests assert the property that was previously untested: that the engine actually
attached to a live ``VerificationPipeline`` (``pipeline.nli_engine``) — not some other
NLIEngine class — exposes and can execute ``diagnostics()``.

Like the rest of the hardening suite, model/pydantic/pipeline imports are guarded so this
file runs fully on the Windows validation host and skips cleanly in the dependency-light
sandbox (no torch / transformers / pydantic).
"""
from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
VERIFIER_DIR = os.path.join(PROJECT_ROOT, "agents", "verifier_agent")
for _p in (PROJECT_ROOT, VERIFIER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- Pydantic / model-dependent imports (guarded) ---------------------------
try:
    # The canonical engine as the pipeline imports it (`from nli import NLIEngine`).
    from nli import NLIEngine
    # The concrete production-safe implementation it must resolve to.
    from nli.robust_entailment import NLIEngine as RobustNLIEngine
    from schemas.models import EntailmentLabel
    from schemas.retrieval_trace import ModelExecutionTrace

    _HAS_MODELS = True
    _MODELS_ERR = ""
except Exception as _e:  # pragma: no cover - environment dependent
    _HAS_MODELS = False
    _MODELS_ERR = repr(_e)

try:
    from api.pipeline import VerificationPipeline

    _HAS_PIPELINE = True
    _PIPELINE_ERR = ""
except Exception as _e:  # pragma: no cover - environment dependent
    _HAS_PIPELINE = False
    _PIPELINE_ERR = repr(_e)

needs_models = pytest.mark.skipif(not _HAS_MODELS, reason=f"model deps unavailable: {_MODELS_ERR}")
needs_pipeline = pytest.mark.skipif(not _HAS_PIPELINE, reason=f"pipeline deps unavailable: {_PIPELINE_ERR}")

# §8 required diagnostics contract (keys + fresh-engine "not_run" defaults).
_EXPECTED_DIAGNOSTICS = {
    "component": "deberta_nli",
    "model": "cross-encoder/nli-deberta-v3-base",
    "loaded": False,
    "inference_executed": False,
    "degraded": False,
    "status": "not_run",
    "device": "unknown",
    "latency_ms": 0,
    "batch_size": 0,
}


class _FakeNLIPipeline:
    """Stand-in for a loaded HF text-classification pipeline (no torch, no network)."""

    def __init__(self, rows, device: str = "cpu", raise_on_call: bool = False) -> None:
        self._rows = list(rows)
        self.device = device
        self._raise = raise_on_call

    def __call__(self, batch):
        if self._raise:
            raise RuntimeError("simulated NLI inference failure")
        # One list-of-label-dicts per input pair.
        return [list(self._rows) for _ in batch]


# ===========================================================================
# Step 12 — the pipeline binds ONE canonical engine that exposes diagnostics()
# ===========================================================================
@needs_pipeline
def test_pipeline_nli_engine_is_canonical_and_exposes_diagnostics():
    pipeline = VerificationPipeline()
    engine = pipeline.nli_engine

    # It is the canonical engine exported by `from nli import NLIEngine` ...
    assert isinstance(engine, NLIEngine)
    # ... concretely the production-safe robust implementation, not a competing class.
    assert isinstance(engine, RobustNLIEngine)
    assert type(engine).__module__.endswith("robust_entailment"), type(engine).__module__

    # The exact attribute whose absence crashed the live pipeline.
    assert hasattr(engine, "diagnostics") is True
    assert callable(engine.diagnostics)

    # And calling it must not raise (this is what threw AttributeError before the fix).
    diag = engine.diagnostics()
    assert isinstance(diag, dict)
    assert diag["component"] == "deberta_nli"
    # The pipeline unpacks it straight into a ModelExecutionTrace — prove that works too.
    assert ModelExecutionTrace(**diag).component == "deberta_nli"


# ===========================================================================
# Step 13 — the diagnostics output structure (contract + trace round-trip)
# ===========================================================================
@needs_models
def test_canonical_nli_diagnostics_contract_structure():
    engine = NLIEngine()  # construction is lightweight; no model load happens here
    diag = engine.diagnostics()

    # Full §8 contract, exact keys and fresh-engine values.
    assert diag == _EXPECTED_DIAGNOSTICS

    # Types (guards against, e.g., `loaded` leaking a truthy non-bool).
    assert isinstance(diag["loaded"], bool)
    assert isinstance(diag["inference_executed"], bool)
    assert isinstance(diag["degraded"], bool)
    assert isinstance(diag["status"], str)
    assert isinstance(diag["device"], str)
    assert isinstance(diag["latency_ms"], int)
    assert isinstance(diag["batch_size"], int)

    # Every key must be a valid ModelExecutionTrace field so `ModelExecutionTrace(**diag)`
    # (api/pipeline.py) can never raise on an unexpected key.
    allowed = set(ModelExecutionTrace.model_fields.keys())
    assert set(diag).issubset(allowed), set(diag) - allowed

    met = ModelExecutionTrace(**diag)
    assert met.component == "deberta_nli"
    assert met.model == "cross-encoder/nli-deberta-v3-base"
    assert met.status == "not_run"
    assert met.inference_executed is False
    assert met.degraded is False


@needs_models
def test_canonical_engine_records_execution_proof_across_states():
    """The ported §15 instrumentation must actually work on the *canonical* engine,
    not merely exist on the other one. Torch-free via a fake HF pipeline."""

    # not_run: an empty call resets to not_run and returns nothing.
    engine = NLIEngine()
    assert engine.batch_classify("claim", []) == []
    assert engine.diagnostics()["status"] == "not_run"

    # unavailable: model could not load -> fail-soft neutrals, flagged (never masqueraded).
    engine = NLIEngine()
    engine._is_available = False
    results = engine.batch_classify("claim", ["ev one", "ev two"])
    diag = engine.diagnostics()
    assert diag["status"] == "unavailable"
    assert diag["inference_executed"] is False
    assert diag["degraded"] is True
    assert len(results) == 2 and all(r["degraded"] is True for r in results)

    # executed: a real forward pass is recorded with batch size and device.
    engine = NLIEngine()
    engine.pipeline = _FakeNLIPipeline(
        rows=[
            {"label": "entailment", "score": 0.8},
            {"label": "contradiction", "score": 0.1},
            {"label": "neutral", "score": 0.1},
        ],
        device="cpu",
    )
    engine._is_available = True  # pipeline present -> _load_model() is a no-op
    results = engine.batch_classify("claim", ["ev one", "ev two"])
    diag = engine.diagnostics()
    assert diag["status"] == "executed"
    assert diag["inference_executed"] is True
    assert diag["degraded"] is False
    assert diag["batch_size"] == 2
    assert diag["device"] == "cpu"
    # robust_entailment._decision (unchanged production code) returns real scored
    # results WITHOUT a per-item "degraded" key on the success path; only the
    # _neutral() fallback carries degraded=True. So "not degraded" here means the
    # key is absent/falsey — which is exactly how the pipeline reads it
    # (result.get("degraded", False)). Asserting r["degraded"] would be wrong.
    assert all(r.get("degraded", False) is False for r in results)
    # Robust decision policy is preserved (strong entailment margin -> ENTAILMENT).
    assert results[0]["label"] == EntailmentLabel.ENTAILMENT

    # degraded: an inference exception is flagged, not silently presented as real.
    engine = NLIEngine()
    engine.pipeline = _FakeNLIPipeline(rows=[], raise_on_call=True)
    engine._is_available = True
    results = engine.batch_classify("claim", ["ev"])
    diag = engine.diagnostics()
    assert diag["status"] == "degraded"
    assert diag["inference_executed"] is False
    assert diag["degraded"] is True
    assert len(results) == 1
