"""Public agent surface — mirrors V1's ``DetectorAgent().detect(...)``.

The pipeline (and its models) is built ONCE per agent instance and reused across
calls, so nothing is reloaded per inference. ``detect`` returns the extended dict
(canonical six fields + claim detail); ``detect_output`` returns the rich pydantic
object; ``detect_canonical`` returns strictly the six fields.

Default detector (Stage 7): a bare ``DetectorAgent()`` wires the calibrated Stage-2
DeBERTa encoder via the existing :func:`build_stage7_agent` factory. The trained
encoder weights are **not vendored into git** (they are ~283 MB); instead they are
downloaded once from a pinned Hugging Face model revision and cached locally, so the
detector is self-contained and never depends on the sibling ``DetV2`` checkout. The
Stage-7 probability calibrator is small and stays vendored on disk in HalGuard.
The heavy encoder loads lazily on the first inference, so construction stays cheap.

Encoder resolution (no hardcoded paths):
  * ``DETECTORV2_HF_REPO_ID`` / ``DETECTORV2_HF_REVISION`` — the HF model repo and
    pinned commit the encoder is fetched from (defaults below).
  * ``DETECTORV2_ENCODER_DIR`` — optional local override; if it points at a real
    directory the HF download is skipped entirely (offline / air-gapped use).

Failure policy for the default path:
  * encoder **unresolvable** (no local override and the HF snapshot cannot be
    fetched) or calibrator **missing** -> honest fall back to the Stage-0 pipeline,
    which self-identifies as ``detector_v2_stage0_heuristic`` (never disguised as
    Stage 7, never the un-fine-tuned base weights);
  * artifacts **present but fail to load** -> raise loudly, so a deployment can
    never silently run the wrong detector.

Explicit injection is always honored: ``DetectorAgent(pipeline=...)`` uses the given
pipeline unchanged, and ``DetectorAgent(config=...)`` honors the supplied config
(e.g. ``stage="stage0_heuristic"``) instead of loading Stage 7.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .adapter import to_canonical_dict, to_extended_dict
from .config import DetectorV2Config
from .pipeline import DetectorV2Pipeline
from .schemas import DetectorV2Output

# Stage-7 artifact resolution — no hardcoded absolute paths, no sibling dependency.
#
# The probability calibrator is small and vendored on disk relative to this package's
# repo root (HalGuard), so it resolves regardless of the caller's CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CALIBRATOR = os.path.join(_REPO_ROOT, "models", "stage7_calibration", "calibrator.joblib")

# The trained Stage-2 encoder is fetched from a pinned Hugging Face model revision
# (weights are far too large to vendor into git). Repo id + revision are
# env-overridable but default to the pinned production revision.
_HF_REPO_ID = os.environ.get("DETECTORV2_HF_REPO_ID", "Lime379/detv2-stage2-encoder").strip()
_HF_REVISION = os.environ.get(
    "DETECTORV2_HF_REVISION", "9fb526aeddae8f3b140a463e8ef087a35e50abff"
).strip()
# Optional local encoder directory; when set to a real dir the HF download is skipped.
_ENCODER_DIR_OVERRIDE = os.environ.get("DETECTORV2_ENCODER_DIR", "").strip()


def _resolve_encoder_dir() -> Optional[str]:
    """Return a local directory holding the trained Stage-2 encoder, or ``None``.

    Resolution order: explicit ``DETECTORV2_ENCODER_DIR`` override (offline use)
    first, otherwise a cached Hugging Face snapshot at the pinned revision. A
    network/auth failure returns ``None`` (honest Stage-0 fallback), never the
    un-fine-tuned base weights.
    """
    if _ENCODER_DIR_OVERRIDE and os.path.isdir(_ENCODER_DIR_OVERRIDE):
        return _ENCODER_DIR_OVERRIDE
    if not _HF_REPO_ID:
        return None
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=_HF_REPO_ID, revision=_HF_REVISION or None)
    except Exception:  # noqa: BLE001 - unresolvable encoder -> honest Stage-0 fallback
        return None


def _default_stage7_pipeline() -> DetectorV2Pipeline:
    """Build the calibrated Stage-2 encoder + Stage-7 pipeline.

    Encoder unresolvable (no local override and HF snapshot cannot be fetched) or
    calibrator missing -> Stage-0 fallback (self-labeled, honest). Artifacts that
    exist but fail to load -> RuntimeError (never a silent wrong-detector run).
    """
    encoder_dir = _resolve_encoder_dir()
    calibrator_present = os.path.isfile(_CALIBRATOR)
    if not (encoder_dir and calibrator_present):
        # Genuinely unresolvable -> honest Stage-0 default (reports its own model_source).
        return DetectorV2Pipeline()
    try:
        from .calibration.build import build_stage7_agent
        from .calibration.calibrator import ProbabilityCalibrator

        calibrator = ProbabilityCalibrator.load(_CALIBRATOR)
        # Factory returns a DetectorAgent(pipeline=...); reuse just its pipeline.
        return build_stage7_agent(encoder_dir, calibrator).pipeline
    except Exception as exc:  # noqa: BLE001 - present-but-broken must fail loudly
        raise RuntimeError(
            "Stage-7 artifacts are present but failed to load "
            f"(encoder={encoder_dir}, calibrator={_CALIBRATOR}): "
            f"{type(exc).__name__}: {exc}. Refusing to silently fall back to Stage 0."
        ) from exc


class DetectorAgent:
    def __init__(self, config: Optional[DetectorV2Config] = None, pipeline: Optional[DetectorV2Pipeline] = None):
        if pipeline is not None:
            self.pipeline = pipeline                              # explicit DI, unchanged
        elif config is not None:
            self.pipeline = DetectorV2Pipeline(config=config)     # honor supplied config
        else:
            self.pipeline = _default_stage7_pipeline()            # bare agent -> Stage 7

    def detect_output(
        self, user_query: str, llm_response: str, context: Optional[str] = None,
        domain: str = "general", query_id: Optional[str] = None,
    ) -> DetectorV2Output:
        return self.pipeline.detect(user_query, llm_response, context, domain, query_id)

    def detect(self, user_query: str, llm_response: str, context: Optional[str] = None,
               domain: str = "general", query_id: Optional[str] = None) -> Dict[str, Any]:
        return to_extended_dict(self.detect_output(user_query, llm_response, context, domain, query_id))

    def detect_canonical(self, user_query: str, llm_response: str, context: Optional[str] = None,
                         domain: str = "general", query_id: Optional[str] = None) -> Dict[str, Any]:
        return to_canonical_dict(self.detect_output(user_query, llm_response, context, domain, query_id))
