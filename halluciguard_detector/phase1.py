"""Optional, generation-bound Phase 1 uncertainty head.

This is a documented logit-feature adaptation, not a reproduction of the
attention-based LUH architecture.  No probability exists without a compatible
trace, a trained checkpoint, and a held-out calibration artifact.
"""
from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


FEATURE_SCHEMA = "hg-token-logits-v2"
FEATURE_NAMES = ("mean_nll", "max_nll", "mean_entropy", "mean_margin", "log_token_count")
LABELS = {"SUPPORTED": 0, "HALLUCINATION_RELATED": 1}


class TokenFeature(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    nll: float = Field(ge=0)
    entropy: float = Field(ge=0)
    margin: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def valid(self):
        if self.end <= self.start or not all(math.isfinite(x) for x in (self.nll, self.entropy, self.margin)):
            raise ValueError("token offsets and features must be finite and increasing")
        return self


class GenerationTrace(BaseModel):
    schema_version: Literal["hg-token-logits-v2"] = FEATURE_SCHEMA
    source: Literal["local_transformers_generate_raw_logits"]
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    complete: bool = True
    tokens: list[TokenFeature] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self):
        if any(b.start < a.end for a, b in zip(self.tokens, self.tokens[1:])):
            raise ValueError("overlapping generation-token offsets")
        return self


class Phase1Config(BaseModel):
    mode: Literal["disabled", "shadow", "fast_path"] = "disabled"
    model_id: str = ""
    model_revision: str = ""
    tokenizer_id: str = ""
    tokenizer_revision: str = ""
    head_dir: Path = Path("artifacts/phase1-uq")
    allowed_domains: tuple[str, ...] = ("general",)

    @classmethod
    def from_env(cls) -> "Phase1Config":
        return cls(
            mode=os.getenv("HALLUCIGUARD_PHASE1_MODE", "disabled"),
            model_id=os.getenv("HALLUCIGUARD_PHASE1_MODEL_ID", ""),
            model_revision=os.getenv("HALLUCIGUARD_PHASE1_MODEL_REVISION", ""),
            tokenizer_id=os.getenv("HALLUCIGUARD_PHASE1_TOKENIZER_ID", ""),
            tokenizer_revision=os.getenv("HALLUCIGUARD_PHASE1_TOKENIZER_REVISION", ""),
            head_dir=Path(os.getenv("HALLUCIGUARD_PHASE1_HEAD_DIR", "artifacts/phase1-uq")),
            allowed_domains=tuple(x.strip() for x in os.getenv("HALLUCIGUARD_PHASE1_DOMAINS", "general").split(",") if x.strip()),
        )


def claim_features(trace: GenerationTrace, answer: str, start: int, end: int) -> list[float]:
    """Aggregate genuine generated-token signals only for a fully aligned claim."""
    if end <= start:
        raise ValueError("empty claim")
    covered = [t for t in trace.tokens if t.end > start and t.start < end]
    if not covered or covered[0].start > start or covered[-1].end < end:
        raise ValueError("claim boundary is not aligned to generated tokens")
    if answer[covered[0].start:start].strip() or answer[end:covered[-1].end].strip():
        raise ValueError("claim boundary cuts through a generated token")
    if any(b.start != a.end for a, b in zip(covered, covered[1:])):
        raise ValueError("claim token coverage has gaps")
    n = len(covered)
    return [
        sum(t.nll for t in covered) / n,
        max(t.nll for t in covered),
        sum(t.entropy for t in covered) / n,
        sum(t.margin for t in covered) / n,
        math.log1p(n),
    ]


@dataclass(frozen=True)
class Phase1Result:
    status: str
    reason: str | None = None
    mode: str = "disabled"
    raw_scores: tuple[float, ...] = ()
    calibrated_risks: tuple[float, ...] = ()
    response_risk: float | None = None
    bypass_eligible: bool = False
    head_id: str | None = None
    calibration_id: str | None = None
    target: str = "P(HALLUCINATION_RELATED label | exact generator features)"
    aggregation: str = "max_claim_risk"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "reason": self.reason, "mode": self.mode,
            "raw_scores": list(self.raw_scores),
            "calibrated_risks": list(self.calibrated_risks),
            "response_risk": self.response_risk,
            "bypass_eligible": self.bypass_eligible,
            "head_id": self.head_id, "calibration_id": self.calibration_id,
            "target": self.target, "aggregation": self.aggregation,
        }


class Phase1Service:
    """Lazy, cached inference with exact generator/checkpoint compatibility."""

    def __init__(self, config: Phase1Config | None = None):
        self.config = config or Phase1Config.from_env()
        self._lock = threading.Lock()
        self._head = None
        self._metadata: dict[str, Any] | None = None
        self._calibration: dict[str, Any] | None = None

    def _load(self) -> None:
        if self._head is not None:
            return
        with self._lock:
            if self._head is not None:
                return
            import torch
            from safetensors.torch import load_file

            folder = self.config.head_dir
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            calibration = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
            if meta.get("feature_schema") != FEATURE_SCHEMA or meta.get("feature_names") != list(FEATURE_NAMES):
                raise ValueError("checkpoint feature schema mismatch")
            for key in ("feature_mean", "feature_std"):
                values = meta.get(key)
                if not isinstance(values, list) or len(values) != len(FEATURE_NAMES) or not all(
                    isinstance(value, (int, float)) and math.isfinite(value) and
                    (key != "feature_std" or value > 0) for value in values
                ):
                    raise ValueError(f"invalid checkpoint {key}")
            if calibration.get("head_sha256") != meta.get("head_sha256"):
                raise ValueError("calibration does not match checkpoint")
            if meta.get("schema") != "hg-phase1-head-v1" or calibration.get("schema") != "hg-phase1-calibration-v1":
                raise ValueError("checkpoint or calibration schema mismatch")
            import hashlib
            calibration_id = calibration.get("calibration_id")
            calibration_body = {key: value for key, value in calibration.items() if key != "calibration_id"}
            expected_calibration_id = hashlib.sha256(
                json.dumps(calibration_body, sort_keys=True).encode()
            ).hexdigest()
            if calibration_id != expected_calibration_id:
                raise ValueError("calibration artifact integrity mismatch")
            temperature = float(calibration.get("temperature", 0))
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("calibration is missing or invalid")
            threshold = calibration.get("threshold")
            if calibration.get("release_validated") and (
                not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or not 0 <= threshold <= 1
            ):
                raise ValueError("invalid release threshold")
            head_path = folder / "head.safetensors"
            if hashlib.sha256(head_path.read_bytes()).hexdigest() != meta["head_sha256"]:
                raise ValueError("checkpoint checksum mismatch")
            head = torch.nn.Linear(len(FEATURE_NAMES), 1)
            head.load_state_dict(load_file(str(head_path)))
            head.eval()
            self._metadata, self._calibration, self._head = meta, calibration, head

    def score(self, answer: str, claims: list[dict[str, Any]], trace_data: dict[str, Any] | None, domain: str = "general") -> Phase1Result:
        cfg = self.config
        if cfg.mode == "disabled":
            return Phase1Result(status="disabled", mode=cfg.mode)
        if not trace_data:
            return Phase1Result(status="unavailable", reason="missing_generation_trace", mode=cfg.mode)
        try:
            import hashlib
            import torch
            trace = GenerationTrace.model_validate(trace_data)
            if not trace.complete or hashlib.sha256(answer.encode("utf-8")).hexdigest() != trace.text_sha256:
                raise ValueError("generation trace does not match response")
            expected = (cfg.model_id, cfg.model_revision, cfg.tokenizer_id, cfg.tokenizer_revision)
            actual = (trace.model_id, trace.model_revision, trace.tokenizer_id, trace.tokenizer_revision)
            if not all(expected) or actual != expected:
                return Phase1Result(status="unavailable", reason="incompatible_generator", mode=cfg.mode)
            if domain not in cfg.allowed_domains:
                return Phase1Result(status="unavailable", reason="unevaluated_domain", mode=cfg.mode)
            features = [claim_features(trace, answer, int(c["start"]), int(c["end"])) for c in claims]
            if not features:
                raise ValueError("no aligned factual claims")
            self._load()
            assert self._head is not None and self._metadata is not None and self._calibration is not None
            meta = self._metadata
            if actual != tuple(meta.get(k) for k in ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision")):
                return Phase1Result(status="unavailable", reason="incompatible_checkpoint", mode=cfg.mode)
            mean, std = meta["feature_mean"], meta["feature_std"]
            matrix = torch.tensor([[((x - m) / s) for x, m, s in zip(row, mean, std)] for row in features], dtype=torch.float32)
            with torch.inference_mode():
                raw = self._head(matrix).flatten().tolist()
            if not all(math.isfinite(x) for x in raw):
                raise ValueError("non-finite UQ output")
            temp = float(self._calibration["temperature"])
            risks = [1 / (1 + math.exp(-max(-80.0, min(80.0, x / temp)))) for x in raw]
            risk = max(risks)
            bypass = (
                cfg.mode == "fast_path"
                and meta.get("label_unit") == "sentence"
                and bool(self._calibration["release_validated"])
                and risk <= float(self._calibration["threshold"])
            )
            return Phase1Result(
                status="scored", mode=cfg.mode, raw_scores=tuple(raw),
                calibrated_risks=tuple(risks), response_risk=risk,
                bypass_eligible=bypass, head_id=meta["head_sha256"],
                calibration_id=self._calibration.get("calibration_id"),
            )
        except FileNotFoundError:
            return Phase1Result(status="unavailable", reason="missing_head_or_calibration", mode=cfg.mode)
        except Exception as exc:
            return Phase1Result(status="unavailable", reason=type(exc).__name__, mode=cfg.mode)
