from __future__ import annotations

import logging
import hashlib
import copy
import math
import time
from typing import Any, Dict, List, Optional

from schemas.models import EntailmentLabel
from models.model_manager import get_model_manager

logger = logging.getLogger(__name__)


def _canonical_label(
    raw_label: Any, id2label: Optional[Dict[Any, str]] = None
) -> Optional[str]:
    label = str(raw_label or "").strip().lower()
    # Tolerate common head decorations (e.g. "entailment_score", "contradiction_label")
    # by stripping only KNOWN suffixes -- not loose substring matching, which HG-007
    # removed (audit #42). The pinned model emits clean labels; this only hardens
    # against a differently-decorated head without widening the match surface.
    for _suffix in ("_score", "_label", "_prob", "_probability", "_logit"):
        if label.endswith(_suffix):
            label = label[: -len(_suffix)]
            break
    if label in {"entailment", "entails", "supported", "supports", "support"}:
        return "entailment"
    if label in {"contradiction", "contradicted", "contradicts", "refuted", "refutes", "refutation"}:
        return "contradiction"
    if label in {"neutral", "not_enough_info"}:
        return "neutral"
    if label.startswith("label_"):
        try:
            index = int(label.split("_", 1)[1])
        except (ValueError, IndexError):
            return None
        mapped = id2label.get(index) if id2label else None
        if mapped is None and id2label:
            mapped = id2label.get(str(index))
        if mapped:
            return _canonical_label(mapped, None)
        return {0: "contradiction", 1: "entailment", 2: "neutral"}.get(index)
    return None


def _flatten_predictions(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if not isinstance(raw, list):
        return []
    if raw and isinstance(raw[0], dict):
        return raw
    flattened: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            flattened.append(item)
        elif isinstance(item, list):
            flattened.extend(x for x in item if isinstance(x, dict))
    return flattened


def _normalize_scores(
    items: List[Dict[str, Any]],
    id2label: Optional[Dict[Any, str]] = None,
) -> Dict[str, float]:
    scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
    seen = set()
    for item in items:
        label = _canonical_label(item.get("label"), id2label)
        if label is None:
            raise ValueError("Unknown NLI label")
        try:
            score = float(item["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Missing or invalid NLI score") from exc
        if label in seen or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Duplicate label or invalid NLI probability")
        seen.add(label)
        scores[label] = score
    total = sum(scores.values())
    if seen != set(scores) or not math.isclose(total, 1.0, abs_tol=1e-5):
        raise ValueError("Incomplete or inconsistent NLI probabilities")
    scores = {key: value / total for key, value in scores.items()}
    return scores


def _decision(scores: Dict[str, float]) -> Dict[str, Any]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ordered[0]
    second_score = ordered[1][1]
    label = (
        EntailmentLabel.NEUTRAL
        if top_score < 0.45 or (top_score - second_score) < 0.05
        else EntailmentLabel(top_label)
    )
    return {
        "label": label,
        "entailment_score": round(scores["entailment"], 6),
        "contradiction_score": round(scores["contradiction"], 6),
        "neutral_score": round(scores["neutral"], 6),
    }


class NLIEngine:
    """Production-safe NLI wrapper with explicit label mapping and alignment."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base") -> None:
        self.model_name = model_name
        self.pipeline = None
        self._is_available = True

        # --- §15 engine-level execution diagnostics (real-execution-only proof) ---
        # This is the canonical pipeline NLI engine (see nli/__init__.py). It mirrors
        # nli/entailment.py's diagnostics so the pipeline can build a ModelExecutionTrace
        # (§26) and certification can fail-closed on fallback (§28). The decision logic
        # below (_normalize_scores/_decision, batch alignment) is intentionally unchanged.
        self.last_status: str = "not_run"  # executed | degraded | unavailable | not_run
        self.last_inference_executed: bool = False
        self.last_degraded: bool = False
        self.last_device: str = "unknown"
        self.last_latency_ms: int = 0
        self.last_batch_size: int = 0
        self.last_input_trace: list[dict] = []
        self.last_attempted = False
        self.last_initialization_attempted = False
        self.last_tokenizer_observability = "not_exposed"

    def _reset_run_diagnostics(self) -> None:
        self.last_status = "not_run"
        self.last_inference_executed = False
        self.last_degraded = False
        self.last_latency_ms = 0
        self.last_batch_size = 0
        self.last_input_trace = []
        self.last_attempted = False
        self.last_initialization_attempted = False
        self.last_tokenizer_observability = "not_exposed"

    def is_loaded(self) -> bool:
        """True iff the NLI pipeline is loaded and usable."""
        return self.pipeline is not None and self._is_available

    def _detect_device(self) -> str:
        """Best-effort device read from the loaded HF pipeline (never raises)."""
        pipe = self.pipeline
        if pipe is None:
            return "unknown"
        try:
            dev = getattr(pipe, "device", None)
            if dev is not None:
                return str(dev)
        except Exception:
            pass
        model = getattr(pipe, "model", None)
        try:
            dev = getattr(model, "device", None)
            if dev is not None:
                return str(dev)
        except Exception:
            pass
        return "unknown"

    def diagnostics(self) -> dict:
        """Return the last-run execution proof for tracing/certification (§15/§26)."""
        return {
            "component": "deberta_nli",
            "model": self.model_name,
            "loaded": self.is_loaded(),
            "inference_executed": self.last_inference_executed,
            "degraded": self.last_degraded,
            "status": self.last_status,
            "device": self.last_device,
            "latency_ms": self.last_latency_ms,
            "batch_size": self.last_batch_size,
            "attempted": self.last_attempted,
            "initialization_attempted": self.last_initialization_attempted,
            "submitted_inputs": self.last_input_trace,
            "tokenizer_observability": self.last_tokenizer_observability,
        }

    def _invoke(self, inputs, **kwargs):
        """Observe this invocation's actual preprocessing without mutating a
        cached/shared pipeline. Weights and tokenizer are shared read-only;
        the shallow pipeline copy owns only this local preprocessing callback.
        Observation errors never replace model outputs with invented scores.
        """
        if not callable(getattr(self.pipeline, "preprocess", None)) or not callable(
            getattr(self.pipeline, "tokenizer", None)
        ):
            return self.pipeline(inputs, **({} if isinstance(inputs, list) else kwargs))
        local = copy.copy(self.pipeline)
        original = self.pipeline.preprocess
        def observed(item, **params):
            processed = original(item, **params)
            try:
                ids = processed["input_ids"].tolist()[0]
                mask = processed.get("attention_mask")
                mask = mask.tolist()[0] if mask is not None else [1] * len(ids)
                retained = [i for i, m in zip(ids, mask) if m]
                text = item["text"] if isinstance(item, dict) else item
                pair = item.get("text_pair") if isinstance(item, dict) else None
                full = local.tokenizer(text, text_pair=pair, truncation=False,
                    add_special_tokens=True, return_attention_mask=False)["input_ids"]
                self.last_input_trace.append({"stage": "tokenized",
                    "premise_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "untruncated_pair_tokens": len(full), "retained_pair_tokens": len(retained),
                    "input_ids_sha256": hashlib.sha256(str(retained).encode()).hexdigest(),
                    "tokenizer_truncated": len(full) > len(retained),
                    "max_length": params.get("max_length"),
                })
                self.last_tokenizer_observability = "observed_preprocess"
            except Exception:
                self.last_tokenizer_observability = "observation_failed"
            return processed
        local.preprocess = observed
        return local(inputs, **kwargs)

    def _load_model(self) -> None:
        if self.pipeline is not None or not self._is_available:
            return
        self.last_initialization_attempted = True
        try:
            self.pipeline = get_model_manager().load_nli_model(self.model_name)
        except Exception as exc:
            logger.warning(
                "NLI model unavailable (%s); using degraded neutral result", type(exc).__name__
            )
            self._is_available = False

    def _get_id2label(self) -> Optional[Dict[Any, str]]:
        model = getattr(self.pipeline, "model", None)
        config = getattr(model, "config", None)
        return getattr(config, "id2label", None)

    @staticmethod
    def _neutral() -> Dict[str, Any]:
        return {
            "label": EntailmentLabel.NEUTRAL,
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 1.0,
            "degraded": True,
            "error": "nli_model_unavailable_or_failed",
        }

    def classify(
        self, claim: str, evidence: str, model_name: str | None = None
    ) -> Dict[str, Any]:
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self.pipeline = None
            self._is_available = True
        self._load_model()
        if not self._is_available or self.pipeline is None:
            return self._neutral()
        try:
            # Premise = evidence, hypothesis = claim.
            raw = self._invoke({"text": (evidence or "")[:1500], "text_pair": (claim or "")[:500]}, truncation=True, max_length=512)
            scores = _normalize_scores(_flatten_predictions(raw), self._get_id2label())
            return _decision(scores) if sum(scores.values()) > 0 else self._neutral()
        except Exception as exc:
            logger.warning("NLI classification failed: %s", type(exc).__name__)
            return self._neutral()

    def predict(self, claim: str, evidence: str) -> EntailmentLabel:
        return self.classify(claim, evidence)["label"]

    def batch_classify(
        self,
        claim: str,
        evidences: List[str],
        model_name: str | None = None,
    ) -> List[Dict[str, Any]]:
        # §15/§26 execution proof: reset first so an empty call reads "not_run".
        self._reset_run_diagnostics()
        if not evidences:
            return []
        self.last_attempted = True
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self.pipeline = None
            self._is_available = True
        self._load_model()
        if not self._is_available or self.pipeline is None:
            self.last_status = "unavailable"
            self.last_degraded = True
            self.last_inference_executed = False
            return [self._neutral() for _ in evidences]
        try:
            batch = [
                {"text": evidence or "", "text_pair": claim or ""}
                for evidence in evidences
            ]
            self.last_input_trace = [{
                "premise_sha256": hashlib.sha256((ev or "").encode()).hexdigest(),
                "premise_characters": len(ev or ""), "hypothesis_characters": len(claim or ""),
                "character_truncated": False, "tokenizer_truncated": None,
            } for ev in evidences]
            _t0 = time.perf_counter()
            raw_batch = self._invoke(batch, truncation=True, max_length=512)
            self.last_latency_ms = int((time.perf_counter() - _t0) * 1000)
            if not isinstance(raw_batch, list) or len(raw_batch) != len(evidences):
                raise ValueError("NLI batch output is not aligned with input batch")
            id2label = self._get_id2label()
            outputs = []
            malformed = 0
            for raw_item in raw_batch:
                try:
                    scores = _normalize_scores(_flatten_predictions(raw_item), id2label)
                    outputs.append(
                        _decision(scores) if sum(scores.values()) > 0 else self._neutral()
                    )
                except Exception:
                    # A single malformed prediction becomes NEUTRAL; it must NOT
                    # abort the whole batch and mis-mark a real DeBERTa run as
                    # degraded/not-executed, nor silently collapse every item
                    # (audit #16/#17). The degradation below is observable.
                    malformed += 1
                    outputs.append(self._neutral())
            if malformed:
                logger.warning(
                    "NLI batch: %d/%d predictions malformed -> NEUTRAL", malformed, len(evidences)
                )
            # Real DeBERTa inference ran; only FULLY degraded if every item failed.
            self.last_status = "executed" if malformed < len(evidences) else "degraded"
            self.last_inference_executed = True
            self.last_degraded = malformed == len(evidences)
            self.last_device = self._detect_device()
            self.last_batch_size = len(evidences)
            return outputs
        except Exception as exc:
            logger.warning("Batched NLI failed; retrying individually: %s", type(exc).__name__)
            self.last_status = "degraded"
            self.last_degraded = True
            self.last_inference_executed = False
            outputs = [
                self.classify(claim, evidence, model_name=self.model_name)
                for evidence in evidences
            ]
            # The retry path has a distinct character bound. Preserve both
            # attempted batch inputs and actually submitted retry pairs.
            self.last_input_trace.extend({
                "stage": "individual_retry",
                "premise_sha256": hashlib.sha256((ev or "")[:1500].encode()).hexdigest(),
                "premise_characters": min(len(ev or ""), 1500),
                "hypothesis_characters": min(len(claim or ""), 500),
                "character_truncated": len(ev or "") > 1500 or len(claim or "") > 500,
                "tokenizer_truncated": None,
            } for ev in evidences if self.is_loaded())
            self.last_inference_executed = any(not o.get("degraded", False) for o in outputs)
            return outputs
