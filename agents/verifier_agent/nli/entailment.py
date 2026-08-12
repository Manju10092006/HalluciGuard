from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from schemas.models import EntailmentLabel
from models.model_manager import get_model_manager

logger = logging.getLogger(__name__)


def _normalize_nli_scores(
    result_items: List[Dict[str, Any]],
    id2label: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    """
    Dynamically maps raw HuggingFace classification outputs to standard NLI keys:
      - entailment
      - contradiction
      - neutral
    """
    scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}

    for item in result_items:
        raw_label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))

        if "entail" in raw_label or raw_label == "supports":
            scores["entailment"] = score
        elif "contradict" in raw_label or "refute" in raw_label:
            scores["contradiction"] = score
        elif "neutral" in raw_label:
            scores["neutral"] = score
        elif raw_label.startswith("label_"):
            try:
                idx = int(raw_label.split("_")[1])
                if id2label and idx in id2label:
                    canonical = str(id2label[idx]).lower()
                    if "entail" in canonical:
                        scores["entailment"] = score
                    elif "contradict" in canonical:
                        scores["contradiction"] = score
                    elif "neutral" in canonical:
                        scores["neutral"] = score
                else:
                    # Standard DeBERTa/RoBERTa MNLI mapping fallback: 0=contradiction, 1=entailment, 2=neutral
                    if idx == 0:
                        scores["contradiction"] = score
                    elif idx == 1:
                        scores["entailment"] = score
                    elif idx == 2:
                        scores["neutral"] = score
            except (ValueError, IndexError):
                logger.debug("Could not parse label index %s", raw_label)

    # Softmax normalization if sum > 0
    total = sum(scores.values())
    if total > 0 and abs(total - 1.0) > 0.01:
        scores = {k: v / total for k, v in scores.items()}

    return scores


class NLIEngine:
    """Natural Language Inference engine for entailment classification."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base") -> None:
        self.model_name = model_name
        self.pipeline = None
        self._is_available = True
        self._load_attempts = 0

    def _load_model(self) -> None:
        if self.pipeline is not None or not self._is_available:
            return
            
        if self._load_attempts >= 2:
            self._is_available = False
            return
            
        self._load_attempts += 1

        try:
            self.pipeline = get_model_manager().load_nli_model(self.model_name)
            self._load_attempts = 0
        except ImportError:
            logger.warning("transformers not installed. NLIEngine falling back.")
            self._is_available = False
        except Exception as e:
            logger.warning(f"Error loading NLI model {self.model_name}: {e}. Falling back.")
            if self._load_attempts >= 2:
                self._is_available = False

    def _get_id2label(self) -> Optional[Dict[int, str]]:
        if self.pipeline and hasattr(self.pipeline, "model") and hasattr(self.pipeline.model, "config"):
            return getattr(self.pipeline.model.config, "id2label", None)
        return None

    def classify(self, claim: str, evidence: str, model_name: str | None = None) -> Dict[str, Any]:
        """
        Classify the entailment relationship between claim (Hypothesis) and evidence (Premise).
        Follows standard FEVER/SciFact ordering: Premise=evidence, Hypothesis=claim.
        """
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self.pipeline = None
            self._is_available = True

        self._load_model()

        fallback = {
            "label": EntailmentLabel.NEUTRAL,
            "entailment_score": 0.33,
            "contradiction_score": 0.33,
            "neutral_score": 0.34,
            "degraded": True,
        }

        if not self._is_available or self.pipeline is None:
            logger.warning("Returning fallback results from classify (model %s unavailable or not loaded)", self.model_name)
            return fallback

        try:
            # FEVER / SciFact sentence pair pairing: Premise=evidence, Hypothesis=claim
            logger.debug("[NLI Pipeline Input] Premise(Evidence): %s | Hypothesis(Claim): %s", evidence[:100], claim[:100])
            
            result = self.pipeline({"text": evidence, "text_pair": claim})
            if result and isinstance(result[0], list):
                result = result[0]

            id2label = self._get_id2label()
            logger.debug("[NLI Model Config] id2label: %s", id2label)
            
            scores = _normalize_nli_scores(result, id2label)

            entailment = round(scores["entailment"], 6)
            contradiction = round(scores["contradiction"], 6)
            neutral = round(scores["neutral"], 6)

            prob_sum = round(entailment + contradiction + neutral, 4)
            logger.debug("[NLI Softmax Probabilities] E=%.4f, C=%.4f, N=%.4f (sum=%.4f)", entailment, contradiction, neutral, prob_sum)

            if entailment > contradiction and entailment > neutral:
                final_label = EntailmentLabel.ENTAILMENT
            elif contradiction > entailment and contradiction > neutral:
                final_label = EntailmentLabel.CONTRADICTION
            else:
                final_label = EntailmentLabel.NEUTRAL

            return {
                "label": final_label,
                "entailment_score": entailment,
                "contradiction_score": contradiction,
                "neutral_score": neutral,
                "degraded": False,
            }

        except Exception as e:
            logger.warning("Error during NLI classification with model %s: %s. Returning fallback.", self.model_name, e)
            return fallback

    def predict(self, claim: str, evidence: str) -> EntailmentLabel:
        """Helper returning top EntailmentLabel."""
        res = self.classify(claim, evidence)
        return res["label"]

    def batch_classify(
        self,
        claim: str,
        evidences: List[str],
        model_name: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple evidence passages against a single claim.
        """
        if not evidences:
            return []

        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self.pipeline = None
            self._is_available = True

        self._load_model()

        if not self._is_available or self.pipeline is None:
            logger.warning("Returning fallback results from batch_classify (model %s unavailable)", self.model_name)
            return [self.classify(claim, ev) for ev in evidences]

        try:
            batch = [{"text": ev, "text_pair": claim} for ev in evidences]
            raw_results = self.pipeline(batch)
            id2label = self._get_id2label()

            outputs: List[Dict[str, Any]] = []
            for result in raw_results:
                rows = result if result and isinstance(result[0], dict) else result[0]
                scores = _normalize_nli_scores(rows, id2label)

                entailment = scores["entailment"]
                contradiction = scores["contradiction"]
                neutral = scores["neutral"]

                if entailment > contradiction and entailment > neutral:
                    label = EntailmentLabel.ENTAILMENT
                elif contradiction > entailment and contradiction > neutral:
                    label = EntailmentLabel.CONTRADICTION
                else:
                    label = EntailmentLabel.NEUTRAL

                outputs.append(
                    {
                        "label": label,
                        "entailment_score": entailment,
                        "contradiction_score": contradiction,
                        "neutral_score": neutral,
                        "degraded": False,
                    }
                )
            return outputs
        except Exception as e:
            logger.warning("Error during batched NLI classification with model %s: %s. Falling back.", self.model_name, e)
            return [self.classify(claim, ev) for ev in evidences]
