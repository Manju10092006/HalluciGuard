"""
HalluciGuard - Judge Agent Natural Language Inference (NLI) Engine
Uses Hugging Face DeBERTa-v3 / BART-MNLI to perform entailment, neutral, and contradiction scoring
between premise (evidence) and hypothesis (claim).
"""

import logging
from typing import Dict, List, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HalluciGuard.NLIEngine")

class NLIEngine:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", use_hf: bool = True):
        self.model_name = model_name
        self.use_hf = use_hf
        self.pipeline = None
        self.is_loaded = False
        
        if self.use_hf:
            self._initialize_pipeline()

    def _initialize_pipeline(self):
        try:
            from transformers import pipeline
            logger.info(f"Loading Hugging Face NLI model: {self.model_name}...")
            # We attempt loading text-classification or zero-shot-classification pipeline
            self.pipeline = pipeline("text-classification", model=self.model_name, return_all_scores=True)
            self.is_loaded = True
            logger.info("Hugging Face NLI model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load Hugging Face pipeline ({e}). Falling back to Heuristic/Semantic NLI Engine.")
            self.is_loaded = False

    def predict(self, evidence: str, claim: str) -> Dict[str, float]:
        """
        Evaluates NLI relationship between evidence (premise) and claim (hypothesis).
        Returns dictionary with probabilities for 'entailment', 'neutral', and 'contradiction'.
        """
        if not evidence or not claim:
            return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}

        if self.is_loaded and self.pipeline:
            try:
                # Format pair for Cross-Encoder model
                # DeBERTa-v3 cross-encoders usually expect text pairs
                inputs = f"{evidence} [SEP] {claim}"
                results = self.pipeline(inputs)[0]
                
                scores = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
                for item in results:
                    label = item["label"].lower()
                    score = item["score"]
                    if "entail" in label:
                        scores["entailment"] = score
                    elif "neutral" in label:
                        scores["neutral"] = score
                    elif "contradict" in label:
                        scores["contradiction"] = score
                
                # Normalize if needed
                total = sum(scores.values()) or 1.0
                return {k: round(v / total, 4) for k, v in scores.items()}
            except Exception as e:
                logger.warning(f"HF Inference failed: {e}. Using fallback engine.")

        return self._fallback_heuristic_nli(evidence, claim)

    def batch_predict(self, claim_evidence_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a batch of claim-evidence pairs.
        Each dict should contain 'claim' and 'evidence'.
        """
        evaluated_pairs = []
        for pair in claim_evidence_pairs:
            claim = pair.get("claim", "")
            evidence = pair.get("evidence", "")
            nli_scores = self.predict(evidence, claim)
            
            evaluated_pairs.append({
                **pair,
                "nli_scores": nli_scores,
                "top_relation": max(nli_scores, key=nli_scores.get)
            })
        return evaluated_pairs

    def _fallback_heuristic_nli(self, evidence: str, claim: str) -> Dict[str, float]:
        """
        Robust heuristic NLI engine based on token overlap, negation detection, and key phrase alignment.
        Used as zero-dependency fallback.
        """
        ev_lower = evidence.lower()
        cl_lower = claim.lower()

        # Contradiction indicators
        contradiction_keywords = {"contraindicated", "prohibited", "incompatible", "incorrect", "refuted", "denied", "false", "unauthorized", "banned"}
        has_contradict_word = any(w in ev_lower for w in contradiction_keywords) and not any(w in cl_lower for w in contradiction_keywords)

        # Negation mismatch detector
        negation_words = {"not", "never", "no", "neither", "nor", "didn't", "cannot", "won't"}
        ev_has_neg = any(w in ev_lower.split() for w in negation_words)
        cl_has_neg = any(w in cl_lower.split() for w in negation_words)

        # Token overlap analysis
        ev_words = set(w.strip(".,!?:;") for w in ev_lower.split() if len(w) > 2)
        cl_words = set(w.strip(".,!?:;") for w in cl_lower.split() if len(w) > 2)

        if not cl_words:
            return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}

        overlap = cl_words.intersection(ev_words)
        overlap_ratio = len(overlap) / len(cl_words)

        # Explicit contradiction detection
        if (ev_has_neg != cl_has_neg and overlap_ratio > 0.3) or (has_contradict_word and overlap_ratio > 0.25):
            return {"entailment": 0.05, "neutral": 0.15, "contradiction": 0.80}

        # Check for numbers/dates conflict
        import re
        ev_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', evidence))
        cl_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', claim))
        if ev_nums and cl_nums and not ev_nums.intersection(cl_nums):
            # Numeric conflict found!
            return {"entailment": 0.05, "neutral": 0.20, "contradiction": 0.75}

        if overlap_ratio >= 0.65:
            return {"entailment": round(0.70 + 0.25 * overlap_ratio, 4), 
                    "neutral": round(0.20 * (1 - overlap_ratio), 4), 
                    "contradiction": 0.05}
        elif overlap_ratio >= 0.35:
            return {"entailment": 0.40, "neutral": 0.50, "contradiction": 0.10}
        else:
            return {"entailment": 0.15, "neutral": 0.75, "contradiction": 0.10}
