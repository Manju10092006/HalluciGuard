"""
HalluciGuard - NLI Inference Engine
The Judge Agent's own analytical tool for claim-evidence entailment analysis.
Uses HuggingFace DeBERTa/BART with robust heuristic fallback.
"""

import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger("HalluciGuard.NLIEngine")


class NLIEngine:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", use_hf: bool = True):
        self.model_name = model_name
        self.pipeline = None
        self.is_loaded = False
        if use_hf:
            self._init_pipeline()

    def _init_pipeline(self):
        try:
            from transformers import pipeline
            self.pipeline = pipeline("text-classification", model=self.model_name, return_all_scores=True)
            self.is_loaded = True
            logger.info(f"HF NLI model '{self.model_name}' loaded.")
        except Exception as e:
            logger.warning(f"HF pipeline unavailable ({e}). Using heuristic fallback.")
            self.is_loaded = False

    def predict(self, evidence: str, claim: str) -> Dict[str, float]:
        if not evidence or not claim:
            return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}
        if self.is_loaded and self.pipeline:
            try:
                results = self.pipeline(f"{evidence} [SEP] {claim}")[0]
                scores = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
                for item in results:
                    lbl = item["label"].lower()
                    if "entail" in lbl: scores["entailment"] = item["score"]
                    elif "neutral" in lbl: scores["neutral"] = item["score"]
                    elif "contradict" in lbl: scores["contradiction"] = item["score"]
                total = sum(scores.values()) or 1.0
                return {k: round(v / total, 4) for k, v in scores.items()}
            except Exception:
                pass
        return self._heuristic_nli(evidence, claim)

    def batch_predict(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for p in pairs:
            nli = self.predict(p.get("evidence", ""), p.get("claim", ""))
            results.append({**p, "nli_scores": nli, "top_relation": max(nli, key=nli.get)})
        return results

    def _heuristic_nli(self, evidence: str, claim: str) -> Dict[str, float]:
        ev, cl = evidence.lower(), claim.lower()

        # Refutation keyword detection
        refutation_kw = {"contraindicated", "prohibited", "incorrect", "false", "denied",
                         "unauthorized", "banned", "not recommended", "refuted", "do not use",
                         "dangerous", "fatal", "risk of death"}
        ev_has_refute = any(w in ev for w in refutation_kw)
        cl_has_refute = any(w in cl for w in refutation_kw)

        # Strong safety refutation — keywords that indicate explicit prohibition
        safety_refute_kw = {"contraindicated", "fatal", "risk of death", "do not use", "dangerous"}
        ev_has_safety_refute = any(w in ev for w in safety_refute_kw)

        # Negation detection
        neg_words = {"not", "never", "no", "neither", "nor", "cannot", "didn't", "won't"}
        ev_neg = any(w in ev.split() for w in neg_words)
        cl_neg = any(w in cl.split() for w in neg_words)

        # Token overlap
        ev_tok = set(w.strip(".,!?:;\"'") for w in ev.split() if len(w) > 2)
        cl_tok = set(w.strip(".,!?:;\"'") for w in cl.split() if len(w) > 2)
        if not cl_tok:
            return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}
        overlap = len(cl_tok & ev_tok) / len(cl_tok)

        # Safety refutation: evidence contains safety prohibition keywords with any topic overlap
        if ev_has_safety_refute and not cl_has_refute and overlap > 0.05:
            return {"entailment": 0.02, "neutral": 0.05, "contradiction": 0.93}

        # Explicit refutation: evidence refutes but claim does not
        if (ev_has_refute and not cl_has_refute and overlap > 0.15):
            return {"entailment": 0.05, "neutral": 0.10, "contradiction": 0.85}

        # Negation mismatch with topic overlap (only if direct opposing claim vs evidence negation)
        if ev_neg != cl_neg and overlap > 0.45:
            # Only trigger if the negation is directly on shared key verbs/predicates
            ev_is_not = "is not" in ev or "was not" in ev or "cannot" in ev or "does not" in ev
            cl_is_not = "is not" in cl or "was not" in cl or "cannot" in cl or "does not" in cl
            if (ev_is_not or cl_is_not) and (ev_is_not != cl_is_not):
                # Check if it's describing vulnerability/risk vs stating a fact
                if not any(w in ev for w in ["vulnerability", "exploit", "attack", "risk", "protect", "defeat"]):
                    return {"entailment": 0.05, "neutral": 0.15, "contradiction": 0.80}

        # Numeric conflict
        cl_nums = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', claim))
        ev_nums = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', evidence))
        if cl_nums and ev_nums and not cl_nums & ev_nums and overlap > 0.25:
            return {"entailment": 0.05, "neutral": 0.15, "contradiction": 0.80}

        # High overlap -> entailment
        if overlap >= 0.60:
            return {"entailment": round(0.85 + 0.14 * overlap, 4), "neutral": round(0.10 * (1 - overlap), 4), "contradiction": 0.02}
        elif overlap >= 0.35:
            return {"entailment": 0.45, "neutral": 0.45, "contradiction": 0.10}
        else:
            return {"entailment": 0.15, "neutral": 0.75, "contradiction": 0.10}
