from __future__ import annotations
import logging
from typing import List, Dict, Any

from schemas.models import EntailmentLabel

class NLIEngine:
    """Natural Language Inference engine for entailment classification."""

    def __init__(self, model_name: str = 'microsoft/deberta-v3-base-mnli') -> None:
        self.model_name = model_name
        self.pipeline = None
        self._is_available = True
        
    def _load_model(self) -> None:
        if self.pipeline is not None or not self._is_available:
            return
            
        try:
            from transformers import pipeline
            self.pipeline = pipeline('zero-shot-classification', model=self.model_name)
            # Actually, standard NLI uses text-classification. 
            # If using zero-shot, we can provide candidate labels.
            # However, for MNLI models, standard text-classification pipeline is usually better.
            # Assuming standard text classification.
            self.pipeline = pipeline('text-classification', model=self.model_name, return_all_scores=True)
        except ImportError:
            logging.warning("transformers not installed. NLIEngine falling back.")
            self._is_available = False
        except Exception as e:
            logging.warning(f"Error loading NLI model: {e}. Falling back.")
            self._is_available = False

    def classify(self, claim: str, evidence: str) -> Dict[str, Any]:
        """
        Classify the entailment relationship between claim and evidence.
        
        Args:
            claim: The claim text.
            evidence: The evidence text.
            
        Returns:
            Dict containing label, entailment_score, contradiction_score, neutral_score.
        """
        self._load_model()
        
        if not self._is_available:
            return {
                'label': EntailmentLabel.NEUTRAL,
                'entailment_score': 0.33,
                'contradiction_score': 0.33,
                'neutral_score': 0.34
            }
            
        try:
            # For MNLI models, input format is typically [CLS] premise [SEP] hypothesis [SEP]
            # When using text-classification pipeline, we can pass it as a dict or a string.
            # Here we format it directly.
            result = self.pipeline({'text': evidence, 'text_pair': claim})
            
            # Format results
            scores = {}
            for item in result:
                label = item['label'].lower()
                scores[label] = item['score']
                
            # Map standard NLI labels
            entailment = scores.get('entailment', scores.get('label_0', 0.0))
            neutral = scores.get('neutral', scores.get('label_1', 0.0))
            contradiction = scores.get('contradiction', scores.get('label_2', 0.0))
            
            max_score = max(entailment, neutral, contradiction)
            if max_score == entailment:
                final_label = EntailmentLabel.ENTAILMENT
            elif max_score == contradiction:
                final_label = EntailmentLabel.CONTRADICTION
            else:
                final_label = EntailmentLabel.NEUTRAL
                
            return {
                'label': final_label,
                'entailment_score': entailment,
                'contradiction_score': contradiction,
                'neutral_score': neutral
            }
            
        except Exception as e:
            logging.warning(f"Error during NLI classification: {e}. Returning neutral.")
            return {
                'label': EntailmentLabel.NEUTRAL,
                'entailment_score': 0.33,
                'contradiction_score': 0.33,
                'neutral_score': 0.34
            }

    def predict(self, claim: str, evidence: str) -> EntailmentLabel:
        """Helper returning the top EntailmentLabel."""
        res = self.classify(claim, evidence)
        return res['label']


    def batch_classify(self, claim: str, evidences: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple evidences against a single claim.
        """
        return [self.classify(claim, ev) for ev in evidences]
