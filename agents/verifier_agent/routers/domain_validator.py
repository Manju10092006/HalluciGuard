from __future__ import annotations
import logging
from typing import Tuple

class DomainValidator:
    """Validates the domain classification of a claim."""

    DOMAIN_LABELS = [
        "healthcare", "finance", "legal", "technology", "science", "politics", 
        "sports", "entertainment", "history", "geography", "education", 
        "environment", "business", "military", "religion", "arts", "culture",
        "health", "medicine", "economics", "law", "engineering", "general"
    ]

    def __init__(self) -> None:
        self.pipeline = None
        self._is_available = True

    def _load_model(self) -> None:
        if self.pipeline is not None or not self._is_available:
            return
            
        try:
            from transformers import pipeline
            self.pipeline = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        except ImportError:
            logging.warning("transformers not installed. DomainValidator falling back.")
            self._is_available = False
        except Exception as e:
            logging.warning(f"Error loading DomainValidator model: {e}. Falling back.")
            self._is_available = False

    def validate(self, claim_text: str, detector_domain: str) -> Tuple[str, bool]:
        """
        Validate if the detected domain matches the model's classification.
        
        Args:
            claim_text: The text to classify.
            detector_domain: The domain provided by the detector.
            
        Returns:
            Tuple containing (validated_domain, was_original_correct).
        """
        self._load_model()
        
        if not self._is_available:
            logging.warning("DomainValidator model not available. Trusting detector domain.")
            return (detector_domain, True)
            
        try:
            result = self.pipeline(claim_text, self.DOMAIN_LABELS)
            top_domain = result['labels'][0]
            
            # Simple equivalence check
            is_match = (top_domain.lower() == detector_domain.lower() or 
                        (top_domain in ['health', 'medicine'] and detector_domain == 'healthcare'))
            
            if is_match:
                return (detector_domain, True)
            else:
                logging.warning(f"Domain mismatch: detector={detector_domain}, model={top_domain}")
                return (top_domain, False)
                
        except Exception as e:
            logging.warning(f"Domain validation failed: {e}. Trusting detector domain.")
            return (detector_domain, True)
