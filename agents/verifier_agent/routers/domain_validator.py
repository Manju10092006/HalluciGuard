from __future__ import annotations
import logging
from typing import Tuple

from models.model_manager import get_model_manager

class DomainValidator:
    """Validates the domain classification of a claim."""

    DOMAIN_LABELS = [
        "healthcare", "finance", "legal_general", "cybersecurity", "ai_research",
        "programming", "scientific", "education", "government", "news",
        "mathematics", "physics", "chemistry", "biology", "space", "history",
        "geography", "economics", "climate", "sports", "business",
        "manufacturing", "pharmaceuticals", "medicine", "legal", "law",
        "technology", "science", "environment", "general"
    ]

    def __init__(self) -> None:
        self.pipeline = None
        self._is_available = True

    def _canonical_domain(self, domain: str) -> str:
        normalized = domain.lower()
        aliases = {
            "health": "healthcare",
            "medicine": "healthcare",
            "pharmaceuticals": "healthcare",
            "legal": "legal_general",
            "law": "legal_general",
            "technology": "programming",
            "science": "scientific",
            "environment": "climate",
        }
        return aliases.get(normalized, normalized)

    def _load_model(self) -> None:
        if self.pipeline is not None or not self._is_available:
            return
            
        try:
            self.pipeline = get_model_manager().load_zero_shot_model()
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
            canonical_top = self._canonical_domain(top_domain)
            canonical_detector = self._canonical_domain(detector_domain)
            
            is_match = canonical_top == canonical_detector
            
            if is_match:
                return (detector_domain, True)
            else:
                logging.warning(f"Domain mismatch: detector={detector_domain}, model={top_domain}")
                return (canonical_top, False)
                
        except Exception as e:
            logging.warning(f"Domain validation failed: {e}. Trusting detector domain.")
            return (detector_domain, True)
