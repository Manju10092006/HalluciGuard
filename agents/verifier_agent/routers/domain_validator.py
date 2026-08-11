from __future__ import annotations
import logging
from typing import Tuple

from config.settings import get_settings
from models.domain_intelligence import get_domain_intelligence_registry
from models.model_manager import get_model_manager

logger = logging.getLogger(__name__)

class DomainValidator:
    """Validates the domain classification of a claim."""

    def __init__(self) -> None:
        self.pipeline = None
        self._is_available = True
        self.domain_registry = get_domain_intelligence_registry()
        self.DOMAIN_LABELS = self.domain_registry.list_domains()

    def _canonical_domain(self, domain: str) -> str:
        return self.domain_registry.canonicalize(domain)

    def _load_model(self) -> None:
        if self.pipeline is not None or not self._is_available:
            return
            
        try:
            self.pipeline = get_model_manager().load_zero_shot_model()
        except ImportError:
            logger.warning("transformers not installed. DomainValidator falling back.")
            self._is_available = False
        except Exception as e:
            logger.warning(f"Error loading DomainValidator model: {e}. Falling back.")
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
        canonical_detector = self._canonical_domain(detector_domain)
        if not get_settings().enable_domain_classifier:
            return (canonical_detector, True)

        self._load_model()
        
        if not self._is_available:
            logger.warning("DomainValidator model not available. Trusting detector domain.")
            return (canonical_detector, True)
            
        try:
            result = self.pipeline(claim_text, self.DOMAIN_LABELS)
            top_domain = result['labels'][0]
            canonical_top = self._canonical_domain(top_domain)
            is_match = canonical_top == canonical_detector
            
            if is_match:
                return (canonical_detector, True)
            else:
                logger.warning(f"Domain mismatch: detector={detector_domain}, model={top_domain}")
                return (canonical_top, False)
                
        except Exception as e:
            logger.warning(f"Domain validation failed: {e}. Trusting detector domain.")
            return (self._canonical_domain(detector_domain), True)
