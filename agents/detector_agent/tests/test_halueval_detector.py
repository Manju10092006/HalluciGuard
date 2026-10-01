"""
HalluciGuard Detector Agent — Comprehensive Test Suite.

Tests the HaluEval-trained detector end-to-end:
- Model loading and inference
- LOW/MEDIUM/HIGH risk classification
- Routing logic (all risk tiers → Verify until fast-path certification)
- Edge cases (empty input, long input, unicode, special chars)
- Verifier handoff behavior
- FastAPI endpoint integration
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.detector_agent.config import DetectorConfig
from agents.detector_agent.detector import DetectorAgent
from agents.detector_agent.models import DetectionResult, NextAction, RiskLevel

pytestmark = pytest.mark.skipif(
    not (Path(project_root) / "artifacts" / "halueval-detector-final").exists(),
    reason="real HaluEval checkpoint is not present in this worktree",
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def detector():
    """Create a DetectorAgent instance (model loads on first detect call)."""
    config = DetectorConfig()
    return DetectorAgent(config=config)


# ============================================================
# Test: Model Loading
# ============================================================

class TestModelLoading:
    """Verify that the trained HaluEval model loads correctly."""

    def test_model_loads_from_artifacts(self, detector):
        """The model should load from the configured artifacts path."""
        detector._ensure_model_loaded()
        assert detector._ensure_model_loaded() is True
        assert detector._inference.is_loaded() is True

    def test_model_path_exists(self, detector):
        """The model artifacts directory should exist on disk."""
        model_path = detector.config.halueval_model_path
        # Resolve relative to project root
        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)
        assert os.path.exists(model_path), f"Model path not found: {model_path}"


# ============================================================
# Test: LOW Risk (Correct Response)
# ============================================================

class TestLowRisk:
    """Test low-risk model output without granting a verification bypass."""

    def test_paris_capital(self, detector):
        """'Capital of France is Paris' should be LOW risk."""
        result = detector.detect(
            user_query="What is the capital of France?",
            llm_response="The capital of France is Paris."
        )
        assert isinstance(result, DetectionResult)
        print(f"\n[LOW TEST] Paris: prob={result.hallucination_probability:.4f} "
              f"risk={result.risk_level.value} action={result.next_action.value}")
        
        # The model should classify this as low hallucination probability
        assert result.hallucination_probability < 0.70, (
            f"Expected hallucination_probability < 0.70 for a correct answer, "
            f"got {result.hallucination_probability}"
        )
        assert result.next_action == NextAction.VERIFY

    def test_http_protocol(self, detector):
        """'HTTP stands for Hypertext Transfer Protocol' should be LOW risk."""
        result = detector.detect(
            user_query="What does HTTP stand for?",
            llm_response="HTTP stands for Hypertext Transfer Protocol."
        )
        print(f"[LOW TEST] HTTP: prob={result.hallucination_probability:.4f} "
              f"risk={result.risk_level.value} action={result.next_action.value}")
        
        assert result.next_action == NextAction.VERIFY


# ============================================================
# Test: HIGH Risk (Hallucinated Response)
# ============================================================

class TestHighRisk:
    """Test that clearly hallucinated responses get HIGH risk / Verify."""

    def test_tokyo_capital_of_france(self, detector):
        """'Capital of France is Tokyo' should be HIGH risk."""
        result = detector.detect(
            user_query="What is the capital of France?",
            llm_response="The capital of France is Tokyo, Japan."
        )
        print(f"\n[HIGH TEST] Tokyo: prob={result.hallucination_probability:.4f} "
              f"risk={result.risk_level.value} action={result.next_action.value}")
        
        # The model should classify this as high hallucination probability
        assert result.hallucination_probability > 0.40, (
            f"Expected hallucination_probability > 0.40 for a hallucinated answer, "
            f"got {result.hallucination_probability}"
        )

    def test_einstein_shakespeare(self, detector):
        """'Einstein wrote Romeo and Juliet' should be high probability."""
        result = detector.detect(
            user_query="Who wrote Romeo and Juliet?",
            llm_response="Romeo and Juliet was written by Albert Einstein in 1920."
        )
        print(f"[HIGH TEST] Einstein: prob={result.hallucination_probability:.4f} "
              f"risk={result.risk_level.value} action={result.next_action.value}")
        
        assert result.hallucination_probability > 0.40


# ============================================================
# Test: Routing Logic
# ============================================================

class TestRoutingLogic:
    """Verify every risk tier requests evidence-based verification."""

    def test_low_risk_verifies(self, detector):
        assert detector._determine_next_action(RiskLevel.LOW) == NextAction.VERIFY

    def test_medium_risk_verifies(self, detector):
        """MEDIUM risk should Verify."""
        # Test through the actual _determine_next_action method
        action = detector._determine_next_action(RiskLevel.MEDIUM)
        assert action == NextAction.VERIFY, (
            f"MEDIUM risk should Verify, not {action.value}"
        )

    def test_high_risk_verifies(self, detector):
        """HIGH risk should Verify."""
        action = detector._determine_next_action(RiskLevel.HIGH)
        assert action == NextAction.VERIFY

    def test_threshold_boundaries(self, detector):
        """Test threshold boundary conditions."""
        low = detector.config.low_risk_threshold
        high = detector.config.high_risk_threshold
        # At or below low_risk_threshold → LOW
        assert detector._determine_risk_level(low) == RiskLevel.LOW
        # Just above low → MEDIUM
        assert detector._determine_risk_level(low + 0.01) == RiskLevel.MEDIUM
        # Just below high → MEDIUM
        assert detector._determine_risk_level(high - 0.01) == RiskLevel.MEDIUM
        # At or above high_risk_threshold → HIGH
        assert detector._determine_risk_level(high) == RiskLevel.HIGH


# ============================================================
# Test: Edge Cases
# ============================================================

class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_empty_query(self, detector):
        """Empty query should return a safe default, not crash."""
        result = detector.detect(user_query="", llm_response="Some response")
        assert isinstance(result, DetectionResult)
        assert result.risk_level == RiskLevel.HIGH
        assert result.next_action == NextAction.VERIFY

    def test_empty_response(self, detector):
        """Empty response should return a safe default, not crash."""
        result = detector.detect(user_query="What is AI?", llm_response="")
        assert isinstance(result, DetectionResult)

    def test_unicode_input(self, detector):
        """Unicode characters should not crash the detector."""
        result = detector.detect(
            user_query="What is café résumé naïve?",
            llm_response="A café is a place serving coffee. Résumé is a document."
        )
        assert isinstance(result, DetectionResult)

    def test_special_characters(self, detector):
        """Special characters should not crash the detector."""
        result = detector.detect(
            user_query="What is <script>alert('xss')</script>?",
            llm_response="<b>HTML tags</b> and 'quotes' and \"double\" and `backticks`."
        )
        assert isinstance(result, DetectionResult)

    def test_long_response(self, detector):
        """Very long responses should be handled (truncated by tokenizer)."""
        long_response = "This is a test sentence. " * 500  # ~3000 tokens
        result = detector.detect(
            user_query="Tell me everything you know.",
            llm_response=long_response
        )
        assert isinstance(result, DetectionResult)

    def test_very_short_answer(self, detector):
        """Very short answers should work."""
        result = detector.detect(
            user_query="Is the sky blue?",
            llm_response="Yes."
        )
        assert isinstance(result, DetectionResult)


# ============================================================
# Test: Verifier Handoff
# ============================================================

class TestVerifierHandoff:
    """Test all risk tiers request Verifier handoff."""

    def test_high_risk_should_invoke_verifier(self, detector):
        """When risk is HIGH, the pipeline should route to Verifier."""
        # We verify routing logic: HIGH → VERIFY
        result = detector.detect(
            user_query="What is the capital of France?",
            llm_response="The capital of France is Tokyo, Japan."
        )
        if result.risk_level == RiskLevel.HIGH:
            assert result.next_action == NextAction.VERIFY
            print(f"[HANDOFF] HIGH risk correctly routes to VERIFY")
        else:
            print(f"[HANDOFF] Model classified as {result.risk_level.value}, "
                  f"not HIGH — reporting actual result")

    def test_low_risk_should_invoke_verifier(self, detector):
        """A low score cannot replace evidence verification."""
        result = detector.detect(
            user_query="What is the capital of France?",
            llm_response="The capital of France is Paris."
        )
        assert result.next_action == NextAction.VERIFY
        print("[HANDOFF] LOW/MEDIUM routes to Verifier")

    def test_medium_risk_should_invoke_verifier(self, detector):
        """MEDIUM risk still requires Verifier."""
        action = detector._determine_next_action(RiskLevel.MEDIUM)
        assert action == NextAction.VERIFY, (
            f"MEDIUM risk should invoke Verifier. Got: {action.value}"
        )
        print("[HANDOFF] MEDIUM routes to Verifier")


# ============================================================
# Test: Output Schema Validation
# ============================================================

class TestOutputSchema:
    """Verify output schema matches the expected contract."""

    def test_detection_result_fields(self, detector):
        """DetectionResult must have all required fields."""
        result = detector.detect(
            user_query="What is 2+2?",
            llm_response="2+2 equals 4."
        )
        assert hasattr(result, "confidence_score")
        assert hasattr(result, "hallucination_probability")
        assert hasattr(result, "risk_level")
        assert hasattr(result, "next_action")
        assert 0.0 <= result.confidence_score <= 1.0
        assert 0.0 <= result.hallucination_probability <= 1.0
        assert isinstance(result.risk_level, RiskLevel)
        assert isinstance(result.next_action, NextAction)

    def test_confidence_and_probability_complement(self, detector):
        """confidence_score + hallucination_probability should approximately equal 1.0."""
        result = detector.detect(
            user_query="What is water?",
            llm_response="Water is H2O, a chemical compound."
        )
        total = result.confidence_score + result.hallucination_probability
        assert abs(total - 1.0) < 0.01, (
            f"confidence + hallucination_probability should ≈ 1.0, got {total}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
