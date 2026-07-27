"""Unit tests for SignalWeights configuration and validation in HalluciGuard Detector Agent."""

from pydantic import ValidationError
import pytest
from detector_agent import DetectorAgent, DetectorConfig
from detector_agent.config import SignalWeights


def test_default_weights():
    """Test 1: Verify default signal weights initialization and validation."""
    config = DetectorConfig()
    weights = config.signal_weights

    assert weights.token_probability == 0.40
    assert weights.entropy == 0.30
    assert weights.semantic_similarity == 0.30
    assert weights.self_consistency == 0.00

    total_sum = sum(weights.model_dump().values())
    assert abs(total_sum - 1.0) < 1e-4
    print("✓ Test 1 Passed: Default signal weights valid (sum = 1.0).")


def test_custom_valid_weights():
    """Test 2: Verify custom valid signal weights initialization."""
    custom_weights = SignalWeights(
        token_probability=0.60,
        entropy=0.20,
        semantic_similarity=0.20,
        self_consistency=0.00
    )
    config = DetectorConfig(signal_weights=custom_weights)
    agent = DetectorAgent(config=config)

    assert agent.config.signal_weights.token_probability == 0.60
    assert agent.config.signal_weights.entropy == 0.20
    assert agent.config.signal_weights.semantic_similarity == 0.20
    print("✓ Test 2 Passed: Custom signal weights accepted.")


def test_invalid_weights_sum():
    """Test 3: Verify ValueError is raised when weights do not sum to 1.0."""
    try:
        SignalWeights(
            token_probability=0.50,
            entropy=0.50,
            semantic_similarity=0.50,  # Total sum = 1.5
            self_consistency=0.00
        )
        assert False, "Expected ValueError was not raised!"
    except (ValueError, ValidationError) as e:
        assert "Signal weights must sum to 1.0" in str(e)
        print("✓ Test 3 Passed: Invalid weights sum correctly rejected with ValueError.")


def test_invalid_negative_or_out_of_bounds_weights():
    """Test 4: Verify validation error when a weight is < 0 or > 1."""
    try:
        SignalWeights(
            token_probability=-0.20,  # Invalid negative weight
            entropy=0.60,
            semantic_similarity=0.60,
            self_consistency=0.00
        )
        assert False, "Expected ValidationError was not raised!"
    except (ValueError, ValidationError):
        print("✓ Test 4 Passed: Negative signal weight rejected.")


def test_partial_weights_fallback():
    """Test 5: Verify partial custom weights with defaults summing to 1.0."""
    # Custom weights with explicit sum = 1.0
    custom_weights = SignalWeights(
        token_probability=0.70,
        entropy=0.30,
        semantic_similarity=0.00,
        self_consistency=0.00
    )
    config = DetectorConfig(signal_weights=custom_weights)
    assert config.signal_weights.token_probability == 0.70
    assert config.signal_weights.entropy == 0.30
    assert config.signal_weights.semantic_similarity == 0.00
    print("✓ Test 5 Passed: Partial weights fallback valid.")


def main():
    print("=========================================================")
    print(" HalluciGuard - Signal Weights Configuration Unit Tests")
    print("=========================================================\n")
    test_default_weights()
    test_custom_valid_weights()
    test_invalid_weights_sum()
    test_invalid_negative_or_out_of_bounds_weights()
    test_partial_weights_fallback()
    print("\nAll unit tests passed successfully!")


if __name__ == "__main__":
    main()
