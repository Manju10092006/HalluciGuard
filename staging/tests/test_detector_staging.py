"""
HalluciGuard Detector Agent — Staging Test Suite.

Validates end-to-end detector staging behavior against the real Hugging Face model:
Manjunath2000006/halluciguard-detector
"""

import os
import pytest
from fastapi.testclient import TestClient

from staging.app import app, detector_agent
from staging.detector.config import DetectorConfig
from staging.detector.detector import DetectorAgent
from staging.detector.models import NextAction, RiskLevel


@pytest.fixture(scope="session")
def client():
    """Create a FastAPI test client with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_01_factual_response(client: TestClient):
    """TEST 1: Correct factual response.

    Expected:
    - Low hallucination probability (<= 0.30)
    - Risk level LOW
    - Next action ACCEPT
    - Execution diagnostics: model_loaded=True, inference_executed=True, degraded=False
    """
    payload = {
        "user_query": "What is the capital of France?",
        "llm_response": "The capital of France is Paris.",
    }
    response = client.post("/detect", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert data["hallucination_probability"] <= 0.30, (
        f"Factual response should have low hallucination prob, got {data['hallucination_probability']}"
    )
    assert data["risk_level"] == RiskLevel.LOW.value
    assert data["next_action"] == NextAction.ACCEPT.value
    assert data["detector_model_loaded"] is True
    assert data["detector_inference_executed"] is True
    assert data["detector_degraded"] is False
    assert "Manjunath2000006/halluciguard-detector" in data["model_source"]


def test_02_fabricated_response(client: TestClient):
    """TEST 2: Clearly fabricated response.

    Expected:
    - High hallucination probability (>= 0.50)
    - Risk level HIGH
    - Next action VERIFY
    - Execution diagnostics: model_loaded=True, inference_executed=True, degraded=False
    """
    payload = {
        "user_query": "Who wrote Romeo and Juliet?",
        "llm_response": "Romeo and Juliet was written by Albert Einstein in 1920 in Germany.",
    }
    response = client.post("/detect", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert data["hallucination_probability"] >= 0.50, (
        f"Fabricated response should have high hallucination prob, got {data['hallucination_probability']}"
    )
    assert data["risk_level"] == RiskLevel.HIGH.value
    assert data["next_action"] == NextAction.VERIFY.value
    assert data["detector_model_loaded"] is True
    assert data["detector_inference_executed"] is True
    assert data["detector_degraded"] is False


def test_03_empty_query(client: TestClient):
    """TEST 3: Empty query.

    Expected:
    - Safe validation response (422 Unprocessable Entity due to min_length=1)
    - Or safe fallback if called directly on agent
    """
    # API schema validation test
    payload = {
        "user_query": "",
        "llm_response": "The capital of France is Paris.",
    }
    response = client.post("/detect", json=payload)
    assert response.status_code == 422, "API should reject empty query with validation error"

    # Agent internal fallback test
    agent_result = detector_agent.detect(user_query="", llm_response="Valid response")
    assert agent_result.detector_degraded is True
    assert agent_result.next_action == NextAction.ACCEPT


def test_04_empty_response(client: TestClient):
    """TEST 4: Empty response.

    Expected:
    - Safe validation response (422 Unprocessable Entity due to min_length=1)
    - Or safe fallback if called directly on agent
    """
    # API schema validation test
    payload = {
        "user_query": "What is the capital of France?",
        "llm_response": "",
    }
    response = client.post("/detect", json=payload)
    assert response.status_code == 422, "API should reject empty response with validation error"

    # Agent internal fallback test
    agent_result = detector_agent.detect(user_query="Valid query", llm_response="")
    assert agent_result.detector_degraded is True
    assert agent_result.next_action == NextAction.ACCEPT


def test_05_model_loading_failure():
    """TEST 5: Model loading failure / fallback behavior.

    Expected:
    - When initialized with an invalid path, the agent handles it safely.
    - Result has detector_degraded=True, detector_inference_executed=False.
    """
    bad_config = DetectorConfig(
        halueval_model_path="nonexistent/model_repo_that_does_not_exist_12345"
    )
    # Create isolated agent with bad model path
    bad_agent = DetectorAgent(config=bad_config)
    # Force fresh inference instance
    from staging.detector.halueval_inference import HaluEvalInference
    bad_agent._inference = HaluEvalInference(model_path=bad_config.halueval_model_path)
    bad_agent._SHARED_MODEL_LOADED = False

    result = bad_agent.detect(
        user_query="Is this working?",
        llm_response="This should trigger degraded mode."
    )
    assert result.detector_degraded is True
    assert result.detector_inference_executed is False
    assert result.detector_model_loaded is False
    assert "baseline" in result.detector_model_source or "default" in result.detector_model_source


def test_06_normal_successful_inference_diagnostics(client: TestClient):
    """TEST 6: Normal successful inference diagnostics with real Hugging Face model.

    Expected:
    - detector_model_loaded=True
    - detector_inference_executed=True
    - detector_degraded=False
    - Real Hugging Face provenance
    """
    payload = {
        "user_query": "Summarize the following document.",
        "llm_response": "The president visited Paris on Monday.",
        "context": "The president visited Paris on Monday to discuss international trade."
    }
    response = client.post("/detect", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["detector_model_loaded"] is True
    assert data["detector_inference_executed"] is True
    assert data["detector_degraded"] is False
    assert data["confidence_score"] > 0.70
    assert data["hallucination_probability"] < 0.30
    assert data["risk_level"] == RiskLevel.LOW.value
    assert data["next_action"] == NextAction.ACCEPT.value
    assert "Manjunath2000006/halluciguard-detector" in data["model_source"]
    assert "Manjunath2000006/halluciguard-detector" in data["detector_model_source"]


def test_07_health_endpoint(client: TestClient):
    """Test GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "Manjunath2000006/halluciguard-detector" in data["model_source"]


def test_08_model_info_endpoint(client: TestClient):
    """Test GET /model-info endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "Manjunath2000006/halluciguard-detector" in data["model_repository"]
    assert data["model_loaded"] is True
    assert data["max_sequence_length"] == 384
    assert data["label_mapping"]["0"] == "NO_HALLUCINATION"
    assert data["label_mapping"]["1"] == "HALLUCINATION"
    assert data["risk_thresholds"]["low_risk_threshold"] == 0.30
    assert data["risk_thresholds"]["high_risk_threshold"] == 0.50
    assert data["routing_policy"]["LOW"] == "ACCEPT"
    assert data["routing_policy"]["MEDIUM"] == "ACCEPT"
    assert data["routing_policy"]["HIGH"] == "VERIFY"
