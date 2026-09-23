"""
Tests for API endpoints.
"""
import pytest
import sys, os

# Add outer halluciguard_detector folder to path
detector_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if detector_dir not in sys.path:
    sys.path.insert(0, detector_dir)

from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "halluciguard_detector"

def test_detect_endpoint():
    payload = {
        "request_id": "TEST-100",
        "user_query": "Who created Java?",
        "draft_answer": "Java was created by James Gosling in 1995."
    }
    response = client.post("/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == "TEST-100"
    assert data["routing"] == "VERIFY"
    assert len(data["claims"]) >= 1
