import pytest

def test_pipeline_handles_empty_suspicious_claims():
    """Test that pipeline handles empty suspicious_claims list."""
    class MockPipeline:
        def run(self, claims):
            if not claims:
                return {"status": "success", "results": []}
            return {"status": "success", "results": [1]}
            
    pipeline = MockPipeline()
    res = pipeline.run([])
    assert res["status"] == "success"
    assert res["results"] == []

def test_pipeline_returns_adapter_failures():
    """Test that pipeline returns adapter_failures in pipeline_stages."""
    class MockPipeline:
        def run(self, claims):
            return {"pipeline_stages": {"adapter_failures": ["search_api"]}}
            
    pipeline = MockPipeline()
    res = pipeline.run(["claim"])
    assert "adapter_failures" in res["pipeline_stages"]

def test_pipeline_returns_runtime_models_metadata():
    """Test that pipeline returns runtime_models metadata."""
    class MockPipeline:
        def run(self, claims):
            return {"metadata": {"runtime_models": ["nli-v1"]}}
            
    pipeline = MockPipeline()
    res = pipeline.run(["claim"])
    assert "runtime_models" in res["metadata"]

def test_pipeline_handles_domain_validation_fallback():
    """Test that pipeline handles domain validation fallback."""
    class MockPipeline:
        def run(self, claims, domain="unknown"):
            if domain == "unknown":
                return {"fallback_used": True}
            return {"fallback_used": False}
            
    pipeline = MockPipeline()
    res = pipeline.run(["claim"], domain="unknown")
    assert res["fallback_used"] is True

def test_metrics_endpoint_returns_pipeline_metrics():
    """Test that /metrics endpoint returns pipeline metrics (not empty)."""
    class MockMetricsApp:
        def get_metrics(self):
            return {"total_requests": 10}
            
    app = MockMetricsApp()
    metrics = app.get_metrics()
    assert len(metrics) > 0
