from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from agents.detector_agent import DetectionResult, DetectorAgent, NextAction, RiskLevel
from services.base_llm_service import BaseLLMConfig, BaseLLMService, GenerationResult
from services.llm_detector_verifier_service import BaseLLMDetectorVerifierService


class StubBaseLLMService(BaseLLMService):
    def __init__(self, generation_result: GenerationResult):
        super().__init__(BaseLLMConfig(api_key="test-key"))
        self._stub_result = generation_result
        self.calls = 0

    async def generate(self, user_query: str, **kwargs) -> GenerationResult:
        self.calls += 1
        return self._stub_result


class DummyDetectorAgent(DetectorAgent):
    def __init__(self, detection_result: DetectionResult | Exception):
        self._stub_result = detection_result
        self.detect_calls = 0

    def detect(self, user_query: str, llm_response: str) -> DetectionResult:
        self.detect_calls += 1
        if isinstance(self._stub_result, Exception):
            raise self._stub_result
        return self._stub_result


def make_mock_verifier_output(retrieved=5, verified=2, confidence=0.85):
    mock_ev = MagicMock()
    mock_ev.title = "Wikipedia Article"
    mock_ev.source = "wikipedia"
    mock_ev.url = "https://en.wikipedia.org/wiki/OS"
    mock_ev.publication_date = "2024-01-01"
    mock_ev.snippet = "An operating system manages hardware."
    mock_ev.entailment_label.value = "entailment"
    mock_ev.entailment_score = 0.94
    mock_ev.credibility_score = 0.92

    mock_claim = MagicMock()
    mock_claim.claim_id = "c1"
    mock_claim.claim_text = "An operating system manages hardware resources."
    mock_claim.verdict.value = "verified"
    mock_claim.support_score = 0.94
    mock_claim.contradiction_score = 0.02
    mock_claim.trust_score = 0.92
    mock_claim.confidence_score = 0.88
    mock_claim.explanation = "Corroborated by Wikipedia."
    mock_claim.evidence = [mock_ev]

    mock_out = MagicMock()
    mock_out.domain = "general"
    mock_out.domain_validated = True
    mock_out.retrieved_sources = retrieved
    mock_out.verified_sources = verified
    mock_out.overall_evidence_confidence = confidence
    mock_out.latency_ms = 1200
    mock_out.cache_hit = False
    mock_out.claim_evidence = [mock_claim]

    return mock_out


@pytest.mark.asyncio
async def test_low_detector_result_skips_verifier():
    gen_result = GenerationResult(
        user_query="Low risk question",
        draft_response="Standard low risk text.",
        model="qwen/qwen-2.5-7b-instruct",
        provider="openrouter",
        generation_mode="normal",
        mode="normal",
        temperature=0.7,
        latency_ms=1000,
        finish_reason="stop",
        request_id="req-low",
        status="success",
    )
    det_result = DetectionResult(
        confidence_score=0.98,
        hallucination_probability=0.02,
        risk_level=RiskLevel.LOW,
        next_action=NextAction.ACCEPT,
        model_source="halueval-distilbert",
    )

    mock_verifier_pipeline = AsyncMock()
    mock_verifier_pipeline.verify.return_value = make_mock_verifier_output()

    llm_stub = StubBaseLLMService(gen_result)
    det_stub = DummyDetectorAgent(det_result)
    service = BaseLLMDetectorVerifierService(
        llm_service=llm_stub,
        detector_agent=det_stub,
        verifier_pipeline=mock_verifier_pipeline,
    )

    result = await service.execute_slice("Low risk question")

    assert result.detector["risk_tier"] == "LOW"
    assert result.detector["decision"] == "ACCEPT"
    assert result.verifier["executed"] is False
    assert "skipped" in result.verifier["reason"].lower()
    mock_verifier_pipeline.verify.assert_not_called()


@pytest.mark.asyncio
async def test_medium_detector_result_invokes_verifier():
    gen_result = GenerationResult(
        user_query="Medium risk question",
        draft_response="Medium risk text requiring evidence check.",
        model="qwen/qwen-2.5-7b-instruct",
        provider="openrouter",
        generation_mode="normal",
        mode="normal",
        temperature=0.7,
        latency_ms=1100,
        finish_reason="stop",
        request_id="req-med",
        status="success",
    )
    det_result = DetectionResult(
        confidence_score=0.60,
        hallucination_probability=0.40,
        risk_level=RiskLevel.MEDIUM,
        next_action=NextAction.ACCEPT,
        model_source="halueval-distilbert",
    )

    mock_verifier_pipeline = AsyncMock()
    mock_verifier_pipeline.verify.return_value = make_mock_verifier_output()

    llm_stub = StubBaseLLMService(gen_result)
    det_stub = DummyDetectorAgent(det_result)
    service = BaseLLMDetectorVerifierService(
        llm_service=llm_stub,
        detector_agent=det_stub,
        verifier_pipeline=mock_verifier_pipeline,
    )

    result = await service.execute_slice("Medium risk question")

    assert result.detector["risk_tier"] == "MEDIUM"
    assert result.verifier["executed"] is True
    assert result.verifier["retrieved_sources"] == 5
    mock_verifier_pipeline.verify.assert_called_once()


@pytest.mark.asyncio
async def test_high_detector_result_invokes_verifier():
    gen_result = GenerationResult(
        user_query="High risk question",
        draft_response="High risk text with hallucinated claims.",
        model="qwen/qwen-2.5-7b-instruct",
        provider="openrouter",
        generation_mode="normal",
        mode="normal",
        temperature=0.7,
        latency_ms=1200,
        finish_reason="stop",
        request_id="req-high",
        status="success",
    )
    det_result = DetectionResult(
        confidence_score=0.20,
        hallucination_probability=0.80,
        risk_level=RiskLevel.HIGH,
        next_action=NextAction.VERIFY,
        model_source="halueval-distilbert",
    )

    mock_verifier_pipeline = AsyncMock()
    mock_verifier_pipeline.verify.return_value = make_mock_verifier_output()

    llm_stub = StubBaseLLMService(gen_result)
    det_stub = DummyDetectorAgent(det_result)
    service = BaseLLMDetectorVerifierService(
        llm_service=llm_stub,
        detector_agent=det_stub,
        verifier_pipeline=mock_verifier_pipeline,
    )

    result = await service.execute_slice("High risk question")

    assert result.detector["risk_tier"] == "HIGH"
    assert result.detector["decision"] == "VERIFY"
    assert result.verifier["executed"] is True
    assert result.verifier["verified_sources"] == 2
    mock_verifier_pipeline.verify.assert_called_once()


@pytest.mark.asyncio
async def test_detector_failure_does_not_invoke_verifier():
    gen_result = GenerationResult(
        user_query="Detector error test",
        draft_response="Valid draft response.",
        model="qwen/qwen-2.5-7b-instruct",
        provider="openrouter",
        generation_mode="normal",
        mode="normal",
        temperature=0.7,
        latency_ms=800,
        finish_reason="stop",
        request_id="req-det-fail",
        status="success",
    )

    mock_verifier_pipeline = AsyncMock()

    llm_stub = StubBaseLLMService(gen_result)
    det_stub = DummyDetectorAgent(RuntimeError("Detector model crash"))
    service = BaseLLMDetectorVerifierService(
        llm_service=llm_stub,
        detector_agent=det_stub,
        verifier_pipeline=mock_verifier_pipeline,
    )

    result = await service.execute_slice("Detector error test")

    assert result.detector["status"] == "failed"
    assert "Detector model crash" in result.detector["error"]
    assert result.verifier is None
    mock_verifier_pipeline.verify.assert_not_called()


@pytest.mark.asyncio
async def test_empty_draft_does_not_invoke_verifier():
    gen_result = GenerationResult(
        user_query="Empty draft test",
        draft_response="",
        model="qwen/qwen-2.5-7b-instruct",
        provider="openrouter",
        generation_mode="normal",
        mode="normal",
        temperature=0.7,
        latency_ms=100,
        finish_reason="stop",
        request_id="req-empty-draft",
        status="success",
    )

    mock_verifier_pipeline = AsyncMock()

    llm_stub = StubBaseLLMService(gen_result)
    det_stub = DummyDetectorAgent(
        DetectionResult(
            confidence_score=0.9,
            hallucination_probability=0.1,
            risk_level=RiskLevel.LOW,
            next_action=NextAction.ACCEPT,
        )
    )
    service = BaseLLMDetectorVerifierService(
        llm_service=llm_stub,
        detector_agent=det_stub,
        verifier_pipeline=mock_verifier_pipeline,
    )

    result = await service.execute_slice("Empty draft test")

    assert result.detector is None
    assert result.verifier is None
    mock_verifier_pipeline.verify.assert_not_called()


@pytest.mark.asyncio
async def test_verifier_failure_handled_cleanly():
    gen_result = GenerationResult(
        user_query="Verifier error test",
        draft_response="High risk draft.",
        model="qwen/qwen-2.5-7b-instruct",
        provider="openrouter",
        generation_mode="normal",
        mode="normal",
        temperature=0.7,
        latency_ms=1000,
        finish_reason="stop",
        request_id="req-ver-fail",
        status="success",
    )
    det_result = DetectionResult(
        confidence_score=0.10,
        hallucination_probability=0.90,
        risk_level=RiskLevel.HIGH,
        next_action=NextAction.VERIFY,
    )

    mock_verifier_pipeline = AsyncMock()
    mock_verifier_pipeline.verify.side_effect = TimeoutError("Retrieval API timed out")

    llm_stub = StubBaseLLMService(gen_result)
    det_stub = DummyDetectorAgent(det_result)
    service = BaseLLMDetectorVerifierService(
        llm_service=llm_stub,
        detector_agent=det_stub,
        verifier_pipeline=mock_verifier_pipeline,
    )

    result = await service.execute_slice("Verifier error test")

    assert result.verifier["executed"] is True
    assert result.verifier["status"] == "failed"
    assert "Retrieval API timed out" in result.verifier["error"]
    assert result.verifier["claim_evidence"] == []


@pytest.mark.asyncio
async def test_existing_verifier_output_preserved():
    gen_result = GenerationResult(
        user_query="Output preservation check",
        draft_response="Draft text to verify.",
        model="qwen/qwen-2.5-7b-instruct",
        provider="openrouter",
        generation_mode="normal",
        mode="normal",
        temperature=0.7,
        latency_ms=1500,
        finish_reason="stop",
        request_id="req-pres",
        status="success",
    )
    det_result = DetectionResult(
        confidence_score=0.30,
        hallucination_probability=0.70,
        risk_level=RiskLevel.HIGH,
        next_action=NextAction.VERIFY,
    )

    mock_verifier_pipeline = AsyncMock()
    mock_verifier_pipeline.verify.return_value = make_mock_verifier_output(
        retrieved=10, verified=4, confidence=0.88
    )

    llm_stub = StubBaseLLMService(gen_result)
    det_stub = DummyDetectorAgent(det_result)
    service = BaseLLMDetectorVerifierService(
        llm_service=llm_stub,
        detector_agent=det_stub,
        verifier_pipeline=mock_verifier_pipeline,
    )

    result = await service.execute_slice("Output preservation check")

    assert result.verifier["executed"] is True
    assert result.verifier["retrieved_sources"] == 10
    assert result.verifier["verified_sources"] == 4
    assert result.verifier["overall_evidence_confidence"] == 0.88
    claims = result.verifier["claim_evidence"]
    assert len(claims) == 1
    assert claims[0]["verdict"] == "verified"
    assert len(claims[0]["evidence"]) == 1
    assert claims[0]["evidence"][0]["source"] == "wikipedia"
    assert claims[0]["evidence"][0]["entailment_label"] == "entailment"


@pytest.mark.asyncio
async def test_step3_result_is_json_serializable():
    gen_result = GenerationResult(
        user_query="JSON test",
        draft_response="Draft text.",
        model="qwen/qwen-2.5-7b-instruct",
        provider="openrouter",
        generation_mode="normal",
        mode="normal",
        temperature=0.7,
        latency_ms=1000,
        finish_reason="stop",
        request_id="req-json",
        status="success",
    )
    det_result = DetectionResult(
        confidence_score=0.40,
        hallucination_probability=0.60,
        risk_level=RiskLevel.HIGH,
        next_action=NextAction.VERIFY,
    )

    mock_verifier_pipeline = AsyncMock()
    mock_verifier_pipeline.verify.return_value = make_mock_verifier_output()

    llm_stub = StubBaseLLMService(gen_result)
    det_stub = DummyDetectorAgent(det_result)
    service = BaseLLMDetectorVerifierService(
        llm_service=llm_stub,
        detector_agent=det_stub,
        verifier_pipeline=mock_verifier_pipeline,
    )

    result = await service.execute_slice("JSON test")
    dumped = result.model_dump()
    serialized = json.dumps(dumped)

    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)
    assert deserialized["user_query"] == "JSON test"
    assert deserialized["detector"]["risk_tier"] == "HIGH"
    assert deserialized["verifier"]["executed"] is True
    assert deserialized["verifier"]["claim_evidence"][0]["verdict"] == "verified"
