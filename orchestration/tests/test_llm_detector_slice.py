from __future__ import annotations

import json
from unittest.mock import MagicMock
import httpx
import pytest

from agents.detector_agent import DetectionResult, DetectorAgent, NextAction, RiskLevel
from services.base_llm_service import BaseLLMConfig, BaseLLMService, GenerationResult
from services.llm_detector_service import BaseLLMDetectorService


class StubBaseLLMService(BaseLLMService):
    def __init__(self, generation_result: GenerationResult):
        super().__init__(BaseLLMConfig(api_key="test-key"))
        self._stub_result = generation_result
        self.calls = 0

    async def generate(self, user_query: str, **kwargs) -> GenerationResult:
        self.calls += 1
        return self._stub_result


class DummyDetectorAgent(DetectorAgent):
    def __init__(self, detection_result: DetectionResult):
        self._stub_result = detection_result
        self.recorded_query = None
        self.recorded_response = None
        self.detect_calls = 0

    def detect(self, user_query: str, llm_response: str) -> DetectionResult:
        self.detect_calls += 1
        self.recorded_query = user_query
        self.recorded_response = llm_response
        return self._stub_result


@pytest.mark.asyncio
async def test_successful_generation_reaches_detector():
    gen_result = GenerationResult(
        user_query="Explain OS in one full paragraph",
        draft_response="An operating system is system software that manages hardware and software resources.",
        model="qwen/qwen-2.5-7b-instruct",
        provider="openrouter",
        generation_mode="normal",
        mode="normal",
        temperature=0.7,
        latency_ms=2100,
        finish_reason="stop",
        request_id="req-123",
        status="success",
    )
    det_result = DetectionResult(
        confidence_score=0.95,
        hallucination_probability=0.05,
        risk_level=RiskLevel.LOW,
        next_action=NextAction.ACCEPT,
        model_source="halueval-distilbert",
    )

    llm_stub = StubBaseLLMService(gen_result)
    det_stub = DummyDetectorAgent(det_result)
    service = BaseLLMDetectorService(llm_service=llm_stub, detector_agent=det_stub)

    result = await service.execute_slice("Explain OS in one full paragraph")

    assert result.user_query == "Explain OS in one full paragraph"
    assert result.draft_response == gen_result.draft_response
    assert result.generation["status"] == "success"
    assert result.generation["model"] == "qwen/qwen-2.5-7b-instruct"
    assert result.detector is not None
    assert result.detector["hallucination_probability"] == 0.05
    assert result.detector["risk_tier"] == "LOW"
    assert result.detector["decision"] == "ACCEPT"


@pytest.mark.asyncio
async def test_detector_receives_exact_generated_response():
    gen_result = GenerationResult(
        user_query="What is gravity?",
        draft_response="Gravity is a fundamental interaction that pulls objects together.",
        model="qwen/qwen-2.5-7b-instruct",
        provider="openrouter",
        generation_mode="normal",
        mode="normal",
        temperature=0.7,
        latency_ms=1500,
        finish_reason="stop",
        request_id="req-456",
        status="success",
    )
    det_result = DetectionResult(
        confidence_score=0.90,
        hallucination_probability=0.10,
        risk_level=RiskLevel.LOW,
        next_action=NextAction.ACCEPT,
    )

    llm_stub = StubBaseLLMService(gen_result)
    det_stub = DummyDetectorAgent(det_result)
    service = BaseLLMDetectorService(llm_service=llm_stub, detector_agent=det_stub)

    await service.execute_slice("What is gravity?")

    assert det_stub.detect_calls == 1
    assert det_stub.recorded_query == "What is gravity?"
    assert det_stub.recorded_response == "Gravity is a fundamental interaction that pulls objects together."


@pytest.mark.asyncio
async def test_detector_output_preserved_without_transformation():
    gen_result = GenerationResult(
        user_query="Claim test",
        draft_response="Unverified claim statement.",
        model="qwen/qwen-2.5-7b-instruct",
        provider="openrouter",
        generation_mode="normal",
        mode="normal",
        temperature=0.7,
        latency_ms=1800,
        finish_reason="stop",
        request_id="req-789",
        status="success",
    )
    det_result = DetectionResult(
        confidence_score=0.40,
        hallucination_probability=0.75,
        risk_level=RiskLevel.HIGH,
        next_action=NextAction.VERIFY,
        model_source="halueval-distilbert",
    )

    llm_stub = StubBaseLLMService(gen_result)
    det_stub = DummyDetectorAgent(det_result)
    service = BaseLLMDetectorService(llm_service=llm_stub, detector_agent=det_stub)

    result = await service.execute_slice("Claim test")

    assert result.detector["confidence_score"] == 0.40
    assert result.detector["hallucination_probability"] == 0.75
    assert result.detector["risk_tier"] == "HIGH"
    assert result.detector["decision"] == "VERIFY"
    assert result.detector["model_source"] == "halueval-distilbert"


@pytest.mark.asyncio
async def test_llm_generation_failure_prevents_detector_execution():
    gen_result = GenerationResult(
        user_query="Test failure",
        draft_response="",
        model="qwen/qwen-2.5-7b-instruct",
        provider="openrouter",
        generation_mode="normal",
        mode="normal",
        temperature=0.7,
        latency_ms=10,
        finish_reason=None,
        request_id="req-fail",
        status="failed",
        error="OPENROUTER_API_KEY is not configured",
        error_code="MISSING_API_KEY",
    )
    det_result = DetectionResult(
        confidence_score=0.90,
        hallucination_probability=0.10,
        risk_level=RiskLevel.LOW,
        next_action=NextAction.ACCEPT,
    )

    llm_stub = StubBaseLLMService(gen_result)
    det_stub = DummyDetectorAgent(det_result)
    service = BaseLLMDetectorService(llm_service=llm_stub, detector_agent=det_stub)

    result = await service.execute_slice("Test failure")

    assert result.generation["status"] == "failed"
    assert result.generation["error_code"] == "MISSING_API_KEY"
    assert result.detector is None
    assert det_stub.detect_calls == 0


@pytest.mark.asyncio
async def test_empty_generated_response_prevents_detector_execution():
    gen_result = GenerationResult(
        user_query="Empty output",
        draft_response="   ",
        model="qwen/qwen-2.5-7b-instruct",
        provider="openrouter",
        generation_mode="normal",
        mode="normal",
        temperature=0.7,
        latency_ms=100,
        finish_reason="stop",
        request_id="req-empty",
        status="success",
    )
    det_result = DetectionResult(
        confidence_score=0.90,
        hallucination_probability=0.10,
        risk_level=RiskLevel.LOW,
        next_action=NextAction.ACCEPT,
    )

    llm_stub = StubBaseLLMService(gen_result)
    det_stub = DummyDetectorAgent(det_result)
    service = BaseLLMDetectorService(llm_service=llm_stub, detector_agent=det_stub)

    result = await service.execute_slice("Empty output")

    assert result.detector is None
    assert det_stub.detect_calls == 0


@pytest.mark.asyncio
async def test_final_contract_is_json_serializable():
    gen_result = GenerationResult(
        user_query="JSON check",
        draft_response="Serializable content.",
        model="qwen/qwen-2.5-7b-instruct",
        provider="openrouter",
        generation_mode="normal",
        mode="normal",
        temperature=0.7,
        latency_ms=1200,
        finish_reason="stop",
        request_id="req-json",
        status="success",
    )
    det_result = DetectionResult(
        confidence_score=0.88,
        hallucination_probability=0.12,
        risk_level=RiskLevel.LOW,
        next_action=NextAction.ACCEPT,
        model_source="halueval-distilbert",
    )

    llm_stub = StubBaseLLMService(gen_result)
    det_stub = DummyDetectorAgent(det_result)
    service = BaseLLMDetectorService(llm_service=llm_stub, detector_agent=det_stub)

    result = await service.execute_slice("JSON check")
    dumped = result.model_dump()
    serialized = json.dumps(dumped)

    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)
    assert deserialized["user_query"] == "JSON check"
    assert deserialized["detector"]["decision"] == "ACCEPT"
