from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.base_llm_service import BaseLLMService

from .graph import run_verification
from .runtime_validation import validate_orchestration_startup


class VerificationRequest(BaseModel):
    user_query: str = Field(min_length=1)
    llm_response: str | None = None
    generation_mode: Literal["normal", "stress_test"] = "normal"
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    domain: str = "general"
    request_id: str | None = None


app = FastAPI(
    title="HalluciGuard Verification Engine",
    version="1.1.0",
    description="LangGraph supervisor for Generator, Detector, Verifier, and Memory. Judge/Corrector are retained but disabled.",
)


@app.get("/health")
async def health() -> dict:
    validation = validate_orchestration_startup()
    base_llm = (await BaseLLMService().health(check_network=True)).model_dump()
    return {
        "status": "ok" if validation["ok"] else "degraded",
        "engine": "langgraph",
        "active_agents": ["generator", "supervisor", "detector", "verifier", "memory"],
        "disabled_agents": {
            "judge": {"enabled": False, "status": "not_executed"},
            "corrector": {"enabled": False, "status": "not_executed"},
        },
        "runtime_validation": validation,
        "base_llm": base_llm,
    }


def _total_latency_ms(result: dict) -> int:
    return sum(
        int(event.get("latency_ms", 0) or 0) for event in result.get("trace", [])
    )


def _response(result: dict) -> dict:
    return {
        "execution_id": result.get("execution_id"),
        "request_id": result.get("request_id"),
        "generation": result.get("generation"),
        "draft_response": result.get("draft_response") or result.get("llm_response"),
        "final_response": result.get(
            "final_response", result.get("draft_response", result.get("llm_response"))
        ),
        "terminal_status": result.get("terminal_status"),
        "verification_status": result.get("verification_status"),
        "total_latency_ms": _total_latency_ms(result),
        "detector": result.get("detector"),
        "verifier": result.get("verifier"),
        "memory": result.get("memory"),
        "active_agents": result.get("active_agents", []),
        "disabled_agents": result.get("disabled_agents", {}),
        "judge": {"enabled": False, "status": "not_executed"},
        "corrector": {"enabled": False, "status": "not_executed"},
        "retry_count": result.get("retry_count", 0),
        "errors": result.get("errors", []),
        "audit": result.get("audit", {}),
        "trace": result.get("trace", []),
        "inter_agent_bus": result.get("inter_agent_bus", []),
    }


async def _run(request: VerificationRequest) -> dict:
    result = await run_verification(
        user_query=request.user_query,
        llm_response=request.llm_response,
        domain=request.domain,
        request_id=request.request_id,
        generation_mode=request.generation_mode,
        conversation_history=request.conversation_history,
    )
    return _response(result)


@app.post("/verify")
async def verify(request: VerificationRequest) -> dict:
    try:
        return await _run(request)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Verification engine failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/api/v1/verify")
async def verify_v1(request: VerificationRequest) -> dict:
    return await verify(request)
