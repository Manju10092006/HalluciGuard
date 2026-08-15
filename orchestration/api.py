from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.base_llm_service import BaseLLMService

from .graph import run_verification
from .runtime_validation import validate_orchestration_startup


class VerificationRequest(BaseModel):
    user_query: str = Field(min_length=1, description="The user's prompt or question to verify.")
    generation_mode: Literal["normal", "stress_test"] = Field(default="normal", description="'normal' or 'stress_test'.")
    llm_response: Optional[str] = Field(default=None, description="Optional pre-supplied draft response.")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="Optional conversation context.")
    domain: str = Field(default="general", description="Verification domain (e.g. general, biomedical, finance).")
    request_id: Optional[str] = Field(default=None, description="Optional caller correlation ID.")


app = FastAPI(
    title="HalluciGuard Verification Engine",
    version="2.0.0",
    description="Production LangGraph supervisor orchestrating OpenRouter Base LLM, Detector, Verifier, and Memory agents.",
)

# Configure CORS for Vercel Frontend and Local Development
raw_cors = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,https://*.vercel.app",
)
allowed_origins = [origin.strip() for origin in raw_cors.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if "*" not in allowed_origins else ["*"],
    allow_origin_regex=r"https://.*\.vercel\.app" if "https://*.vercel.app" in allowed_origins else None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "service": "HalluciGuard Verification Engine API",
        "status": "online",
        "version": "2.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "verify_endpoint": "/verify",
        "frontend_url": "https://frontend-alpha-umber-63.vercel.app",
    }


@app.get("/health")
async def health(deep: bool = False) -> Dict[str, Any]:
    if not deep:
        return {
            "status": "healthy",
            "backend_status": "healthy",
            "environment": os.environ.get("HALLUCIGUARD_ENV", "production"),
            "engine": "langgraph_production_supervisor",
            "active_agents": ["generator", "supervisor", "detector", "verifier", "memory"],
            "disabled_agents": {
                "judge": {"enabled": False, "status": "not_executed"},
                "corrector": {"enabled": False, "status": "not_executed"},
            },
        }

    validation = validate_orchestration_startup()
    try:
        base_llm = (await BaseLLMService().health(check_network=True)).model_dump()
    except Exception as exc:
        base_llm = {"available": False, "provider": "openrouter", "error": str(exc)}

    return {
        "status": "healthy" if validation.get("ok") and base_llm.get("available") else "degraded",
        "backend_status": "healthy" if validation.get("ok") and base_llm.get("available") else "degraded",
        "environment": os.environ.get("HALLUCIGUARD_ENV", "production"),
        "engine": "langgraph_production_supervisor",
        "active_agents": ["generator", "supervisor", "detector", "verifier", "memory"],
        "disabled_agents": {
            "judge": {"enabled": False, "status": "not_executed"},
            "corrector": {"enabled": False, "status": "not_executed"},
        },
        "base_llm": base_llm,
        "detector": validation.get("detector", {}),
        "verifier": validation.get("verifier", {}),
        "memory": validation.get("memory", {}),
        "runtime_validation": validation,
    }


def _total_latency_ms(result: Dict[str, Any]) -> int:
    return sum(
        int(event.get("latency_ms", 0) or 0) for event in result.get("trace", [])
    )


async def _execute_verification(request: VerificationRequest) -> Dict[str, Any]:
    try:
        result = await run_verification(
            user_query=request.user_query,
            llm_response=request.llm_response or "",
            domain=request.domain,
            request_id=request.request_id,
            generation_mode=request.generation_mode,
            conversation_history=request.conversation_history,
        )
        return {
            "execution_id": result.get("execution_id"),
            "request_id": result.get("request_id"),
            "generation": result.get("generation") or result.get("base_llm"),
            "draft_response": result.get("draft_response") or result.get("llm_response"),
            "final_response": result.get("final_response") or result.get("draft_response", result.get("llm_response")),
            "terminal_status": result.get("terminal_status", "accepted"),
            "verification_status": result.get("verification_status", "unverified"),
            "total_latency_ms": _total_latency_ms(result),
            "detector": result.get("detector"),
            "verifier": result.get("verifier"),
            "memory": result.get("memory"),
            "active_agents": ["generator", "supervisor", "detector", "verifier", "memory"],
            "disabled_agents": {
                "judge": {"enabled": False, "status": "not_executed"},
                "corrector": {"enabled": False, "status": "not_executed"},
            },
            "judge": {"enabled": False, "status": "not_executed"},
            "corrector": {"enabled": False, "status": "not_executed"},
            "retry_count": result.get("retry_count", 0),
            "errors": result.get("errors", []),
            "inter_agent_bus": result.get("inter_agent_bus", []),
            "trace": result.get("trace", []),
            "audit": result.get("audit", {}),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Verification engine failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/verify")
async def verify(request: VerificationRequest) -> Dict[str, Any]:
    return await _execute_verification(request)


@app.post("/api/v1/verify")
async def verify_v1(request: VerificationRequest) -> Dict[str, Any]:
    return await _execute_verification(request)
