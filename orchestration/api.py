from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .graph import ACTIVE_AGENTS, DISABLED_AGENTS, run_verification
from .runtime_validation import validate_orchestration_startup


class VerificationRequest(BaseModel):
    user_query: str = Field(min_length=1)
    llm_response: str = Field(min_length=1)
    domain: str = "general"
    request_id: str | None = None


app = FastAPI(
    title="HalluciGuard Verification Engine",
    version="2.0.0",
    description="LangGraph supervisor for the active Detector → Verifier → Memory trust path.",
)


@app.get("/health")
async def health() -> dict:
    validation = validate_orchestration_startup()
    return {
        "status": "ok" if validation["ok"] else "degraded",
        "engine": "langgraph",
        "active_agents": list(ACTIVE_AGENTS),
        "disabled_agents": list(DISABLED_AGENTS),
        "judge": {"enabled": False, "status": "not_executed"},
        "corrector": {"enabled": False, "status": "not_executed"},
        "runtime_validation": validation,
    }


@app.post("/verify")
async def verify(request: VerificationRequest) -> dict:
    try:
        result = await run_verification(
            user_query=request.user_query,
            llm_response=request.llm_response,
            domain=request.domain,
            request_id=request.request_id,
        )
        return {
            "execution_id": result.get("execution_id"),
            "request_id": result.get("request_id"),
            "final_response": result.get("final_response", result.get("llm_response")),
            "terminal_status": result.get("terminal_status"),
            "verification_status": result.get("verification_status"),
            "active_agents": list(ACTIVE_AGENTS),
            "disabled_agents": list(DISABLED_AGENTS),
            "judge": {"enabled": False, "status": "not_executed"},
            "corrector": {"enabled": False, "status": "not_executed"},
            "detector": result.get("detector"),
            "verifier": result.get("verifier"),
            "memory": result.get("memory"),
            "retry_count": result.get("retry_count", 0),
            "errors": result.get("errors", []),
            "audit": result.get("audit", {}),
            "inter_agent_bus": result.get("inter_agent_bus", []),
            "trace": result.get("trace", []),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Verification engine failed: {type(exc).__name__}: {exc}",
        ) from exc
