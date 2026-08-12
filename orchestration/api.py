from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .graph import run_verification


class VerificationRequest(BaseModel):
    user_query: str = Field(min_length=1)
    llm_response: str = Field(min_length=1)
    domain: str = "general"
    request_id: str | None = None


app = FastAPI(
    title="HalluciGuard Verification Engine",
    version="1.0.0",
    description="LangGraph supervisor for the five-agent HalluciGuard verification workflow.",
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "engine": "langgraph", "agents": ["detector", "verifier", "judge", "corrector", "memory"]}


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
            "request_id": result.get("request_id"),
            "final_response": result.get("final_response", result.get("llm_response")),
            "detector": result.get("detector"),
            "verifier": result.get("verifier"),
            "judge": result.get("judge"),
            "corrector": result.get("corrector"),
            "memory": result.get("memory"),
            "retry_count": result.get("retry_count", 0),
            "trace": result.get("trace", []),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Verification engine failed: {type(exc).__name__}: {exc}") from exc
