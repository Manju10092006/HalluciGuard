from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv

# Ensure local .env is loaded in development and production
load_dotenv()

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field

from services.base_llm_service import BaseLLMService

from .auth import (
    authenticate_user,
    clear_user_history,
    get_user_by_token,
    get_user_history,
    register_user,
    save_user_history,
)
from .graph import run_verification
from .runtime_validation import validate_orchestration_startup


class VerificationRequest(BaseModel):
    user_query: str = Field(min_length=1, description="The user's prompt or question to verify.")
    generation_mode: Literal["normal", "stress_test"] = Field(default="normal", description="'normal' or 'stress_test'.")
    llm_response: Optional[str] = Field(default=None, description="Optional pre-supplied draft response.")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="Optional conversation context.")
    domain: str = Field(default="general", description="Verification domain (e.g. general, biomedical, finance).")
    request_id: Optional[str] = Field(default=None, description="Optional caller correlation ID.")


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, description="User email address")
    password: str = Field(min_length=6, description="User password (min 6 characters)")
    name: Optional[str] = Field(default=None, description="User display name")


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, description="User email address")
    password: str = Field(min_length=1, description="User password")


class SaveHistoryRequest(BaseModel):
    query: str = Field(min_length=1, description="User query text")
    result: Dict[str, Any] = Field(description="Verification result JSON payload")


app = FastAPI(
    title="HalluciGuard Verification Engine",
    version="2.0.0",
    description="Production LangGraph supervisor orchestrating OpenRouter Base LLM, Detector, Verifier, and Memory agents with JWT Auth.",
)

# Configure CORS for Vercel Frontend and Local Development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://frontend-v2-ashen-five.vercel.app",
        "https://frontend-v2-c7km8aozi-s-manjunath-reddys-projects.vercel.app",
        "https://frontend-alpha-umber-63.vercel.app",
    ],
    allow_origin_regex=r"^https://.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def _get_auth_user(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split("Bearer ", 1)[1].strip()
    if not token:
        return None
    try:
        return get_user_by_token(token)
    except Exception:
        return None


def _require_auth_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    user = _get_auth_user(authorization)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ── Authentication Endpoints ──────────────────────────────────────────

@app.post("/auth/register")
async def auth_register(req: RegisterRequest) -> Dict[str, Any]:
    try:
        user_dict, token = register_user(
            email=req.email,
            password=req.password,
            name=req.name,
        )
        return {
            "status": "success",
            "access_token": token,
            "token_type": "bearer",
            "user": user_dict,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Registration failed: {exc}") from exc


@app.post("/auth/login")
async def auth_login(req: LoginRequest) -> Dict[str, Any]:
    try:
        user_dict, token = authenticate_user(
            email=req.email,
            password=req.password,
        )
        return {
            "status": "success",
            "access_token": token,
            "token_type": "bearer",
            "user": user_dict,
        }
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Login failed: {exc}") from exc


@app.get("/auth/me")
async def auth_me(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    user = _require_auth_user(authorization)
    return {
        "status": "success",
        "user": user,
    }


@app.post("/auth/logout")
async def auth_logout() -> Dict[str, Any]:
    return {"status": "success", "message": "Successfully signed out"}


# ── Authenticated History Endpoints ────────────────────────────────────

@app.get("/api/history")
async def get_history(
    limit: int = 50,
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    user = _require_auth_user(authorization)
    items = get_user_history(user["id"], limit=limit)
    return {"status": "success", "history": items}


@app.post("/api/history")
async def save_history(
    req: SaveHistoryRequest,
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    user = _require_auth_user(authorization)
    saved = save_user_history(user["id"], req.query, req.result)
    return {"status": "success", "record": saved}


@app.delete("/api/history")
async def delete_history(
    history_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    user = _require_auth_user(authorization)
    clear_user_history(user["id"], history_id=history_id)
    return {"status": "success", "message": "History cleared"}


# ── System Endpoints ──────────────────────────────────────────────────

@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "service": "HalluciGuard Verification Engine API",
        "status": "online",
        "version": "2.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "verify_endpoint": "/verify",
        "auth_endpoints": {
            "register": "/auth/register",
            "login": "/auth/login",
            "me": "/auth/me",
            "logout": "/auth/logout",
        },
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
            "active_agents": ["base_llm", "detector", "verifier", "memory"],
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
        "active_agents": ["base_llm", "detector", "verifier", "memory"],
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


async def _execute_verification(
    request: VerificationRequest, authorization: Optional[str] = None
) -> Dict[str, Any]:
    try:
        result = await run_verification(
            user_query=request.user_query,
            llm_response=request.llm_response or "",
            domain=request.domain,
            request_id=request.request_id,
            generation_mode=request.generation_mode,
            conversation_history=request.conversation_history,
        )
        resp = {
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
            "active_agents": ["base_llm", "detector", "verifier", "memory"],
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

        # Auto-save history if caller is authenticated
        if authorization:
            user = _get_auth_user(authorization)
            if user:
                try:
                    save_user_history(user["id"], request.user_query, resp)
                except Exception:
                    pass

        return resp
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Verification engine failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/verify")
async def verify(
    request: VerificationRequest,
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    return await _execute_verification(request, authorization=authorization)


@app.post("/api/v1/verify")
async def verify_v1(
    request: VerificationRequest,
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    return await _execute_verification(request, authorization=authorization)
