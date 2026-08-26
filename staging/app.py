"""
HalluciGuard Detector Agent — Staging FastAPI Application.

Minimal, production-ready staging entry point for hallucination detection inference.
Loads the fine-tuned HaluEval model directly from Hugging Face Hub.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Support both staging package execution and direct script run
try:
    from .detector.detector import DetectorAgent
    from .detector.models import DetectionInput, DetectionResult
    from .detector.config import DetectorConfig
except ImportError:
    from detector.detector import DetectorAgent
    from detector.models import DetectionInput, DetectionResult
    from detector.config import DetectorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("halluciguard.staging")

# Initialize global detector agent
config = DetectorConfig()
detector_agent = DetectorAgent(config=config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup model warming and shutdown."""
    print("[STARTUP] HalluciGuard Detector starting")
    logger.info("[STARTUP] HalluciGuard Detector starting")
    try:
        detector_agent._ensure_model_loaded()
        if detector_agent._inference and detector_agent._inference.is_loaded():
            logger.info("[STARTUP] Model pre-loaded and ready for requests.")
        else:
            logger.warning("[STARTUP] Model failed to load; running in degraded mode.")
    except Exception as exc:
        logger.error(f"[STARTUP] Error during startup model loading: {exc}")
    yield
    print("[SHUTDOWN] HalluciGuard Detector stopping")
    logger.info("[SHUTDOWN] HalluciGuard Detector stopping")


app = FastAPI(
    title="HalluciGuard Detector Agent (Staging)",
    description=(
        "Production-ready staging API for the HalluciGuard Detector Agent. "
        "Evaluates LLM output hallucination risk using the fine-tuned HaluEval classifier."
    ),
    version="1.0.0-staging",
    lifespan=lifespan,
)

# Configure CORS origins
frontend_origin_env = os.environ.get("FRONTEND_ORIGIN", "")
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]
if frontend_origin_env:
    for orig in frontend_origin_env.split(","):
        clean_orig = orig.strip().rstrip("/")
        if clean_orig and clean_orig not in allowed_origins:
            allowed_origins.append(clean_orig)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if frontend_origin_env else ["*"],
    allow_origin_regex=r"https://.*\.vercel\.app" if not frontend_origin_env else None,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Root Service Status",
    tags=["Diagnostics"],
)
def root() -> Dict[str, Any]:
    """Root endpoint providing service metadata and API directory."""
    inference = detector_agent._inference
    model_loaded = inference.is_loaded() if inference else False
    return {
        "service": "HalluciGuard Detector Agent",
        "status": "online",
        "model_loaded": model_loaded,
        "model_reference": config.halueval_model_path,
        "docs_url": "/docs",
        "endpoints": {
            "detect": "POST /detect",
            "health": "GET /health",
            "model_info": "GET /model-info",
        },
    }


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    tags=["Diagnostics"],
)
def health_check() -> Dict[str, Any]:
    """Health check endpoint indicating model readiness and device info."""
    inference = detector_agent._inference
    model_loaded = inference.is_loaded() if inference else False
    model_source = (
        inference.model_path if (inference and inference.model_path) else config.halueval_model_path
    )
    device = inference.device if inference else "unknown"

    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_source": model_source,
        "device": device,
    }


@app.get(
    "/model-info",
    status_code=status.HTTP_200_OK,
    summary="Model Metadata",
    tags=["Diagnostics"],
)
def model_info() -> Dict[str, Any]:
    """Provides non-sensitive metadata regarding the active detector model."""
    inference = detector_agent._inference
    model_loaded = inference.is_loaded() if inference else False
    model_source = (
        inference.model_path if (inference and inference.model_path) else config.halueval_model_path
    )
    device = inference.device if inference else "unknown"

    return {
        "model_repository": model_source,
        "model_loaded": model_loaded,
        "device": device,
        "max_sequence_length": config.halueval_max_length,
        "label_mapping": {
            "0": "NO_HALLUCINATION",
            "1": "HALLUCINATION",
        },
        "risk_thresholds": {
            "low_risk_threshold": config.low_risk_threshold,
            "high_risk_threshold": config.high_risk_threshold,
        },
        "routing_policy": {
            "LOW": "ACCEPT",
            "MEDIUM": "ACCEPT",
            "HIGH": "VERIFY",
        },
        "version": "1.0.0-staging",
    }


@app.post(
    "/detect",
    response_model=DetectionResult,
    status_code=status.HTTP_200_OK,
    summary="Detect Hallucinations",
    tags=["Inference"],
)
def detect(payload: DetectionInput) -> DetectionResult:
    """Run hallucination detection inference on a query/response pair."""
    try:
        result = detector_agent.detect(
            user_query=payload.user_query,
            llm_response=payload.llm_response,
            context=payload.context,
        )
        return result
    except Exception as exc:
        logger.error(f"[API] Detection endpoint error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("staging.app:app", host="0.0.0.0", port=8000, reload=True)
