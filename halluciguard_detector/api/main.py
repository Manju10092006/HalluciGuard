"""
api / main.py
─────────────
FastAPI service exposing POST /detect and GET /health.
Runs standalone on its own port (default: 8080).
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from halluciguard_detector.detector import StandaloneDetector
from halluciguard_detector.schemas import DetectorRequest, DetectorResponse

app = FastAPI(
    title="HalluciGuard Detector V2 Service",
    description="Standalone hallucination risk detector microservice",
    version="1.0.0",
)

# Singleton detector instance
detector = StandaloneDetector()


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "halluciguard_detector",
        "version": "1.0.0",
        "model": detector.model_name,
    }


@app.post("/detect", response_model=DetectorResponse)
def detect_hallucination(req: DetectorRequest):
    """
    Main detection endpoint.
    Accepts request_id, user_query, draft_answer, optional claims.
    """
    if not req.draft_answer:
        raise HTTPException(status_code=400, detail="draft_answer cannot be empty")

    try:
        response = detector.detect(
            user_query=req.user_query,
            draft_answer=req.draft_answer,
            request_id=req.request_id,
            supplied_claims=req.claims,
        )
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(exc)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DETECTOR_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
