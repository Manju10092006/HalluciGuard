import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .detector import Detector
from .schemas import DetectRequest, DetectResponse


app = FastAPI(title="HalluciGuard Detector", version="0.1.0")


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    repo_root = Path(__file__).resolve().parent.parent
    model_path = os.getenv("HALLUCIGUARD_DETECTOR_MODEL", "").strip()
    return Detector(Path(model_path) if model_path else repo_root / "artifacts" / "detector-best")


@app.get("/health")
def health():
    return {"status": "ok", "model": get_detector().version}


@app.post("/v1/detect", response_model=DetectResponse)
def detect(request: DetectRequest):
    try:
        return get_detector().detect(
            draft_answer=request.draft_answer,
            evidence=request.evidence,
            user_query=request.user_query,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
