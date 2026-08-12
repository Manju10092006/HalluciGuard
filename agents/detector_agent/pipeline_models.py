"""Schemas for the Detector → Verifier integrated pipeline.

The existing /detect contract is intentionally untouched. These schemas are
used only by the new /analyze endpoint that performs conditional verification.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .models import DetectionResult


class AnalysisInput(BaseModel):
    """Full pipeline input: original query plus the LLM candidate response."""

    user_query: str = Field(..., min_length=1)
    llm_response: str = Field(..., min_length=1)
    domain: str = Field(default="general", min_length=1)
    query_id: Optional[str] = None


class AnalysisResult(BaseModel):
    """Combined Detector + optional Verifier result."""

    query_id: str
    detection: DetectionResult
    verifier_invoked: bool
    verifier_result: Optional[Dict[str, Any]] = None
    final_status: str
    message: str


__all__ = ["AnalysisInput", "AnalysisResult"]
