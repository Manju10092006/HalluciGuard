from __future__ import annotations
import uuid
import time
from typing import Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from schemas.models import VerifierInputV2, VerifierOutputV2, DomainStatistics
from api.pipeline import VerificationPipeline
from adapters.registry import get_registry
from cache import SqliteCache
from models import get_model_manager
from metrics import MetricsCollector
from version import get_version_info
from utils.logging import log_verification_request

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    pipeline = VerificationPipeline()
    app.state.pipeline = pipeline
    await pipeline.cache.init_db()
    yield
    # Shutdown
    pass

app = FastAPI(
    title="HalluciGuard Verifier Agent",
    description="Evidence retrieval and claim verification engine for the HalluciGuard trust layer.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

@app.post("/verify", response_model=VerifierOutputV2)
async def verify(payload: VerifierInputV2, request: Request) -> VerifierOutputV2:
    pipeline: VerificationPipeline = request.app.state.pipeline
    response = await pipeline.verify(payload)
    log_verification_request(response.request_id, payload.domain, len(payload.suspicious_claims))
    return response

@app.get("/health")
async def health(request: Request) -> Dict[str, Any]:
    pipeline: VerificationPipeline = request.app.state.pipeline
    registry = get_registry()
    model_manager = get_model_manager()
    
    adapters_health = []
    for domain, adapter in registry.get_all_adapters().items():
        adapters_health.append(adapter.get_health_status())
        
    return {
        "status": "healthy",
        "version": get_version_info(),
        "models": model_manager.status(),
        "cache": await pipeline.cache.stats(),
        "adapters": adapters_health,
        "uptime_seconds": time.time()
    }

@app.get("/domains", response_model=List[DomainStatistics])
async def domains() -> List[DomainStatistics]:
    registry = get_registry()
    stats = []
    for domain, adapter in registry.get_all_adapters().items():
        stats.append(DomainStatistics(
            domain=domain,
            sources=adapter.get_sources(),
            status=adapter.get_status(),
            credibility_scores=adapter.get_credibility_scores(),
            is_implemented=adapter.is_implemented(),
            adapter_health=adapter.get_health_status()
        ))
    return stats

@app.get("/pipeline")
async def get_pipeline_info() -> Dict[str, Any]:
    return {
        "stages": [
            {"name": "Domain Validation", "order": 1, "description": "Validates the domain of the claim."},
            {"name": "Claim Decomposition", "order": 2, "description": "Decomposes compound claims into simple claims."},
            {"name": "Query Expansion", "order": 3, "description": "Expands the sub-claim with domain terminology."},
            {"name": "Multi-source Retrieval", "order": 4, "description": "Fetches raw passages from registered adapters."},
            {"name": "Aggregation + Dedup", "order": 5, "description": "Combines and removes duplicate passages."},
            {"name": "Hybrid Retrieval", "order": 6, "description": "Applies dense and sparse retrieval to filter passages."},
            {"name": "Cross-encoder Reranking", "order": 7, "description": "Scores passages against the claim for relevance."},
            {"name": "NLI Entailment", "order": 8, "description": "Predicts logical entailment for top passages."},
            {"name": "Evidence Scoring", "order": 9, "description": "Applies credibility weighting and standardizes scores."},
            {"name": "Conflict Resolution", "order": 10, "description": "Resolves contradictory evidence to form a verdict."},
            {"name": "Explanation Generation", "order": 11, "description": "Produces a human-readable explanation."},
            {"name": "Citation Formatting", "order": 12, "description": "Formats citations metadata."},
        ],
        "last_execution": {}
    }

@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    return MetricsCollector().get_summary()
