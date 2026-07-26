from __future__ import annotations
import time
import logging
from typing import Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas.models import VerifierInputV2, VerifierOutputV2, DomainStatistics
from api.pipeline import VerificationPipeline
from adapters.registry import get_registry
from models import get_model_manager
from metrics import MetricsCollector
from version import get_version_info
from utils.logging import setup_logger, VerificationLogRecord, log_verification_request

logger = setup_logger("api.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing Verifier Agent lifespan...")
    pipeline = VerificationPipeline()
    app.state.pipeline = pipeline
    try:
        await pipeline.cache.init_db()
    except Exception as e:
        logger.warning(f"Failed to initialize SQLite cache: {e}")
    yield
    # Shutdown
    logger.info("Shutting down Verifier Agent...")

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

def _get_pipeline(request: Request) -> VerificationPipeline:
    if not hasattr(request.app.state, 'pipeline') or request.app.state.pipeline is None:
        pipeline = VerificationPipeline()
        request.app.state.pipeline = pipeline
    return request.app.state.pipeline

@app.post("/verify", response_model=VerifierOutputV2)
async def verify(payload: VerifierInputV2, request: Request) -> VerifierOutputV2:
    pipeline = _get_pipeline(request)
    try:
        response = await pipeline.verify(payload)
        rec = VerificationLogRecord(
            query_id=payload.query_id,
            request_id=payload.query_id,
            domain=payload.domain,
            stages_timing={},
            adapters_called=[payload.domain],
            adapters_failed=[],
            verdict=response.claim_evidence[0].verdict.value if response.claim_evidence else "unknown",
            confidence=response.overall_evidence_confidence,
            total_latency_ms=response.latency_ms,
            timestamp=time.time()
        )
        log_verification_request(logger, rec)
        return response
    except Exception as e:
        logger.error(f"Error during verification endpoint call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health(request: Request) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    registry = get_registry()
    model_manager = get_model_manager()
    
    try:
        cache_stats = await pipeline.cache.stats()
    except Exception:
        cache_stats = {"status": "unavailable"}

    return {
        "status": "healthy",
        "version": get_version_info(),
        "models": model_manager.status(),
        "cache": cache_stats,
        "adapters_registered": registry.list_domains(),
        "uptime_seconds": round(time.time(), 2)
    }

@app.get("/domains", response_model=List[DomainStatistics])
async def domains() -> List[DomainStatistics]:
    registry = get_registry()
    all_stats = registry.get_all_statistics()
    result: List[DomainStatistics] = []
    for stat in all_stats:
        result.append(DomainStatistics(
            domain=stat.get("domain", "unknown"),
            sources=stat.get("sources", []),
            status=stat.get("status", "active"),
            credibility_scores=stat.get("credibility_scores", {}),
            is_implemented=stat.get("is_implemented", True),
            adapter_health=[]
        ))
    return result

@app.get("/pipeline")
async def get_pipeline_info() -> Dict[str, Any]:
    return {
        "stages": [
            {"name": "Domain Validation", "order": 1, "description": "Validates domain classification using zero-shot inference."},
            {"name": "Claim Decomposition", "order": 2, "description": "Decomposes compound claims into atomic sub-claims."},
            {"name": "Query Expansion", "order": 3, "description": "Expands sub-claims with domain-specific terminology."},
            {"name": "Multi-source Retrieval", "order": 4, "description": "Fetches evidence from domain-specific APIs in parallel."},
            {"name": "Aggregation + Dedup", "order": 5, "description": "Combines sources & removes >85% Jaccard token overlap duplicates."},
            {"name": "Hybrid RRF Retrieval", "order": 6, "description": "Reciprocal Rank Fusion of BM25 + FAISS Dense vector search."},
            {"name": "Cross-encoder Reranking", "order": 7, "description": "Scores passages against claims using ms-marco-MiniLM."},
            {"name": "DeBERTa NLI Entailment", "order": 8, "description": "Predicts logical entailment (Entailment/Contradiction/Neutral)."},
            {"name": "Evidence Scoring", "order": 9, "description": "Applies credibility weighting and recency decay factors."},
            {"name": "Conflict Resolution", "order": 10, "description": "Resolves contradictory evidence with 2:1 majority logic."},
            {"name": "Explanation Generation", "order": 11, "description": "Produces human-readable natural language explanations."},
            {"name": "Citation Formatting", "order": 12, "description": "Formats metadata citations adhering to VerifierOutputV2 contract."}
        ],
        "last_execution": {}
    }

@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    return MetricsCollector().get_summary()
