from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..config.settings import get_settings
from ..memory.memory_agent import MemoryAgent, get_memory_agent
from ..monitoring import (
    cache_hit_total,
    cache_miss_total,
    contradiction_total,
    recall_duration,
    recall_total,
    request_duration,
    store_duration,
    store_total,
)
from ..schemas.models import (
    BatchStoreResponse,
    DeleteFactResponse,
    GraphAnalytics,
    HealthResponse,
    MemoryStatsResponse,
    PatternQueryRequest,
    PatternType,
    RecallRequest,
    RecallResponse,
    StoreFactRequest,
    StoreFactResponse,
    TrustChangeReason,
    TrustUpdate,
    TrustUpdateRequest,
    UpdateFactRequest,
    UpdateFactResponse,
)
from ..version import MEMORY_AGENT_VERSION

logger = logging.getLogger(__name__)

settings = get_settings()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _require_api_key(api_key: str | None = Depends(_api_key_header)) -> None:
    expected = settings.memory_agent_api_key
    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent = get_memory_agent()
    await agent.initialize()
    app.state.memory_agent = agent
    logger.info("Memory Agent started on port %d", settings.memory_agent_port)
    yield
    await agent.close()
    logger.info("Memory Agent stopped")


app = FastAPI(
    title="HalluciGuard Memory Agent",
    description="Knowledge persistence layer for verified facts, patterns, and source trust.",
    version=MEMORY_AGENT_VERSION,
    lifespan=lifespan,
    dependencies=[Depends(_require_api_key)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _observe_request(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    request_duration.labels(request.url.path).observe(
        time.perf_counter() - start
    )
    return response


def _get_agent(request: Request) -> MemoryAgent:
    return request.app.state.memory_agent


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.post("/store", response_model=StoreFactResponse)
async def store_fact(request: Request, body: StoreFactRequest):
    agent = _get_agent(request)
    start = time.perf_counter()
    try:
        result = await agent.store_fact(body)
        store_duration.observe(time.perf_counter() - start)
        store_total.labels(
            verdict=body.verdict,
            duplicate="true" if result.duplicate_of else "false",
        ).inc()
        for _ in result.contradictions:
            contradiction_total.labels(body.domain).inc()
        return result
    except Exception as e:
        logger.exception("Failed to store fact")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/store/batch", response_model=BatchStoreResponse)
async def store_batch(request: Request, body: list[StoreFactRequest]):
    agent = _get_agent(request)
    try:
        return await agent.store_facts_batch(body)
    except Exception as e:
        logger.exception("Failed to store fact batch")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/facts/{fact_id}", response_model=DeleteFactResponse)
async def delete_fact(request: Request, fact_id: str):
    agent = _get_agent(request)
    try:
        return await agent.delete_fact(fact_id)
    except Exception as e:
        logger.exception("Failed to delete fact")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/facts/{fact_id}", response_model=UpdateFactResponse)
async def update_fact(request: Request, fact_id: str, body: UpdateFactRequest):
    agent = _get_agent(request)
    try:
        body.fact_id = fact_id
        return await agent.update_fact(body)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to update fact")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recall", response_model=RecallResponse)
async def recall(request: Request, body: RecallRequest):
    agent = _get_agent(request)
    start = time.perf_counter()
    try:
        result = await agent.recall(body)
        recall_duration.observe(time.perf_counter() - start)
        recall_total.labels(
            cached="true" if result.cached_verification else "false",
            reranked="true" if result.reranked else "false",
        ).inc()
        if body.include_cache:
            label = body.domain or "general"
            if result.cached_verification:
                cache_hit_total.labels(label).inc()
            else:
                cache_miss_total.labels(label).inc()
        return result
    except Exception as e:
        logger.exception("Failed to recall")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/check")
async def check_cache(request: Request, domain: str, claim_text: str):
    agent = _get_agent(request)
    result = await agent.check_cache(domain, claim_text)
    if result is None:
        return {"cached": False}
    return {"cached": True, "verification": result}


@app.delete("/cache/invalidate")
async def invalidate_cache(request: Request, domain: str, claim_text: str):
    agent = _get_agent(request)
    removed = await agent.invalidate_cache(domain, claim_text)
    return {"removed": removed}


@app.post("/trust/update", response_model=TrustUpdate)
async def update_trust(request: Request, body: TrustUpdateRequest):
    agent = _get_agent(request)
    try:
        return await agent.update_source_trust(
            source_id=body.source_id,
            source_name=body.source_name,
            domain=body.domain,
            reason=body.reason,
            evidence_count=body.evidence_count,
        )
    except Exception as e:
        logger.exception("Failed to update trust")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trust/{source_id}")
async def get_trust(request: Request, source_id: str):
    agent = _get_agent(request)
    record = await agent.get_source_trust(source_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return record


@app.get("/trust/domain/{domain}")
async def get_domain_sources(request: Request, domain: str):
    agent = _get_agent(request)
    sources = await agent.trust.get_domain_sources(domain)
    return sources


@app.post("/patterns/query")
async def query_patterns(request: Request, body: PatternQueryRequest):
    agent = _get_agent(request)
    patterns = await agent.query_patterns(
        domain=body.domain,
        pattern_type=body.pattern_type,
        min_confidence=body.min_confidence,
        top_k=body.top_k,
    )
    return patterns


@app.get("/patterns/domain/{domain}")
async def get_domain_patterns(request: Request, domain: str):
    agent = _get_agent(request)
    summary = await agent.patterns.get_domain_summary(domain)
    return summary


@app.get("/knowledge-graph/stats")
async def kg_stats(request: Request):
    agent = _get_agent(request)
    return agent.kg.get_stats()


@app.get("/knowledge-graph/analytics", response_model=GraphAnalytics)
async def kg_analytics(request: Request, top_k: int = 10):
    agent = _get_agent(request)
    return agent.kg.analyze(top_k=top_k)


@app.get("/knowledge-graph/entity/{entity_id}")
async def get_entity(request: Request, entity_id: str):
    agent = _get_agent(request)
    entity = agent.kg.get_entity(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    return entity


@app.get("/knowledge-graph/entity/{entity_id}/neighbors")
async def get_entity_neighbors(request: Request, entity_id: str):
    agent = _get_agent(request)
    neighbors = agent.kg.get_neighbors(entity_id)
    return [{"entity": n.model_dump(), "edge": e.model_dump()} for n, e in neighbors]


@app.get("/vectors/search")
async def vector_search(request: Request, q: str, domain: str = None, top_k: int = 10):
    agent = _get_agent(request)
    results = agent.vectors.search(
        query=q,
        top_k=top_k,
        metadata_filter={"domain": domain} if domain else None,
    )
    return results


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    agent = _get_agent(request)
    components = {
        "knowledge_graph": "ok",
        "verification_cache": "ok",
        "pattern_learner": "ok",
        "source_trust": "ok",
        "vector_store": "ok",
    }
    try:
        stats = await agent.get_stats()
    except Exception:
        stats = None

    return HealthResponse(
        status="healthy",
        version=MEMORY_AGENT_VERSION,
        components=components,
        stats=stats,
    )


@app.get("/stats", response_model=MemoryStatsResponse)
async def stats(request: Request):
    agent = _get_agent(request)
    return await agent.get_stats()


@app.get("/domains")
async def list_domains():
    return {"domains": settings.supported_domains}


@app.post("/save")
async def save(request: Request):
    agent = _get_agent(request)
    await agent.save_all()
    return {"status": "saved"}


@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
