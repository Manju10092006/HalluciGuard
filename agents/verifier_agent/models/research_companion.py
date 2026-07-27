from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from .domain_intelligence import get_domain_intelligence_registry
from .model_router import ModelRouter


def build_domain_research_report(domain: str) -> Dict[str, Any]:
    """Return notebook-ready documentation for one verifier domain."""
    registry = get_domain_intelligence_registry()
    router = ModelRouter()
    profile = registry.get_profile(domain)
    decision = router.route(profile.domain)
    return {
        "domain": profile.domain,
        "aliases": profile.aliases,
        "adapter": profile.adapter,
        "models": asdict(profile.models),
        "api_sources": [asdict(source) for source in profile.api_sources],
        "knowledge_bases": profile.knowledge_bases,
        "retrieval_strategy": profile.retrieval_strategy,
        "evidence_ranking_strategy": profile.evidence_ranking_strategy,
        "confidence_strategy": profile.confidence_strategy,
        "verification_workflow": profile.verification_workflow,
        "chunking_strategy": profile.chunking_strategy,
        "benchmarks": profile.benchmarks,
        "routing_decision": asdict(decision),
        "production_modules": [
            "models.domain_intelligence",
            "models.model_router",
            "models.model_manager",
            "retrievers.hybrid",
            "rerankers.cross_encoder",
            "nli.entailment",
            "api.pipeline",
        ],
    }


def build_model_research_report(domain: str, model_purpose: str) -> Dict[str, Any]:
    """Return notebook-ready documentation for a selected domain model purpose."""
    report = build_domain_research_report(domain)
    model_id = report["models"][model_purpose]
    return {
        **report,
        "model_purpose": model_purpose,
        "model_id": model_id,
        "input_format": {
            "embedding_model": "list[str] -> normalized dense vectors",
            "dense_model": "query/passages -> vector index scores",
            "sparse_model": "tokenized query/passages -> BM25 scores",
            "cross_encoder": "(claim, passage) pairs -> relevance logits",
            "reranker": "(claim, passage) pairs -> sorted evidence",
            "nli_model": "{text: evidence, text_pair: claim} -> entailment labels",
            "sentence_transformer": "list[str] -> sentence embeddings",
            "entity_recognition_model": "text -> entity spans",
            "classification_model": "text -> domain/task labels",
        }.get(model_purpose, "text inputs handled by the production model manager"),
        "output_format": {
            "embedding_model": "float32 embedding matrix",
            "dense_model": "ranked Passage objects with similarity scores",
            "sparse_model": "ranked Passage objects with lexical scores",
            "cross_encoder": "float relevance scores",
            "reranker": "ranked Passage objects",
            "nli_model": "entailment, contradiction, and neutral probabilities",
            "sentence_transformer": "float32 sentence embedding matrix",
            "entity_recognition_model": "entity labels, character spans, confidence scores",
            "classification_model": "class labels and confidence scores",
        }.get(model_purpose, "production-specific output"),
        "production_recommendation": (
            "Load lazily through ModelManager, keep allow_model_downloads disabled in CI, "
            "enable downloads or prewarm caches only in controlled benchmark/runtime environments."
        ),
    }


def required_notebook_sections() -> List[str]:
    return [
        "Problem solved",
        "Why HalluciGuard needs it",
        "Selection rationale",
        "Architecture",
        "Training data",
        "Parameters",
        "Input format",
        "Output format",
        "Inference examples",
        "Batch inference",
        "GPU benchmarking",
        "CPU benchmarking",
        "Memory usage",
        "Latency",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Failure cases",
        "Limitations",
        "Domain suitability",
        "Verifier Agent integration",
        "Competing model comparison",
        "Production recommendations",
    ]
