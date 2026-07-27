from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .domain_intelligence import DomainProfile, get_domain_intelligence_registry
from .model_manager import get_model_manager


LatencyBudget = Literal["interactive", "balanced", "thorough"]


@dataclass(frozen=True)
class ModelRoutingDecision:
    domain: str
    adapter: str
    embedding_model: str
    dense_model: str
    sparse_model: str
    retriever: str
    cross_encoder: str
    reranker_model: str
    nli_model: str
    sentence_transformer: str
    entity_recognition_model: str
    classification_model: str
    retrieval_strategy: str
    evidence_ranking_strategy: str
    confidence_strategy: str
    chunking_strategy: str
    device: str
    use_gpu: bool


class ModelRouter:
    """Chooses domain-specific models and strategies for a verification claim."""

    def __init__(self) -> None:
        self.registry = get_domain_intelligence_registry()
        self.model_manager = get_model_manager()

    def profile_for(self, domain: str) -> DomainProfile:
        return self.registry.get_profile(domain)

    def route(
        self,
        domain: str,
        claim_text: str = "",
        language: str = "en",
        latency_budget: LatencyBudget = "balanced",
        confidence_level: float = 0.0,
    ) -> ModelRoutingDecision:
        profile = self.profile_for(domain)
        models = profile.models
        device = self.model_manager.device
        use_gpu = device == "cuda"
        reranker = models.reranker
        cross_encoder = models.cross_encoder

        if latency_budget == "interactive" and not use_gpu:
            reranker = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            cross_encoder = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        if confidence_level < 0.4 and latency_budget != "interactive":
            reranker = models.reranker
            cross_encoder = models.cross_encoder

        return ModelRoutingDecision(
            domain=profile.domain,
            adapter=profile.adapter,
            embedding_model=models.embedding_model,
            dense_model=models.dense_model,
            sparse_model=models.sparse_model,
            retriever=models.retriever,
            cross_encoder=cross_encoder,
            reranker_model=reranker,
            nli_model=models.nli_model,
            sentence_transformer=models.sentence_transformer,
            entity_recognition_model=models.entity_recognition_model,
            classification_model=models.classification_model,
            retrieval_strategy=profile.retrieval_strategy,
            evidence_ranking_strategy=profile.evidence_ranking_strategy,
            confidence_strategy=profile.confidence_strategy,
            chunking_strategy=profile.chunking_strategy,
            device=device,
            use_gpu=use_gpu,
        )
