from __future__ import annotations

from adapters.registry import get_registry
from models.domain_intelligence import REQUIRED_DOMAINS, get_domain_intelligence_registry
from models.model_router import ModelRouter


def test_domain_intelligence_covers_required_domains() -> None:
    registry = get_domain_intelligence_registry()

    assert set(registry.list_domains()) == REQUIRED_DOMAINS
    for domain in REQUIRED_DOMAINS:
        profile = registry.get_profile(domain)
        assert profile.models.embedding_model
        assert profile.models.dense_model
        assert profile.models.sparse_model
        assert profile.models.cross_encoder
        assert profile.models.reranker
        assert profile.models.nli_model
        assert profile.models.sentence_transformer
        assert profile.models.entity_recognition_model
        assert profile.models.classification_model
        assert profile.api_sources
        assert profile.knowledge_bases
        assert profile.confidence_strategy
        assert profile.verification_workflow
        assert profile.notebooks
        assert profile.benchmarks


def test_domain_aliases_canonicalize_to_supported_domains() -> None:
    registry = get_domain_intelligence_registry()

    assert registry.canonicalize("medicine") == "medicine"
    assert registry.canonicalize("pharmaceuticals") == "pharmacy"
    assert registry.canonicalize("AI") == "artificial_intelligence"
    assert registry.canonicalize("public policy") == "government_public_policy"
    assert registry.canonicalize("unknown-domain") == "general"


def test_model_router_returns_domain_specific_decision() -> None:
    decision = ModelRouter().route("finance", "Tesla revenue increased")

    assert decision.domain == "finance"
    assert decision.adapter == "finance"
    assert decision.classification_model == "ProsusAI/finbert"
    assert decision.nli_model == "cross-encoder/nli-deberta-v3-base"
    assert decision.confidence_strategy


def test_adapter_registry_registers_all_domain_profiles() -> None:
    adapter_registry = get_registry()
    configured_domains = set(get_domain_intelligence_registry().list_domains())

    assert configured_domains.issubset(set(adapter_registry.list_domains()))
    assert adapter_registry.get_adapter("pharmaceuticals").name == "pharmacy"
    assert adapter_registry.get_adapter("legal").name == "law"
    assert adapter_registry.get_domain_statistics("space")["domain"] == "space_science"
