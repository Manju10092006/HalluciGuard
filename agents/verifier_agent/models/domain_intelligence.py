from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "domain_intelligence.yaml"
REQUIRED_DOMAINS = {
    "healthcare", "medicine", "pharmacy", "biology", "genetics", "chemistry",
    "physics", "mathematics", "astronomy", "space_science",
    "climate_environment", "agriculture", "food_nutrition",
    "artificial_intelligence", "machine_learning", "computer_science",
    "cybersecurity", "programming", "data_science", "finance", "economics",
    "business", "law", "government_public_policy", "history", "geography",
    "education", "psychology", "sociology", "philosophy",
}


@dataclass(frozen=True)
class ApiSourceSpec:
    id: str
    name: str
    base_url: str
    requires_key: bool
    credibility: float


@dataclass(frozen=True)
class DomainModelSpec:
    embedding_model: str
    retriever: str
    dense_model: str
    sparse_model: str
    cross_encoder: str
    reranker: str
    nli_model: str
    sentence_transformer: str
    entity_recognition_model: str
    classification_model: str

    def unique_huggingface_ids(self) -> List[str]:
        model_ids = [
            self.embedding_model,
            self.dense_model,
            self.cross_encoder,
            self.reranker,
            self.nli_model,
            self.sentence_transformer,
            self.entity_recognition_model,
            self.classification_model,
        ]
        return sorted({m for m in model_ids if "/" in m})


@dataclass(frozen=True)
class DomainProfile:
    domain: str
    aliases: List[str]
    adapter: str
    models: DomainModelSpec
    api_sources: List[ApiSourceSpec]
    knowledge_bases: List[str]
    retrieval_strategy: str
    evidence_ranking_strategy: str
    confidence_strategy: str
    verification_workflow: List[str]
    chunking_strategy: str
    notebooks: List[str] = field(default_factory=list)
    benchmarks: List[str] = field(default_factory=list)

    @property
    def source_ids(self) -> List[str]:
        return [source.id for source in self.api_sources]


class DomainIntelligenceRegistry:
    """Loads and validates domain-specific model/API/routing profiles."""

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.config_path = config_path
        raw = self._load_yaml(config_path)
        self.version = str(raw.get("version", "unknown"))
        self.default_domain = str(raw.get("default_domain", "general"))
        self._shared_models = dict(raw.get("shared_models", {}))
        self._defaults = dict(raw.get("defaults", {}))
        self._profiles = self._build_profiles(raw.get("domains", {}))
        self._aliases = self._build_aliases(self._profiles)
        self._validate_required_domains()

    def _load_yaml(self, path: Path) -> Mapping[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, Mapping):
            raise ValueError(f"Domain intelligence config at {path} must be a mapping")
        return data

    def _build_profiles(self, raw_domains: Mapping[str, Any]) -> Dict[str, DomainProfile]:
        profiles: Dict[str, DomainProfile] = {}
        for domain, raw_profile in raw_domains.items():
            if not isinstance(raw_profile, Mapping):
                raise ValueError(f"Domain profile {domain} must be a mapping")
            model_values = {
                "embedding_model": raw_profile.get("embedding_model", self._shared_models["embedding_model"]),
                "retriever": raw_profile.get("retriever", self._defaults["retriever"]),
                "dense_model": raw_profile.get("dense_model", self._shared_models["dense_model"]),
                "sparse_model": raw_profile.get("sparse_model", self._shared_models["sparse_model"]),
                "cross_encoder": raw_profile.get("cross_encoder", self._shared_models["cross_encoder"]),
                "reranker": raw_profile.get("reranker", self._shared_models["reranker"]),
                "nli_model": raw_profile.get("nli_model", self._shared_models["nli_model"]),
                "sentence_transformer": raw_profile.get("sentence_transformer", self._shared_models["sentence_transformer"]),
                "entity_recognition_model": raw_profile.get("entity_recognition_model", self._shared_models["entity_recognition_model"]),
                "classification_model": raw_profile.get("classification_model", self._shared_models["classification_model"]),
            }
            sources = [
                ApiSourceSpec(
                    id=str(item["id"]),
                    name=str(item["name"]),
                    base_url=str(item["base_url"]),
                    requires_key=bool(item.get("requires_key", False)),
                    credibility=float(item.get("credibility", 0.8)),
                )
                for item in raw_profile.get("api_sources", [])
            ]
            if not sources:
                raise ValueError(f"Domain profile {domain} must define at least one API source")
            profiles[str(domain)] = DomainProfile(
                domain=str(domain),
                aliases=[str(alias) for alias in raw_profile.get("aliases", [])],
                adapter=str(raw_profile.get("adapter", domain)),
                models=DomainModelSpec(**model_values),
                api_sources=sources,
                knowledge_bases=[str(kb) for kb in raw_profile.get("knowledge_bases", [])],
                retrieval_strategy=str(raw_profile.get("retrieval_strategy", self._defaults["retrieval_strategy"])),
                evidence_ranking_strategy=str(raw_profile.get("evidence_ranking_strategy", self._defaults["evidence_ranking_strategy"])),
                confidence_strategy=str(raw_profile.get("confidence_strategy", self._defaults["confidence_strategy"])),
                verification_workflow=[str(step) for step in raw_profile.get("verification_workflow", self._defaults["verification_workflow"])],
                chunking_strategy=str(raw_profile.get("chunking_strategy", self._defaults["chunking_strategy"])),
                notebooks=[str(path) for path in raw_profile.get("notebooks", [])],
                benchmarks=[str(name) for name in raw_profile.get("benchmarks", [])],
            )
        return profiles

    def _build_aliases(self, profiles: Mapping[str, DomainProfile]) -> Dict[str, str]:
        aliases: Dict[str, str] = {}
        for domain, profile in profiles.items():
            aliases[self._normalize(domain)] = domain
            aliases[self._normalize(domain.replace("_", " "))] = domain
            for alias in profile.aliases:
                aliases[self._normalize(alias)] = domain
        return aliases

    def _validate_required_domains(self) -> None:
        missing = REQUIRED_DOMAINS.difference(self._profiles)
        if missing:
            raise ValueError(f"Domain intelligence config missing required domains: {sorted(missing)}")

    def _normalize(self, value: str) -> str:
        return value.strip().lower().replace("-", "_").replace(" ", "_")

    def canonicalize(self, domain: str) -> str:
        key = self._normalize(domain or self.default_domain)
        return self._aliases.get(key, self.default_domain)

    def get_profile(self, domain: str) -> DomainProfile:
        return self._profiles[self.canonicalize(domain)]

    def list_domains(self) -> List[str]:
        return sorted(REQUIRED_DOMAINS)

    def all_profiles(self) -> List[DomainProfile]:
        return [self._profiles[domain] for domain in self.list_domains()]

    def unique_model_ids(self, domains: Optional[Iterable[str]] = None) -> List[str]:
        selected = [self.get_profile(domain) for domain in domains] if domains else self.all_profiles()
        model_ids = set()
        for profile in selected:
            model_ids.update(profile.models.unique_huggingface_ids())
        return sorted(model_ids)


@lru_cache()
def get_domain_intelligence_registry() -> DomainIntelligenceRegistry:
    return DomainIntelligenceRegistry()
