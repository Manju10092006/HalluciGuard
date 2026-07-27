"""ML model management for the Verifier Agent."""
from .model_manager import ModelManager, get_model_manager
from .domain_intelligence import (
    ApiSourceSpec,
    DomainIntelligenceRegistry,
    DomainModelSpec,
    DomainProfile,
    get_domain_intelligence_registry,
)
from .model_router import ModelRouter, ModelRoutingDecision
from .research_companion import (
    build_domain_research_report,
    build_model_research_report,
    required_notebook_sections,
)
