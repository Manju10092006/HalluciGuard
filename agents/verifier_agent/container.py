from typing import Any, Optional
from agents.verifier_agent.config.settings import get_settings, Settings
from agents.verifier_agent.models.model_manager import get_model_manager, ModelManager
from agents.verifier_agent.utils.http_client import get_client, ResilientHttpClient

class MockRegistry:
    def get_all_adapters(self) -> list:
        return []

class MockCache:
    pass

class MockMetrics:
    pass

class Container:
    def __init__(self, 
                 settings: Settings,
                 registry: Any,
                 model_manager: ModelManager,
                 cache: Any,
                 http_client: ResilientHttpClient,
                 metrics_collector: Any) -> None:
        self.settings = settings
        self.registry = registry
        self.model_manager = model_manager
        self.cache = cache
        self.http_client = http_client
        self.metrics_collector = metrics_collector

async def create_container() -> Container:
    settings = get_settings()
    http_client = get_client()
    model_manager = get_model_manager()
    
    # Mocking these for now, assuming they will be implemented fully later or elsewhere
    registry = MockRegistry()
    cache = MockCache()
    metrics_collector = MockMetrics()
    
    return Container(
        settings=settings,
        registry=registry,
        model_manager=model_manager,
        cache=cache,
        http_client=http_client,
        metrics_collector=metrics_collector
    )

_container_instance: Optional[Container] = None

async def get_container() -> Container:
    global _container_instance
    if _container_instance is None:
        _container_instance = await create_container()
    return _container_instance
