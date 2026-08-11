import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List
from schemas.models import AdapterHealthStatus

class AdapterHealthChecker:
    def __init__(self) -> None:
        self._cache: Dict[str, tuple[AdapterHealthStatus, float]] = {}
        self._cache_ttl = 300.0  # 5 minutes

    async def check_adapter(self, adapter: Any) -> AdapterHealthStatus:
        now = time.time()
        if adapter.name in self._cache:
            status, timestamp = self._cache[adapter.name]
            if now - timestamp < self._cache_ttl:
                return status

        start = time.perf_counter()
        is_healthy = False
        error_msg = ""
        try:
            # Assuming adapter has a health_check method or we run a dummy query
            if hasattr(adapter, 'health_check'):
                is_healthy = await adapter.health_check()
            else:
                is_healthy = True
        except Exception as e:
            error_msg = str(e)
            
        duration = int((time.perf_counter() - start) * 1000)
        
        status = AdapterHealthStatus(
            adapter_name=adapter.name,
            is_healthy=is_healthy,
            last_check=datetime.now(timezone.utc).isoformat(),
            response_time_ms=duration,
            error=error_msg
        )
        self._cache[adapter.name] = (status, now)
        return status

    async def check_all_adapters(self, registry: Any) -> List[AdapterHealthStatus]:
        tasks = [self.check_adapter(adapter) for adapter in registry.get_all_adapters()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_statuses = []
        for res in results:
            if isinstance(res, Exception):
                # Should not happen given check_adapter catches errors
                continue
            health_statuses.append(res)
            
        return health_statuses
