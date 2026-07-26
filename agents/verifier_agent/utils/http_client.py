from __future__ import annotations
import httpx
import yaml
import asyncio
from typing import Any, Optional, cast
from pathlib import Path

class ResilientHttpClient:
    def __init__(self, config_path: str = "config/retry.yaml") -> None:
        self.config = self._load_config(config_path)
        self.client = httpx.AsyncClient(http2=True, timeout=self.config["default"]["timeout_seconds"])
        
    def _load_config(self, config_path: str) -> dict[str, Any]:
        path = Path(__file__).parent.parent / config_path
        if not path.exists():
            return {
                "default": {
                    "max_retries": 3,
                    "base_delay_seconds": 1.0,
                    "max_delay_seconds": 16.0,
                    "exponential_base": 2,
                    "retry_on_status": [429, 500, 502, 503, 504],
                    "timeout_seconds": 10.0
                },
                "adapters": {}
            }
        with open(path, "r", encoding="utf-8") as f:
            return cast(dict[str, Any], yaml.safe_load(f))
            
    def _get_adapter_config(self, adapter_name: Optional[str]) -> dict[str, Any]:
        default = self.config["default"]
        if not adapter_name or adapter_name not in self.config.get("adapters", {}):
            return default
            
        adapter_config = self.config["adapters"][adapter_name]
        return {**default, **adapter_config}

    async def _execute_with_retry(self, method: str, url: str, adapter_name: Optional[str] = None, **kwargs: Any) -> httpx.Response:
        cfg = self._get_adapter_config(adapter_name)
        max_retries = cfg["max_retries"]
        base_delay = cfg["base_delay_seconds"]
        max_delay = cfg["max_delay_seconds"]
        retry_status = set(cfg["retry_on_status"])
        
        if "timeout" not in kwargs:
            kwargs["timeout"] = cfg["timeout_seconds"]

        last_exception: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                response = await self.client.request(method, url, **kwargs)
                if response.status_code not in retry_status:
                    response.raise_for_status()
                    return response
                    
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = base_delay * (cfg["exponential_base"] ** attempt)
                else:
                    delay = base_delay * (cfg["exponential_base"] ** attempt)
                    
            except httpx.RequestError as e:
                last_exception = e
                delay = base_delay * (cfg["exponential_base"] ** attempt)
            
            if attempt == max_retries:
                if last_exception:
                    raise last_exception
                response.raise_for_status()
                
            delay = min(delay, max_delay)
            await asyncio.sleep(delay)
            
        raise Exception("Unreachable")

    async def get(self, url: str, adapter_name: Optional[str] = None, **kwargs: Any) -> httpx.Response:
        return await self._execute_with_retry("GET", url, adapter_name, **kwargs)

    async def post(self, url: str, adapter_name: Optional[str] = None, **kwargs: Any) -> httpx.Response:
        return await self._execute_with_retry("POST", url, adapter_name, **kwargs)
        
    async def close(self) -> None:
        await self.client.aclose()

_shared_client: Optional[ResilientHttpClient] = None

def get_client() -> ResilientHttpClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = ResilientHttpClient()
    return _shared_client
