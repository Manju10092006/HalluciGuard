import asyncio
import time
from typing import Any, Coroutine, Optional, TypeVar
from dataclasses import dataclass

T = TypeVar("T")

@dataclass
class Result:
    value: Any
    error: Optional[Exception]
    duration_ms: int
    source_name: str = ""

async def execute_with_timeout(coro: Coroutine[Any, Any, T], timeout: float, source_name: str = "") -> Result:
    start = time.perf_counter()
    try:
        value = await asyncio.wait_for(coro, timeout=timeout)
        duration = int((time.perf_counter() - start) * 1000)
        return Result(value=value, error=None, duration_ms=duration, source_name=source_name)
    except Exception as e:
        duration = int((time.perf_counter() - start) * 1000)
        return Result(value=None, error=e, duration_ms=duration, source_name=source_name)

async def parallel_execute(tasks: list[Coroutine[Any, Any, Any]], timeout: float = 10.0) -> list[Result]:
    wrapped_tasks = [
        execute_with_timeout(task, timeout=timeout, source_name=getattr(task, "__name__", str(i))) 
        for i, task in enumerate(tasks)
    ]
    return await asyncio.gather(*wrapped_tasks, return_exceptions=True)

async def execute_with_fallback(primary: Coroutine[Any, Any, T], fallback: Coroutine[Any, Any, T], timeout: float = 10.0) -> Result:
    res = await execute_with_timeout(primary, timeout=timeout, source_name="primary")
    if res.error is None:
        return res
    return await execute_with_timeout(fallback, timeout=timeout, source_name="fallback")
