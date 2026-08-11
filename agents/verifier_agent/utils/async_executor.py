import asyncio
import time
from typing import Any, Awaitable, Coroutine, Optional, TypeVar
from dataclasses import dataclass

import anyio
import logging

logger = logging.getLogger(__name__)

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
        logger.warning("Task '%s' failed or timed out: %s", source_name, e)
        return Result(value=None, error=e, duration_ms=duration, source_name=source_name)

async def parallel_execute(tasks: list[Coroutine[Any, Any, Any]], timeout: float = 10.0) -> list[Result]:
    wrapped_tasks = [
        execute_with_timeout(task, timeout=timeout, source_name=getattr(task, "__name__", str(i))) 
        for i, task in enumerate(tasks)
    ]
    return await asyncio.gather(*wrapped_tasks, return_exceptions=True)

async def gather_results(tasks: list[Awaitable[Any]]) -> list[Any]:
    """Run awaitables concurrently without depending on the asyncio event loop."""
    results: list[Any] = [None] * len(tasks)

    async def run_one(index: int, task: Awaitable[Any]) -> None:
        try:
            results[index] = await task
        except Exception as e:
            results[index] = e

    async with anyio.create_task_group() as task_group:
        for index, task in enumerate(tasks):
            task_group.start_soon(run_one, index, task)

    return results

async def execute_with_fallback(primary: Coroutine[Any, Any, T], fallback: Coroutine[Any, Any, T], timeout: float = 10.0) -> Result:
    res = await execute_with_timeout(primary, timeout=timeout, source_name="primary")
    if res.error is None:
        return res
    logger.warning("Primary task failed, executing fallback task")
    return await execute_with_timeout(fallback, timeout=timeout, source_name="fallback")
