"""Memory agent shutdown must be fail-safe.

Memory is the terminal, audit-only stage. If one subsystem's save/close raises,
``close()`` must still attempt the others (no leaked DB connections) and must NOT
raise out of the caller's ``finally: await close()`` — otherwise a save glitch
would turn a Judge-accepted answer into a memory-node failure.
"""
import pytest

from agents.memory_agent.memory.memory_agent import MemoryAgent


class _Boom:
    """Subsystem whose save/close always raises."""

    def __init__(self):
        self.save_called = False

    def save(self):
        self.save_called = True
        raise RuntimeError("disk full")

    async def close(self):
        raise RuntimeError("connection reset")


class _RecordSync:
    def __init__(self):
        self.saved = False

    def save(self):
        self.saved = True


class _RecordAsync:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_close_is_failsafe_and_closes_all_subsystems():
    kg = _Boom()            # save() raises
    vectors = _RecordSync()  # must still be saved despite kg failure
    cache = _Boom()          # close() raises
    patterns = _RecordAsync()  # must still be closed
    trust = _RecordAsync()     # must still be closed

    agent = MemoryAgent.__new__(MemoryAgent)
    agent.kg = kg
    agent.vectors = vectors
    agent.cache = cache
    agent.patterns = patterns
    agent.trust = trust

    # Must not raise even though kg.save and cache.close both blow up.
    await agent.close()

    assert kg.save_called is True
    assert vectors.saved is True, "vector store save skipped after kg failure"
    assert patterns.closed is True, "patterns close skipped after cache failure"
    assert trust.closed is True, "trust close skipped after cache failure"
