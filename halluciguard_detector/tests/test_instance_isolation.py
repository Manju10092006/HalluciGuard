from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from unittest.mock import Mock, call, sentinel

import pytest

import halluciguard_detector.agent as agent_module
from halluciguard_detector.agent import DetectorAgent


def test_agents_with_different_model_paths_load_their_own_detector(monkeypatch, tmp_path):
    constructor = Mock(side_effect=[sentinel.first_model, sentinel.second_model])
    monkeypatch.setattr(agent_module, "Detector", constructor)
    first = DetectorAgent(model_path=tmp_path / "first-model", device="cpu")
    second = DetectorAgent(model_path=tmp_path / "second-model", device="cpu")

    assert first._get_detector() is sentinel.first_model
    assert second._get_detector() is sentinel.second_model
    assert first._get_detector() is sentinel.first_model
    assert second._get_detector() is sentinel.second_model
    assert constructor.call_args_list == [
        call(tmp_path / "first-model", "cpu"),
        call(tmp_path / "second-model", "cpu"),
    ]


def test_agents_with_different_devices_load_their_own_detector(monkeypatch, tmp_path):
    constructor = Mock(side_effect=[sentinel.cpu_model, sentinel.cuda_model])
    monkeypatch.setattr(agent_module, "Detector", constructor)
    model_path = tmp_path / "same-model"
    cpu_agent = DetectorAgent(model_path=model_path, device="cpu")
    cuda_agent = DetectorAgent(model_path=model_path, device="cuda:0")

    assert cpu_agent._get_detector() is sentinel.cpu_model
    assert cuda_agent._get_detector() is sentinel.cuda_model
    assert cpu_agent._get_detector() is sentinel.cpu_model
    assert cuda_agent._get_detector() is sentinel.cuda_model
    assert constructor.call_args_list == [
        call(model_path, "cpu"),
        call(model_path, "cuda:0"),
    ]


def test_detector_is_initialized_lazily_and_reused(monkeypatch, tmp_path):
    constructor = Mock(return_value=sentinel.lazy_model)
    monkeypatch.setattr(agent_module, "Detector", constructor)
    model_path = tmp_path / "lazy-model"
    agent = DetectorAgent(model_path=model_path, device="cpu")

    constructor.assert_not_called()
    assert agent._get_detector() is sentinel.lazy_model
    assert agent._get_detector() is sentinel.lazy_model
    assert agent._get_detector() is sentinel.lazy_model
    constructor.assert_called_once_with(model_path, "cpu")


def test_failed_initialization_retries_without_another_agents_detector(monkeypatch, tmp_path):
    constructor = Mock(
        side_effect=[
            sentinel.other_model,
            RuntimeError("checkpoint could not be loaded"),
            sentinel.retried_model,
        ]
    )
    monkeypatch.setattr(agent_module, "Detector", constructor)
    other = DetectorAgent(model_path=tmp_path / "other-model", device="cpu")
    retrying = DetectorAgent(model_path=tmp_path / "retry-model", device="cuda:0")

    assert other._get_detector() is sentinel.other_model
    with pytest.raises(RuntimeError, match="checkpoint could not be loaded"):
        retrying._get_detector()

    assert retrying._detector is None
    assert other._get_detector() is sentinel.other_model
    assert retrying._get_detector() is sentinel.retried_model
    assert retrying._get_detector() is sentinel.retried_model
    assert constructor.call_args_list == [
        call(tmp_path / "other-model", "cpu"),
        call(tmp_path / "retry-model", "cuda:0"),
        call(tmp_path / "retry-model", "cuda:0"),
    ]


def test_concurrent_calls_initialize_one_detector(monkeypatch, tmp_path):
    worker_count = 6
    constructor_entered = Event()
    release_constructor = Event()
    all_lock_attempts = Event()

    def construct(model_path, device):
        constructor_entered.set()
        if not release_constructor.wait(timeout=10):
            raise AssertionError("test did not release the detector constructor")
        return sentinel.concurrent_model

    constructor = Mock(side_effect=construct)
    monkeypatch.setattr(agent_module, "Detector", constructor)
    model_path = tmp_path / "concurrent-model"
    agent = DetectorAgent(model_path=model_path, device="cpu")
    original_lock = agent._lock

    class ObservedLock:
        """Signal contention while retaining the agent's original lock."""

        def __init__(self):
            self.attempts = 0
            self.observer_lock = Lock()

        def __enter__(self):
            with self.observer_lock:
                self.attempts += 1
                if self.attempts == worker_count:
                    all_lock_attempts.set()
            return original_lock.__enter__()

        def __exit__(self, *exception_info):
            return original_lock.__exit__(*exception_info)

    monkeypatch.setattr(agent, "_lock", ObservedLock())
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(agent._get_detector) for _ in range(worker_count)]
        try:
            assert constructor_entered.wait(timeout=10), "constructor was not entered"
            assert all_lock_attempts.wait(timeout=10), "workers did not contend on the lock"
        finally:
            release_constructor.set()
        results = [future.result(timeout=10) for future in futures]

    assert all(result is sentinel.concurrent_model for result in results)
    assert agent._get_detector() is sentinel.concurrent_model
    constructor.assert_called_once_with(model_path, "cpu")
