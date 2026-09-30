from concurrent.futures import ThreadPoolExecutor

import pytest

from halluciguard_detector import agent as module


def test_distinct_detector_configurations_never_share_model(monkeypatch, tmp_path):
    calls = []

    class FakeDetector:
        def __init__(self, path, device):
            calls.append((path, device))

    monkeypatch.setattr(module, "Detector", FakeDetector)
    first = module.DetectorAgent(tmp_path / "checkpoint-a", "cpu")
    second = module.DetectorAgent(tmp_path / "checkpoint-b", "cuda")
    assert first._get_detector() is first._get_detector()
    assert second._get_detector() is second._get_detector()
    assert first._get_detector() is not second._get_detector()
    assert calls == [(first.model_path, "cpu"), (second.model_path, "cuda")]


def test_failed_load_does_not_poison_next_attempt(monkeypatch, tmp_path):
    calls = 0

    class FlakyDetector:
        def __init__(self, path, device):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("transient load failure")

    monkeypatch.setattr(module, "Detector", FlakyDetector)
    agent = module.DetectorAgent(tmp_path / "checkpoint")
    with pytest.raises(RuntimeError, match="transient"):
        agent._get_detector()
    assert isinstance(agent._get_detector(), FlakyDetector)
    assert calls == 2


def test_concurrent_requests_initialize_once(monkeypatch, tmp_path):
    calls = 0

    class FakeDetector:
        def __init__(self, path, device):
            nonlocal calls
            calls += 1

    monkeypatch.setattr(module, "Detector", FakeDetector)
    agent = module.DetectorAgent(tmp_path / "checkpoint")
    with ThreadPoolExecutor(max_workers=8) as pool:
        detectors = list(pool.map(lambda _: agent._get_detector(), range(16)))
    assert calls == 1
    assert all(item is detectors[0] for item in detectors)
