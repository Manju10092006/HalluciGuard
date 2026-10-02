"""Controlled loader mocks: enforce policy without downloading any model."""
from types import SimpleNamespace
import pytest
from agents.verifier_agent.models import model_manager as module


def manager(monkeypatch, downloads=False, device="cpu"):
    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(
        allow_model_downloads=downloads, nli_model="fixture", reranker_model="fixture-reranker"))
    monkeypatch.setattr(module.ModelManager, "_detect_device", staticmethod(lambda: device))
    result = module.ModelManager()
    monkeypatch.setattr(result, "_evict_if_needed", lambda: None)
    return result


@pytest.mark.parametrize("downloads", [False, True])
def test_nli_tokenizer_and_weights_follow_same_download_policy(monkeypatch, downloads):
    m = manager(monkeypatch, downloads)
    calls = []
    tokenizer, weights, pipeline = object(), object(), object()
    def load_tokenizer(path, **kwargs):
        calls.append(("tokenizer", path, kwargs))
        return tokenizer
    def load_model(path, **kwargs):
        calls.append(("weights", path, kwargs))
        return weights
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(module.AutoModelForSequenceClassification, "from_pretrained", load_model)
    monkeypatch.setattr(module, "hf_pipeline", lambda **kwargs: pipeline if (
        kwargs["model"] is weights and kwargs["tokenizer"] is tokenizer) else None)
    assert m.load_nli_model() is pipeline
    assert m.load_nli_model() is pipeline
    assert len(calls) == 2
    assert all(c[2]["local_files_only"] is (not downloads) for c in calls)


@pytest.mark.parametrize("component", ["tokenizer", "weights"])
def test_missing_offline_cache_does_not_retry_online_or_cache_failure(monkeypatch, component):
    m = manager(monkeypatch)
    def failure(path, **kwargs):
        assert kwargs["local_files_only"] is True
        raise OSError("dummy-private-provider-body")
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained",
                        failure if component == "tokenizer" else lambda *a, **k: object())
    monkeypatch.setattr(module.AutoModelForSequenceClassification, "from_pretrained", failure)
    monkeypatch.setattr(module, "hf_pipeline", lambda **kw: pytest.fail("missing weights must fail before pipeline"))
    with pytest.raises(OSError):
        m.load_nli_model()
    assert not m._models


def test_nli_device_retry_reuses_loaded_offline_objects(monkeypatch):
    m = manager(monkeypatch, device="cuda")
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(module.AutoModelForSequenceClassification, "from_pretrained", lambda *a, **kw: object())
    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        if kwargs["device"] == 0:
            raise RuntimeError("dummy-private-OOM")
        return "cpu-pipeline"
    monkeypatch.setattr(module, "hf_pipeline", factory)
    assert m.load_nli_model() == "cpu-pipeline"
    assert calls[0]["model"] is calls[1]["model"]
    assert calls[0]["tokenizer"] is calls[1]["tokenizer"]
    assert calls[1]["device"] == -1


def test_reranker_cpu_retry_keeps_local_only_and_sanitizes_log(monkeypatch, caplog):
    m = manager(monkeypatch, device="cuda")
    calls = []
    def cross_encoder(path, **kwargs):
        calls.append(kwargs.copy())
        if kwargs["device"] == "cuda":
            raise RuntimeError("dummy-private-secret")
        return "cpu-reranker"
    monkeypatch.setattr(module, "CrossEncoder", cross_encoder)
    assert m.load_reranker_model() == "cpu-reranker"
    assert [c["local_files_only"] for c in calls] == [True, True]
    assert "dummy-private-secret" not in caplog.text


def test_reranker_missing_cache_does_not_publish_healthy_model(monkeypatch):
    m = manager(monkeypatch)
    def failure(path, **kwargs):
        assert kwargs["local_files_only"]
        raise OSError("uncached")
    monkeypatch.setattr(module, "CrossEncoder", failure)
    with pytest.raises(OSError):
        m.load_reranker_model()
    assert not m._models
