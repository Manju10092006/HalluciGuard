import json

from orchestration.runtime_validation import validate_detector_model_reference, validate_openrouter_configuration
from agents.memory_agent.config.settings import Settings


def test_a_configured_groq_router_does_not_require_openrouter(monkeypatch):
    monkeypatch.setenv('HALLUCIGUARD_LLM_PROVIDER_ORDER', 'groq')
    monkeypatch.setenv('GROQ_API_KEY', 'unit-test-only')
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    result = validate_openrouter_configuration()
    assert result.ok
    assert result.metadata['provider'] == 'groq'
    assert 'unit-test-only' not in json.dumps(result.model_dump())


def test_memory_model_is_independent_of_verifier_embedding_configuration(monkeypatch):
    monkeypatch.setenv('EMBEDDING_MODEL', 'BAAI/bge-m3')
    monkeypatch.delenv('MEMORY_EMBEDDING_MODEL', raising=False)
    settings = Settings(_env_file=None)
    assert settings.embedding_model == 'all-MiniLM-L6-v2'
    assert settings.vector_dimension == 384
    monkeypatch.setenv('MEMORY_EMBEDDING_MODEL', 'custom-memory-model')
    assert Settings(_env_file=None).embedding_model == 'custom-memory-model'


def test_lfs_pointer_is_not_a_ready_detector(tmp_path, monkeypatch):
    monkeypatch.setenv('HALLUCIGUARD_DETECTOR_MODEL', str(tmp_path))
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'calibration.json']:
        (tmp_path / name).write_text('{}')
    (tmp_path / 'model.safetensors').write_text('version https://git-lfs.github.com/spec/v1\noid sha256:test\nsize 283348948\n')
    result = validate_detector_model_reference()
    assert not result.ok
    assert 'Git LFS pointer' in result.detail
