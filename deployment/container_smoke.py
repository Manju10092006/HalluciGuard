"""Actual offline CPU inference, cache and HTTP smoke; no quality claim."""
import hashlib
import json
from pathlib import Path

import torch
import datasets  # Catch Arrow/NumPy ABI incompatibility before serving traffic.
from sentence_transformers import SentenceTransformer  # noqa: F401
from huggingface_hub import snapshot_download

torch.set_num_threads(1)
root = Path('/app')
weight = root / 'artifacts/detector-best/model.safetensors'
assert hashlib.sha256(weight.read_bytes()).hexdigest() == '470261f81870f5c4f97fe85918d02aa08673ac08744b5cae4d4621ee9d631bf9', 'Unexpected Detector checkpoint'
for repo in ['BAAI/bge-m3', 'BAAI/bge-reranker-large', 'cross-encoder/nli-deberta-v3-base', 'sentence-transformers/all-MiniLM-L6-v2']:
    cached = Path(snapshot_download(repo, local_files_only=True))
    assert any((cached / name).is_file() for name in ['model.safetensors', 'pytorch_model.bin']), repo

from halluciguard_detector.detector import Detector
detector = Detector(root / 'artifacts/detector-best', device='cpu')
result = detector.detect('Paris is the capital of France.', ['Paris is the capital of France.'])
assert abs(sum(result.sentences[0].probabilities.values()) - 1) < 1e-5
from fastapi.testclient import TestClient
from orchestration.api import app
with TestClient(app) as client:
    assert client.get('/health').status_code == 200
    assert client.get('/auth/me').status_code == 401
print(json.dumps({'offline_detector_inference': True, 'model_caches_present': True, 'http_health': True, 'model_quality_certified': False}))
