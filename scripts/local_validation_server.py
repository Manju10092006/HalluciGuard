"""Start the actual local API with original configuration and isolated test stores."""
import argparse
import os
from pathlib import Path
import secrets
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
p = argparse.ArgumentParser()
p.add_argument('--env-file', type=Path, required=True)
p.add_argument('--port', type=int, default=8019)
p.add_argument('--runtime-dir', type=Path, help='Separate local validation persistence directory')
a = p.parse_args()
from dotenv import load_dotenv
load_dotenv(a.env_file)
runtime = (a.runtime_dir or root / 'runtime-logs' / 'api-runtime').resolve()
runtime.mkdir(parents=True, exist_ok=True)
for key, name in {'AUTH_DB_PATH':'auth.db','KG_PERSISTENCE_PATH':'kg.json','CACHE_DB_PATH':'cache.db','VECTOR_STORE_PATH':'vectors','PATTERN_DB_PATH':'patterns.db','TRUST_DB_PATH':'trust.db'}.items():
    os.environ[key] = str(runtime / name)
# No configured JWT secret exists locally: a process-only key is sufficient for
# isolated local auth tests; restarting invalidates those test tokens.
if not os.getenv('JWT_SECRET'):
    os.environ['JWT_SECRET'] = secrets.token_urlsafe(48)
os.environ['CORS_ORIGINS'] = 'http://localhost:3019,http://127.0.0.1:3019,http://localhost:3020,http://127.0.0.1:3020'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['CACHE_ENABLED'] = 'false'
os.environ['VERIFIER_CACHE_ENABLED'] = 'false'
os.environ['ALWAYS_VERIFY'] = 'true'
os.environ['ALLOW_DETECTOR_FAST_PATH'] = 'false'
print('Configured provider keys present:', {k: bool(os.getenv(k)) for k in ['GROQ_API_KEY','GEMINI_API_KEY','OPENROUTER_API_KEY','TAVILY_API_KEY']}, flush=True)
import uvicorn
uvicorn.run('orchestration.api:app', host='127.0.0.1', port=a.port)
