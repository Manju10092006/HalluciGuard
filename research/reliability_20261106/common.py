"""Paths and serialization for isolated, offline research runs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/research_20261106_r1'
ARCHIVE = Path.home() / '.codex/visualizations/2026/10/20/01a15a78-5c5a-71a3-8bef-2ef5b5c305e3'
TECH = ARCHIVE / 'technical_20261105/verification/supplement/records'
PRIOR = ARCHIVE / 'final_editorial_20261106/table_work/archival_replay/evaluation'

def configure():
    os.environ.update({'CUDA_VISIBLE_DEVICES':'-1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                       'TOKENIZERS_PARALLELISM':'false','N8N_RETRIEVAL_ENABLED':'false',
                       'CACHE_ENABLED':'false','VERIFIER_CACHE_ENABLED':'false',
                       'HF_HUB_DISABLE_PROGRESS_BARS':'1'})
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(ROOT/'agents/verifier_agent'))

def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x.strip()]

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:
        json.dump(obj,f,indent=2,ensure_ascii=True,default=str)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def provenance(script, inputs, **kwargs):
    return {'experiment_id':RUN.name,'script':str(Path(script).relative_to(ROOT)),
            'script_sha256':sha(script),'commit':subprocess.check_output(['git','-c','core.fsmonitor=false','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'working_code':{str(p.relative_to(ROOT)):sha(p) for d in ['orchestration','services','agents','halluciguard_detector'] for p in (ROOT/d).rglob('*.py')},
            'inputs':{str(p):sha(p) for p in inputs},'python':sys.version,**kwargs}
