"""Freeze research inputs without reading credentials or changing source data."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research_20261106_r1'
ARCHIVE = Path.home() / '.codex/visualizations/2026/10/20/01a15a78-5c5a-71a3-8bef-2ef5b5c305e3'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def git(*args):
    return subprocess.check_output(['git', '-c', 'core.fsmonitor=false', *args], cwd=ROOT).decode().strip()

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    baseline = OUT / 'baseline'
    baseline.mkdir()
    files = []
    for directory in ['orchestration', 'services', 'agents', 'halluciguard_detector', 'halluciguard_judge']:
        for path in (ROOT / directory).rglob('*.py'):
            if '__pycache__' not in path.parts:
                target = baseline / 'source' / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                files.append({'path':str(path.relative_to(ROOT)), 'sha256':sha(path), 'bytes':path.stat().st_size})
    # The reviewed source archive already includes original figure and table files.
    deliver = ARCHIVE / 'final_editorial_20261106/deliverables'
    for name in ['HalluciGuard_Final_Reviewed_Manuscript.pdf', 'HalluciGuard_Final_Reviewed_LaTeX.zip',
                 'HalluciGuard_Final_Reviewed_Supplement.pdf', 'HalluciGuard_Final_Quality_Report.md']:
        src = deliver / name
        shutil.copy2(src, baseline / name)
        files.append({'path':str(src), 'sha256':sha(src), 'bytes':src.stat().st_size})
    evidence = []
    for directory in [ROOT/'artifacts', ROOT/'data', ROOT/'benchmarks',
                      ARCHIVE/'technical_20261105/verification/supplement/records',
                      ARCHIVE/'final_editorial_20261106/table_work/archival_replay/evaluation']:
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix.lower() in {'.json','.jsonl','.npz','.csv','.parquet','.safetensors','.bin','.pt','.pth','.xml'}:
                evidence.append({'path':str(path), 'sha256':sha(path), 'bytes':path.stat().st_size})
    info = {'experiment_id':OUT.name, 'cwd':str(ROOT), 'commit':git('rev-parse','HEAD'),
            'branch':git('branch','--show-current'), 'initial_status':git('status','--short'),
            'python':sys.version, 'executable':sys.executable,
            'packages':{p:metadata.version(p) for p in ['torch','transformers','numpy','scikit-learn','pytest','pydantic','langgraph']},
            'frozen_sources_and_manuscript':files, 'preserved_evidence_inventory':evidence}
    (baseline/'inventory.json').write_text(json.dumps(info,indent=2),encoding='utf8')
    patch = subprocess.check_output(['git','-c','core.fsmonitor=false','diff','--binary','--','README.md','agents','orchestration','services','scripts','halluciguard_detector'],cwd=ROOT)
    (baseline/'preexisting_changes.patch').write_bytes(patch)
    print(json.dumps({'baseline':str(baseline),'sources':len(files),'evidence_files':len(evidence),'commit':info['commit']}))

if __name__ == '__main__':
    main()
