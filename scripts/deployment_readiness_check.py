#!/usr/bin/env python3
"""
HalluciGuard Automated Pre-Flight Deployment Readiness Check Script.
Validates Python runtime, dependencies, module imports, route instantiation,
and environment setup before deployment.
"""

import sys
import os
import subprocess
from typing import List, Tuple

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

def run_check(name: str, fn) -> bool:
    print(f"[*] Checking {name}...", end=" ", flush=True)
    try:
        msg = fn()
        print(f"[PASS] ({msg if msg else 'OK'})")
        return True
    except Exception as exc:
        print(f"[FAIL] ({exc})")
        return False

def check_python_version() -> str:
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        raise RuntimeError(f"Python >= 3.10 required, got {v.major}.{v.minor}")
    return f"Python {v.major}.{v.minor}.{v.micro}"

def check_core_imports() -> str:
    import fastapi
    import uvicorn
    import pydantic
    import torch
    import transformers
    import sentence_transformers
    import faiss
    import datasets
    import aiosqlite
    import networkx
    import numpy
    import huggingface_hub
    return f"PyTorch {torch.__version__}, Transformers {transformers.__version__}, FastAPI {fastapi.__version__}"

def check_app_imports() -> str:
    from services.base_llm_service import BaseLLMService, GenerationErrorCode
    from orchestration.api import app
    from orchestration.graph import build_verification_graph
    from orchestration.runtime_validation import validate_orchestration_startup
    return "FastAPI app & LangGraph supervisor loaded successfully"

def check_fastapi_routes() -> str:
    from orchestration.api import app
    routes = [r.path for r in app.routes]
    expected = ["/", "/health", "/verify", "/api/v1/verify", "/docs", "/openapi.json"]
    missing = [p for p in expected if p not in routes]
    if missing:
        raise ValueError(f"Missing expected FastAPI routes: {missing}")
    return f"All {len(expected)} expected API routes mounted"

def check_git_tracked_binaries() -> str:
    cmd = ["git", "ls-files"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        files = res.stdout.splitlines()
        binaries = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.bin'))]
        if binaries:
            raise ValueError(f"Found {len(binaries)} tracked binary assets in Git index: {binaries[:3]}")
    return "Git index free of raw binary asset files"

def main():
    print("=" * 60)
    print("  HalluciGuard Deployment Readiness Verification  ")
    print("=" * 60)
    
    checks: List[Tuple[str, callable]] = [
        ("Python Version Compatibility", check_python_version),
        ("Core ML & API Dependencies", check_core_imports),
        ("Application Module Imports", check_app_imports),
        ("FastAPI Route Definitions", check_fastapi_routes),
        ("Git Tracked Binary Assets Check", check_git_tracked_binaries),
    ]
    
    results = [run_check(name, fn) for name, fn in checks]
    
    print("-" * 60)
    if all(results):
        print(" SUCCESS: All deployment readiness checks passed cleanly!")
        sys.exit(0)
    else:
        print(" ERROR: Deployment readiness check failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
