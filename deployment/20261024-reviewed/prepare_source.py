"""Export the reviewed Git snapshot to a new, secret-free Cloud Build context.

This only writes local files. It never downloads weights, builds or deploys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess

REVIEW_COMMIT = "55ef3570cf9bb5d6556aeb404699d52e29085d2d"
ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
GIT_DIR = ROOT / "runtime-logs/source-review-20261024-131352.git"
BLOCKED = {"tests", "training", "data", "scratch", "third_party", "research",
           "reports", "runtime-logs", "__pycache__"}
EXCLUDED = {"phase1_cli.py", "phase1_collect.py", "phase1_data.py", "phase1_train.py",
            "local_generation.py", "training.py", "audit.py", "free_space_runtime.py"}
ARTIFACTS = {"config.json", "tokenizer.json", "tokenizer_config.json",
             "calibration.json", "model_metadata.json"}
BUILD_FILES = {"Dockerfile", ".dockerignore", "requirements-linux.lock", "models-pinned.json",
               "download_models.py", "hydrate_detector.py", "container_selftest.py",
               "cloudbuild.yaml", "candidate-config.json", "deploy_candidate.py"}


def permitted(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return False
    if any(part.startswith(".env") for part in path.parts) or set(path.parts) & BLOCKED:
        return False
    if path.parts[:2] == ("artifacts", "detector-best"):
        return len(path.parts) == 3 and path.name in ARTIFACTS
    return (path.parts[0] in {"orchestration", "services", "agents", "halluciguard_detector"}
            and path.name not in EXCLUDED and path.suffix in {".py", ".json", ".yaml", ".yml"})


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "--git-dir=" + str(GIT_DIR), *args])


def prepare(output: Path) -> dict:
    # Read the immutable object, never the dirty primary or another worktree.
    if git("rev-parse", REVIEW_COMMIT + "^{commit}").decode().strip() != REVIEW_COMMIT:
        raise RuntimeError("Reviewed commit is unavailable")
    if not output.parent.is_dir():
        raise ValueError("Output parent must already exist")
    output.mkdir(exist_ok=False)
    manifest = {"review_commit": REVIEW_COMMIT, "source": "immutable Git object",
                "artifact_readiness_certified": False, "quality_certified": False,
                "memory_durable": False, "files": {}}
    names = git("ls-tree", "-r", "--name-only", "-z", REVIEW_COMMIT).decode().split("\0")
    for name in sorted(name for name in names if permitted(name)):
        content = git("show", REVIEW_COMMIT + ":" + name)
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest["files"][name] = hashlib.sha256(content).hexdigest()
    for name in sorted(BUILD_FILES):
        source = BASE / name
        shutil.copyfile(source, output / name)
        manifest["files"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest["detector"] = {
        "source_commit": "82cdc6fb734f5fa31cd345a4060f60178bea2118",
        "bytes": 283348948,
        "sha256": "470261f81870f5c4f97fe85918d02aa08673ac08744b5cae4d4621ee9d631bf9",
        "hydrated_during_cloud_build": True,
    }
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = prepare(args.output.resolve())
    print(json.dumps({"output": str(args.output.resolve()), "files": len(manifest["files"]),
                      "review_commit": REVIEW_COMMIT, "model_downloaded": False}))
