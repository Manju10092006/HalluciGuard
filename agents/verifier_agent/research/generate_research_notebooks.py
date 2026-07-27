from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "domain_intelligence.yaml"

EXPLICIT_NOTEBOOKS: Dict[str, Tuple[str, str]] = {
    "research/healthcare/biobert_demo.ipynb": ("medicine", "embedding_model"),
    "research/healthcare/pubmedbert_demo.ipynb": ("healthcare", "embedding_model"),
    "research/healthcare/medcpt_demo.ipynb": ("healthcare", "dense_model"),
    "research/cybersecurity/secbert_demo.ipynb": ("cybersecurity", "classification_model"),
    "research/finance/finbert_demo.ipynb": ("finance", "classification_model"),
    "research/legal/legalbert_demo.ipynb": ("law", "classification_model"),
    "research/embeddings/bge_demo.ipynb": ("general", "embedding_model"),
    "research/embeddings/e5_demo.ipynb": ("general", "sentence_transformer"),
    "research/rerankers/bge_reranker_demo.ipynb": ("general", "reranker"),
    "research/nli/deberta_demo.ipynb": ("general", "nli_model"),
}


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def _infer_model_purpose(path: str) -> str:
    lower = path.lower()
    if "reranker" in lower:
        return "reranker"
    if "nli" in lower or "deberta" in lower:
        return "nli_model"
    if any(token in lower for token in ["finbert", "legalbert", "secbert", "codebert"]):
        return "classification_model"
    if "ner" in lower:
        return "entity_recognition_model"
    if "dense" in lower or "medcpt" in lower:
        return "dense_model"
    return "embedding_model"


def _notebook(path: str, domain: str, model_purpose: str) -> dict:
    title = Path(path).stem.replace("_", " ").title()
    return {
        "cells": [
            _cell(
                "markdown",
                f"""# {title}

This research companion imports HalluciGuard production modules directly. It documents the selected model, source APIs, routing decision, confidence strategy, benchmarking plan, and operational limits for future Verifier Agent contributors.
""",
            ),
            _cell(
                "code",
                f"""from pathlib import Path
import json
import sys

repo_root = Path.cwd()
while repo_root.name != "verifier_agent" and repo_root.parent != repo_root:
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from models.research_companion import (
    build_model_research_report,
    required_notebook_sections,
)

DOMAIN = {domain!r}
MODEL_PURPOSE = {model_purpose!r}
report = build_model_research_report(DOMAIN, MODEL_PURPOSE)
print(json.dumps({{
    "domain": report["domain"],
    "model_purpose": report["model_purpose"],
    "model_id": report["model_id"],
    "adapter": report["adapter"],
    "device": report["routing_decision"]["device"],
}}, indent=2))
""",
            ),
            _cell(
                "markdown",
                """## Research Questions

- What problem this model solves: domain-specific evidence retrieval, ranking, entailment, entity extraction, or classification as configured by the production registry.
- Why HalluciGuard needs this model: the Verifier Agent must select authoritative evidence and score claims differently across domains.
- Why selected over alternatives: selection is centralized in `config/domain_intelligence.yaml` and can be compared against competing model IDs without changing notebook inference code.
""",
            ),
            _cell(
                "code",
                """for section in required_notebook_sections():
    print(f"- {section}")
""",
            ),
            _cell(
                "markdown",
                """## Model And Data Sheet

The cell below reports architecture-facing metadata, training-data notes available from the configured model family, input/output contracts, integration points, and replacement guidance. Contributors should enrich the registry entry when benchmark evidence changes.
""",
            ),
            _cell(
                "code",
                """print(json.dumps({
    "model_id": report["model_id"],
    "model_purpose": report["model_purpose"],
    "input_format": report["input_format"],
    "output_format": report["output_format"],
    "production_modules": report["production_modules"],
    "knowledge_bases": report["knowledge_bases"],
    "api_sources": report["api_sources"],
}, indent=2))
""",
            ),
            _cell(
                "markdown",
                """## Inference And Batch Inference

Do not duplicate inference code in this notebook. Production inference is reached through `ModelManager`, `HybridRetriever`, `CrossEncoderReranker`, `NLIEngine`, and `VerificationPipeline`. Set `allow_model_downloads=true` in the runtime environment only when intentionally benchmarking or warming model caches.
""",
            ),
            _cell(
                "code",
                """print(json.dumps({
    "routing_decision": report["routing_decision"],
    "retrieval_strategy": report["retrieval_strategy"],
    "ranking_strategy": report["evidence_ranking_strategy"],
    "confidence_strategy": report["confidence_strategy"],
    "chunking_strategy": report["chunking_strategy"],
}, indent=2))
""",
            ),
            _cell(
                "markdown",
                """## Benchmarks And Operations

Record CPU latency, GPU latency, memory usage, accuracy, precision, recall, F1, and failure cases for every benchmark run. The notebook intentionally keeps execution lightweight by default; use production benchmark scripts and cached models for full runs.
""",
            ),
            _cell(
                "code",
                """print(json.dumps({
    "benchmarks": report["benchmarks"],
    "recommendation": report["production_recommendation"],
    "failure_cases": [
        "No authoritative source returns evidence",
        "Evidence is stale or jurisdiction-specific",
        "Domain classifier disagreement",
        "Model unavailable and lexical fallback is used",
    ],
    "limitations": [
        "Registry selection is not a substitute for periodic benchmark refresh",
        "Domain adapters may share broad sources until dedicated API adapters are added",
        "High-recall retrieval can increase latency without GPU prewarming",
    ],
}, indent=2))
""",
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _configured_notebooks(config: dict) -> Iterable[Tuple[str, str, str]]:
    for domain, profile in config["domains"].items():
        for notebook_path in profile.get("notebooks", []):
            yield notebook_path, domain, _infer_model_purpose(notebook_path)
    for notebook_path, (domain, model_purpose) in EXPLICIT_NOTEBOOKS.items():
        yield notebook_path, domain, model_purpose


def main() -> None:
    config = _load_config()
    seen = set()
    for notebook_path, domain, model_purpose in _configured_notebooks(config):
        if notebook_path in seen:
            continue
        seen.add(notebook_path)
        target = ROOT / notebook_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(_notebook(notebook_path, domain, model_purpose), indent=2),
            encoding="utf-8",
        )
    print(f"Generated {len(seen)} research notebooks")


if __name__ == "__main__":
    main()
