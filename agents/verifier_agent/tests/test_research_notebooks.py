from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import yaml

from research.generate_research_notebooks import EXPLICIT_NOTEBOOKS


ROOT = Path(__file__).resolve().parents[1]


def _configured_notebooks() -> Iterable[Path]:
    config = yaml.safe_load((ROOT / "config" / "domain_intelligence.yaml").read_text(encoding="utf-8"))
    seen: set[str] = set()
    for profile in config["domains"].values():
        for notebook_path in profile.get("notebooks", []):
            if notebook_path not in seen:
                seen.add(notebook_path)
                yield ROOT / notebook_path
    for notebook_path in EXPLICIT_NOTEBOOKS:
        if notebook_path not in seen:
            seen.add(notebook_path)
            yield ROOT / notebook_path


def test_all_configured_research_notebooks_exist() -> None:
    notebooks = list(_configured_notebooks())

    assert len(notebooks) >= 30
    missing = [str(path) for path in notebooks if not path.exists()]
    assert missing == []


def test_research_notebooks_import_production_modules() -> None:
    for path in _configured_notebooks():
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        assert "from models.research_companion import" in source
        assert "build_model_research_report" in source
        assert "required_notebook_sections" in source


def test_representative_notebooks_execute_code_cells() -> None:
    representative = [
        ROOT / "research" / "healthcare" / "pubmedbert_demo.ipynb",
        ROOT / "research" / "finance" / "finbert_demo.ipynb",
        ROOT / "research" / "nli" / "deberta_demo.ipynb",
    ]

    for path in representative:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        namespace: dict[str, object] = {}
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                exec("".join(cell.get("source", [])), namespace)
