import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repair_n8n_workflow.py"
SPEC = importlib.util.spec_from_file_location("repair_n8n_workflow", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_repair_removes_missing_connections_and_analyze_dependency():
    workflow = {
        "nodes": [
            {"name": "Receive Claim", "parameters": {}},
            {"name": "Build Context", "parameters": {"jsCode": "broken"}},
        ],
        "connections": {
            "Receive Claim": {
                "main": [[{"node": "Build Context", "type": "main", "index": 0}]]
            },
            "Build Context": {
                "main": [[{"node": "Deleted Node", "type": "main", "index": 0}]]
            },
            "Missing Source": {"main": [[]]},
        },
    }

    repaired = MODULE.repair_workflow(workflow)

    code = repaired["nodes"][1]["parameters"]["jsCode"]
    assert "$('Analyze Claim')" not in code
    assert "body.queries" in code
    assert repaired["connections"]["Build Context"]["main"] == [[]]
    assert "Missing Source" not in repaired["connections"]
