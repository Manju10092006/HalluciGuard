"""
Test strict package import boundary.
Must NEVER import from halluciguard, agents.detector_agent, Tavily, Wikipedia, n8n, etc.
"""
import sys, os
import ast

def test_import_boundary():
    detector_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "halluciguard_detector"))
    forbidden = ["halluciguard", "agents", "tavily", "wikipedia", "n8n", "corrector", "judge_agent"]
    
    violations = []
    for root, _, files in os.walk(detector_root):
        for f in files:
            if f.endswith(".py"):
                filepath = os.path.join(root, f)
                with open(filepath, "r", encoding="utf-8") as file:
                    content = file.read()
                    for forb in forbidden:
                        if f"import {forb}" in content or f"from {forb}" in content:
                            violations.append((filepath, forb))

    assert len(violations) == 0, f"Forbidden import violations found: {violations}"
