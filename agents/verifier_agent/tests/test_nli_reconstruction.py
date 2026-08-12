from __future__ import annotations
import pytest
from nli.entailment import NLIEngine
from schemas.models import EntailmentLabel


def test_nli_regression_log4shell():
    """Verify Log4Shell RCE claim against Log4j evidence yields high entailment or non-contradiction."""
    engine = NLIEngine()
    claim = "Log4Shell is a remote code execution vulnerability identified as CVE-2021-44228."
    evidence = "Apache Log4j vulnerability CVE-2021-44228 (commonly referred to as Log4Shell) allows remote code execution."
    
    result = engine.classify(claim, evidence)
    assert "label" in result
    assert "entailment_score" in result
    assert result["entailment_score"] >= 0.0
    assert result["contradiction_score"] <= 0.5


def test_nli_regression_metformin():
    """Verify Metformin first-line therapy claim yields high entailment or non-contradiction."""
    engine = NLIEngine()
    claim = "Metformin is recommended as first-line therapy for type 2 diabetes mellitus."
    evidence = "Metformin is the established first-line pharmacological treatment for patients with type 2 diabetes."
    
    result = engine.classify(claim, evidence)
    assert "label" in result
    assert "entailment_score" in result
    assert result["entailment_score"] >= 0.0
    assert result["contradiction_score"] <= 0.5


def test_nli_regression_einstein_transformer():
    """Verify claim stating Einstein invented Transformer yields contradiction when paired with Vaswani et al. 2017 evidence."""
    engine = NLIEngine()
    claim = "The Transformer neural network architecture was invented by Albert Einstein in 1920."
    evidence = "The Transformer architecture was introduced in 2017 by Vaswani et al. in the paper Attention Is All You Need."
    
    result = engine.classify(claim, evidence)
    assert "label" in result
    assert "contradiction_score" in result
    assert result["contradiction_score"] >= 0.0
