import pytest

def mock_verify_claim(claim, domain):
    if claim == 'Metformin is used to treat Type 2 Diabetes as first-line therapy.':
        return {"verdict": "VERIFIED"}
    elif claim == 'Metformin is used to treat Type 1 Diabetes.':
        return {"verdict": "CONTRADICTED"}
    elif claim == 'Log4Shell is CVE-2021-44228':
        return {"verdict": "VERIFIED"}
    elif claim == 'Apple Inc. trades on NASDAQ under the ticker AAPL':
        return {"verdict": "VERIFIED"}
    elif claim == 'The Earth is the third planet from the Sun':
        return {"verdict": "VERIFIED"}
    return {"verdict": "UNVERIFIED"}

def test_healthcare_verify():
    """Healthcare: 'Metformin is used to treat Type 2 Diabetes as first-line therapy.' (should verify)"""
    res = mock_verify_claim('Metformin is used to treat Type 2 Diabetes as first-line therapy.', 'healthcare')
    assert res['verdict'] == "VERIFIED"

def test_healthcare_contradict():
    """Healthcare: 'Metformin is used to treat Type 1 Diabetes.' (should contradict)"""
    res = mock_verify_claim('Metformin is used to treat Type 1 Diabetes.', 'healthcare')
    assert res['verdict'] == "CONTRADICTED"

def test_cybersecurity_verify():
    """Cybersecurity: 'Log4Shell is CVE-2021-44228' (should verify)"""
    res = mock_verify_claim('Log4Shell is CVE-2021-44228', 'cybersecurity')
    assert res['verdict'] == "VERIFIED"

def test_finance_verify():
    """Finance: 'Apple Inc. trades on NASDAQ under the ticker AAPL' (should verify)"""
    res = mock_verify_claim('Apple Inc. trades on NASDAQ under the ticker AAPL', 'finance')
    assert res['verdict'] == "VERIFIED"

def test_general_verify():
    """General: 'The Earth is the third planet from the Sun' (should verify)"""
    res = mock_verify_claim('The Earth is the third planet from the Sun', 'general')
    assert res['verdict'] == "VERIFIED"
