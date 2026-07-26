from __future__ import annotations
import pytest
from claims import ClaimDecomposer, ClaimNormalizer, ClaimMerger

def test_decompose_compound_claim():
    decomposer = ClaimDecomposer()
    sub_claims = decomposer.decompose("Vitamin C cures cancer and diabetes")
    assert len(sub_claims) >= 2

def test_decompose_atomic_claim():
    decomposer = ClaimDecomposer()
    sub_claims = decomposer.decompose("Vitamin C cures cancer")
    assert len(sub_claims) == 1

def test_decompose_semicolon():
    decomposer = ClaimDecomposer()
    sub_claims = decomposer.decompose("A is true; B is false")
    assert len(sub_claims) == 2

def test_normalize_claim():
    normalizer = ClaimNormalizer()
    text = "   This is a   claim.  "
    normalized = normalizer.normalize(text)
    assert normalized == "this is a claim."


def test_merger():
    merger = ClaimMerger()
    # Mock some sub-claim results to test the merge logic
    merged = merger.merge([{"verdict": "VERIFIED"}, {"verdict": "UNVERIFIABLE"}])
    assert merged is not None
