"""Shared hermetic test doubles for the Detector test suite.

The Detector is never allowed to load a real model in tests: it is built with
``Detector.__new__`` and its inference seams (atomic-claim decomposition,
evidence selection, and NLI classification) are replaced by these stubs. Keeping
them here lets every test module share one definition instead of copying it.
"""
from __future__ import annotations

from halluciguard_detector.schemas import ClaimLabel


class EvidenceStub:
    """Stands in for ``ClaimEvidenceEngine``.

    ``decomposed`` drives atomic-claim decomposition, ``checkable`` drives the
    non-factual/opinion filter, and ``snippets`` maps a claim to the evidence
    the stubbed NLI will "see". Any claim without an explicit snippet falls back
    to the full evidence list, mirroring the real engine's behaviour of always
    returning something usable.
    """

    def __init__(self, decomposed=None, checkable=True, snippets=None):
        self._decomposed = decomposed
        self._checkable = checkable
        self._snippets = snippets or {}

    def decompose(self, text):
        return list(self._decomposed or [])

    def checkable(self, text):
        return bool(self._checkable)

    def select(self, claim, evidence_texts, trace=None):
        if trace is not None:
            trace["route"] = "stub"
            trace["degraded"] = False
        if claim in self._snippets:
            return list(self._snippets[claim])
        return list(evidence_texts)


def make_detector(monkeypatch, classify=None, claims=None, evidence_stub=None):
    """Build a ``Detector`` shell with no model and stubbed seams.

    ``classify`` replaces the DeBERTa NLI call; ``claims`` replaces atomic-claim
    decomposition. When neither is given, the real (lazy, model-free) code paths
    are used, which is what the evidence-selection tests exercise.
    """
    from halluciguard_detector.calibration import DEFAULT_MAX_LENGTH
    from halluciguard_detector.detector import Detector

    detector = Detector.__new__(Detector)
    detector.temperature = 1.0
    # Separate thresholds: the near-tie guard must be driven by the contradiction
    # threshold, never by the verification-risk cut-off.
    detector.contradiction_threshold = 0.5
    detector.verification_risk_threshold = 0.5
    detector.threshold = detector.verification_risk_threshold
    detector.max_length = DEFAULT_MAX_LENGTH
    detector.version = "test"
    detector.evidence = evidence_stub or EvidenceStub()
    if claims is not None:
        monkeypatch.setattr(detector, "_atomic_claims", lambda answer: list(claims))
    if classify is not None:
        monkeypatch.setattr(detector, "_classify", classify)
    return detector


def mapping(supported, contradicted, unknown):
    """Build a calibrated class-probability mapping for the three labels."""
    return {
        ClaimLabel.SUPPORTED: supported,
        ClaimLabel.CONTRADICTED: contradicted,
        ClaimLabel.NOT_ENOUGH_INFO: unknown,
    }
