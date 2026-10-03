"""
Shared fixtures for verifier agent tests.

The n8n retrieval client now requires an explicit N8N_RETRIEVAL_WEBHOOK_URL
(no hardcoded fallback). These unit tests mock httpx entirely, so a dummy URL
is safe to set here for the whole session.
"""
import os
import re
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True, scope="session")
def _use_dummy_n8n_url() -> None:
    os.environ.setdefault(
        "N8N_RETRIEVAL_WEBHOOK_URL",
        "https://test.n8n.cloud/webhook/halluciguard-verify",
    )
    os.environ.setdefault(
        "N8N_HEALTH_WEBHOOK_URL",
        "https://guru-siesta-excusable.ngrok-free.dev/healthz",
    )


class _DeterministicRerankerModel:
    """Small lexical-overlap reranker used for deterministic CI tests."""

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def predict(self, pairs, batch_size=16):
        scores = []
        for claim, snippet in pairs:
            claim_tokens = set(self._TOKEN_RE.findall((claim or "").lower()))
            snippet_text = (snippet or "").lower()
            snippet_tokens = set(self._TOKEN_RE.findall(snippet_text))
            overlap = len(claim_tokens & snippet_tokens) / max(1, len(claim_tokens))
            if "capital" in claim_tokens and "capital" in snippet_tokens:
                overlap += 0.2
            scores.append(round(overlap, 6))
        return scores


class _DeterministicNLIPipeline:
    """HF-like NLI pipeline returning deterministic label probabilities."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                id2label={0: "contradiction", 1: "entailment", 2: "neutral"}
            )
        )

    @staticmethod
    def _extract_location(text: str) -> str | None:
        match = re.search(r"\bin\s+([a-z][a-z-]*)", (text or "").lower())
        return match.group(1) if match else None

    def _rows(self, premise: str, hypothesis: str):
        premise_l = (premise or "").lower()
        hypothesis_l = (hypothesis or "").lower()
        premise_tokens = set(re.findall(r"[a-z0-9]+", premise_l))
        hypothesis_tokens = set(re.findall(r"[a-z0-9]+", hypothesis_l))
        overlap = len(premise_tokens & hypothesis_tokens) / max(1, len(hypothesis_tokens))

        claim_location = self._extract_location(hypothesis_l)
        evidence_location = self._extract_location(premise_l)
        location_conflict = (
            claim_location
            and evidence_location
            and claim_location != evidence_location
            and {"located", "capital"} & hypothesis_tokens
        )

        if location_conflict:
            entailment, contradiction, neutral = 0.01, 0.98, 0.01
        elif overlap >= 0.35:
            entailment, contradiction, neutral = 0.98, 0.01, 0.01
        else:
            entailment, contradiction, neutral = 0.1, 0.1, 0.8

        return [
            {"label": "entailment", "score": entailment},
            {"label": "contradiction", "score": contradiction},
            {"label": "neutral", "score": neutral},
        ]

    def __call__(self, payload, **kwargs):
        if isinstance(payload, list):
            return [self._rows(item.get("text", ""), item.get("text_pair", "")) for item in payload]
        return self._rows(payload.get("text", ""), payload.get("text_pair", ""))


@pytest.fixture
def deterministic_model_doubles(monkeypatch):
    """Inject deterministic NLI/reranker model doubles into runtime wrappers."""

    class _ModelManagerDouble:
        def load_reranker_model(self, model_name=None):
            return _DeterministicRerankerModel()

        def load_nli_model(self, model_name=None):
            return _DeterministicNLIPipeline()

    model_manager = _ModelManagerDouble()
    monkeypatch.setattr("rerankers.cross_encoder.get_model_manager", lambda: model_manager)
    monkeypatch.setattr("nli.robust_entailment.get_model_manager", lambda: model_manager)
    return model_manager