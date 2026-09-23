"""
Shared fixtures for verifier agent tests.

The n8n retrieval client now requires an explicit N8N_RETRIEVAL_WEBHOOK_URL
(no hardcoded fallback). These unit tests mock httpx entirely, so a dummy URL
is safe to set here for the whole session.
"""
import os

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