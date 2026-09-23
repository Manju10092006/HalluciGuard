"""Groq-backed generator for the HalluciGuard Corrector.

This is the production API path for the RARR-inspired correction loop:
authorized target -> evidence-bound revision -> strict JSON candidate ->
existing HalluciGuard validation -> bounded retry -> deterministic reconstruction.

No Groq SDK dependency is required; the client uses the OpenAI-compatible HTTP
endpoint directly. API failures are surfaced to the existing fail-closed model
status path.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from .contracts import Generator if False else None
