from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger("HalluciGuard.ClaimExtraction")

# Maximum number of atomic claims extracted from a single draft answer.
MAX_DRAFT_CLAIMS = 5


def extract_claims(text: str, max_claims: int = MAX_DRAFT_CLAIMS) -> List[str]:
    """Extract atomic factual claims from a draft answer.

    Uses the verifier's ClaimDecomposer so that the Verifier and ReVerifier
    evaluate the *claims the response actually makes* — never the user query
    or raw full-text dumps. If decomposition yields nothing useful, the whole
    trimmed text is returned as a single claim so the original assertion is
    never silently dropped, but the caller decides how to feed it onward.

    Args:
        text: The draft / corrected answer text to decompose.
        max_claims: Upper bound on the number of claims returned.

    Returns:
        A list of claim strings. Empty only if the input itself is empty.
    """
    if not text or not text.strip():
        return []

    try:
        from agents.verifier_agent.claims.claim_decomposer import ClaimDecomposer

        claims = ClaimDecomposer().decompose(text)
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("Claim extraction failed (%s); falling back to full text.", exc)
        claims = []

    claims = [c.strip() for c in claims if c and c.strip()]
    if not claims:
        claims = [text.strip()]

    return claims[:max_claims]