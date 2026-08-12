"""HalluciGuard Detector Agent — FastAPI Application.

The existing /detect endpoint remains the stable Detector-only contract.
The /analyze endpoint adds the production orchestration path:

    user query + LLM response
             ↓
        Detector Agent
             ↓
      HIGH / Verify only
             ↓
        Verifier Agent

LOW and MEDIUM results never invoke the Verifier.
"""

import uuid

from fastapi import FastAPI, HTTPException, status

from .detector import DetectorAgent
from .models import DetectionInput, DetectionResult, NextAction
from .pipeline_models import AnalysisInput, AnalysisResult
from .verifier_client import VerifierClient, VerifierUnavailableError

app = FastAPI(
    title="HalluciGuard Detector Agent API",
    description=(
        "First agent in the HalluciGuard multi-agent pipeline. "
        "Estimates LLM response hallucination risk using a HaluEval-trained classifier."
    ),
    version="1.1.0",
)

# Initialize DetectorAgent instance (model loads lazily on first request).
agent = DetectorAgent()
verifier_client = VerifierClient()


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, str]:
    """Health check endpoint for the Detector service itself."""
    return {
        "status": "healthy",
        "service": "detector-agent",
        "model": "halueval-distilbert",
    }


@app.post(
    "/detect",
    response_model=DetectionResult,
    status_code=status.HTTP_200_OK,
    summary="Detect Hallucinations",
    description="Estimates hallucination likelihood using a HaluEval-trained classifier.",
)
def detect_hallucination(payload: DetectionInput) -> DetectionResult:
    """Run Detector only. This endpoint intentionally does not call Verifier."""
    try:
        return agent.detect(
            user_query=payload.user_query,
            llm_response=payload.llm_response,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during hallucination detection: {exc}",
        ) from exc


@app.post(
    "/analyze",
    response_model=AnalysisResult,
    status_code=status.HTTP_200_OK,
    summary="Detect and conditionally verify",
    description=(
        "Runs the HaluEval Detector first. Only HIGH-risk responses are "
        "forwarded to the Verifier Agent. LOW/MEDIUM responses stop at the Detector."
    ),
)
async def analyze_response(payload: AnalysisInput) -> AnalysisResult:
    """Run the complete Detector → conditional Verifier flow.

    Important invariants:
    - The Detector always runs first.
    - LOW/MEDIUM never call the Verifier.
    - HIGH/Verify calls the Verifier exactly once for the candidate response.
    - If a HIGH-risk handoff cannot reach the Verifier, the request fails closed
      with HTTP 503 rather than silently accepting an unverified response.
    - The existing /detect endpoint and DetectionResult schema are untouched.
    """
    query_id = payload.query_id or str(uuid.uuid4())

    try:
        detection = agent.detect(
            user_query=payload.user_query,
            llm_response=payload.llm_response,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detector inference failed: {exc}",
        ) from exc

    # The Detector is the gate. Never bypass this decision.
    if detection.next_action != NextAction.VERIFY:
        return AnalysisResult(
            query_id=query_id,
            detection=detection,
            verifier_invoked=False,
            verifier_result=None,
            final_status="ACCEPTED_BY_DETECTOR",
            message=(
                f"Detector classified the response as {detection.risk_level.value}; "
                "Verifier was not invoked."
            ),
        )

    # HIGH risk: hand the original query/response claim to the Verifier.
    try:
        verifier_result = await verifier_client.verify(
            query_id=query_id,
            domain=payload.domain,
            claim_text=payload.llm_response,
        )
    except VerifierUnavailableError as exc:
        # Never turn a failed HIGH-risk verification into ACCEPT.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "High-risk response could not be verified.",
                "reason": str(exc),
                "query_id": query_id,
                "detector": detection.model_dump(mode="json"),
            },
        ) from exc

    reports = verifier_result.get("claim_evidence") or []
    first_report = reports[0] if reports else {}
    verdict = str(first_report.get("verdict", "insufficient_evidence"))
    normalized_verdict = verdict.rsplit(".", 1)[-1].lower()

    if normalized_verdict == "likely_hallucinated":
        final_status = "LIKELY_HALLUCINATED"
        message = "Verifier found evidence indicating the response is likely hallucinated."
    elif normalized_verdict == "verified":
        final_status = "VERIFIED"
        message = "Verifier found supporting evidence for the claim."
    else:
        final_status = "INSUFFICIENT_EVIDENCE"
        message = "Verifier could not establish sufficient evidence for a definitive verdict."

    return AnalysisResult(
        query_id=query_id,
        detection=detection,
        verifier_invoked=True,
        verifier_result=verifier_result,
        final_status=final_status,
        message=message,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agents.detector_agent.app:app", host="0.0.0.0", port=8000, reload=True)
