from fastapi import FastAPI, HTTPException, status
from .detector import DetectorAgent
from .models import DetectionInput, DetectionResult

app = FastAPI(
    title="HalluciGuard Detector Agent API",
    description="First agent in the HalluciGuard multi-agent pipeline. Estimates LLM response hallucination risk.",
    version="0.1.0"
)

# Initialize DetectorAgent instance
agent = DetectorAgent()


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, str]:
    """Health check endpoint to verify service availability."""
    return {"status": "healthy", "service": "detector-agent"}


@app.post(
    "/detect",
    response_model=DetectionResult,
    status_code=status.HTTP_200_OK,
    summary="Detect Hallucinations",
    description="Estimates hallucination likelihood and recommends pipeline action (Accept or Verify)."
)
def detect_hallucination(payload: DetectionInput) -> DetectionResult:
    """FastAPI endpoint exposing DetectorAgent.detect method.
    
    Accepts user_query and llm_response in the body payload, returning
    confidence_score, hallucination_probability, risk_level, and next_action.
    """
    try:
        result = agent.detect(
            user_query=payload.user_query,
            llm_response=payload.llm_response
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during hallucination detection: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("detector_agent.app:app", host="0.0.0.0", port=8000, reload=True)
