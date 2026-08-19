from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from app.models import JudgeVerificationPayload, CorrectorExecutionResult
from app.orchestrator import CorrectorOrchestrator

app = FastAPI(title="HalluciGuard Corrector Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = CorrectorOrchestrator()

@app.post("/correct", response_model=CorrectorExecutionResult)
async def correct_hallucinations(payload: JudgeVerificationPayload):
    try:
        result = await orchestrator.executeCorrectionPipeline(payload)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=repr(e))

# Mount frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def read_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8003, reload=True)
