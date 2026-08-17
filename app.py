"""HalluciGuard — Hugging Face Gradio Space entry point.

Architecture:
  1. FastAPI app with REST endpoints (/health, /verify, /docs) is defined in orchestration.api
  2. Gradio UI is created here and mounted onto that FastAPI app
  3. The combined app is served by Uvicorn on port 7860

HF Spaces with sdk=gradio will run this file directly.
"""

import gradio as gr
from orchestration.api import app as fastapi_app


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
demo = gr.Interface(
    fn=lambda query: (
        "HalluciGuard Verification Engine is online.\n\n"
        "This Space exposes a REST API — use the endpoints below:\n"
        "  • GET  /health          — quick health check\n"
        "  • GET  /health?deep=true — deep component check\n"
        "  • POST /verify          — run hallucination verification\n"
        "  • GET  /docs            — interactive API documentation"
    ),
    inputs=gr.Textbox(label="Query", placeholder="Type anything to check status…"),
    outputs=gr.Textbox(label="Engine Status", lines=8),
    title="🛡️ HalluciGuard Verification Engine",
    description=(
        "Production LangGraph Supervisor API with Detector, Verifier & Memory agents. "
        "Use the REST API endpoints or visit **/docs** for the interactive OpenAPI console."
    ),
    allow_flagging="never",
)

# ---------------------------------------------------------------------------
# Mount Gradio onto the existing FastAPI app so both coexist on one server.
# FastAPI routes: /health, /verify, /docs, /api/v1/verify
# Gradio UI: served at root /
# ---------------------------------------------------------------------------
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# ---------------------------------------------------------------------------
# Startup — HF Spaces runs `python app.py` directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
