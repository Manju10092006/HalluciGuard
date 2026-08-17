import gradio as gr
from orchestration.api import app as fastapi_app

# Mount the FastAPI backend onto the Gradio Space
# This allows Hugging Face Spaces to expose all FastAPI endpoints (/verify, /health, /docs)
app = gr.mount_gradio_app(
    fastapi_app,
    gr.Interface(
        fn=lambda x: f"HalluciGuard Engine Running. Use API endpoints (/verify, /health, /docs).",
        inputs="text",
        outputs="text",
        title="HalluciGuard Verification Engine API",
        description="FastAPI Backend running on Hugging Face 16GB CPU Hardware."
    ),
    path="/ui"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
