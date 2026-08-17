import gradio as gr
from orchestration.api import app as fastapi_app

demo = gr.Interface(
    fn=lambda x: "HalluciGuard Verification Engine API online. Use /health or /verify endpoints.",
    inputs=gr.Textbox(label="Status Check", value="Hello"),
    outputs=gr.Textbox(label="System Status"),
    title="HalluciGuard Engine API",
    description="Production LangGraph Supervisor API running on Hugging Face 16GB CPU Hardware."
)

app = gr.mount_gradio_app(fastapi_app, demo, path="/")
