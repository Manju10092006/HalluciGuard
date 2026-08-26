"""
HalluciGuard Detector Agent — Hugging Face Gradio Space Application (ZeroGPU ready).

Thin Gradio deployment layer wrapping the existing DetectorAgent.
Loads the fine-tuned HaluEval model from Manjunath2000006/halluciguard-detector.
"""

import logging
from typing import Any, Dict, Optional, Tuple
import gradio as gr

# Optional ZeroGPU support on Hugging Face Spaces
try:
    import spaces
    has_spaces = True
except ImportError:
    spaces = None
    has_spaces = False

# Support running directly or as a submodule
try:
    from .detector.detector import DetectorAgent
    from .detector.config import DetectorConfig
    from .detector.models import NextAction, RiskLevel
except ImportError:
    from detector.detector import DetectorAgent
    from detector.config import DetectorConfig
    from detector.models import NextAction, RiskLevel

try:
    from .app import app as fastapi_app
except ImportError:
    try:
        from app import app as fastapi_app
    except ImportError:
        fastapi_app = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("halluciguard.gradio")

# Initialize Detector Agent
config = DetectorConfig()
agent = DetectorAgent(config=config)

# Pre-warm model at startup so logs appear immediately in HF Space logs
print("[STARTUP] HalluciGuard Detector Gradio Space starting")
logger.info("[STARTUP] HalluciGuard Detector Gradio Space starting")
agent._ensure_model_loaded()



def _detect_impl(
    user_query: str,
    llm_response: str,
    context: Optional[str] = None,
) -> Tuple[float, float, str, str, str, Dict[str, Any]]:
    """Runs detection using existing DetectorAgent without duplicating logic."""
    if not user_query or not user_query.strip():
        user_query = ""
    if not llm_response or not llm_response.strip():
        llm_response = ""
    if context and not context.strip():
        context = None

    result = agent.detect(
        user_query=user_query,
        llm_response=llm_response,
        context=context,
    )

    pred_label = (
        "HALLUCINATION"
        if result.hallucination_probability >= 0.50
        else "NO_HALLUCINATION"
    )

    diagnostics = {
        "model_source": result.model_source,
        "detector_model_loaded": result.detector_model_loaded,
        "detector_inference_executed": result.detector_inference_executed,
        "detector_degraded": result.detector_degraded,
        "detector_model_source": result.detector_model_source,
    }

    return (
        result.hallucination_probability,
        result.confidence_score,
        pred_label,
        result.risk_level.value,
        result.next_action.value,
        diagnostics,
    )


# Apply ZeroGPU decorator dynamically if available
if has_spaces and hasattr(spaces, "GPU"):
    detect_hallucination = spaces.GPU(_detect_impl)
    logger.info("[STARTUP] ZeroGPU @spaces.GPU decorator enabled")
else:
    detect_hallucination = _detect_impl


# Build Gradio UI
with gr.Blocks(title="HalluciGuard Detector Agent") as demo:
    gr.Markdown(
        """
        # 🛡️ HalluciGuard Detector Agent (HaluEval)
        
        **Real-time LLM Hallucination Risk Classifier** fine-tuned on HaluEval DistilBERT.
        
        * **Model**: [`Manjunath2000006/halluciguard-detector`](https://huggingface.co/Manjunath2000006/halluciguard-detector)
        * **Hardware**: ZeroGPU Accelerated (Dynamic CUDA forward pass)
        * **Routing Policy**: **LOW (≤ 0.30)** & **MEDIUM (0.30–0.50)** → `Accept` | **HIGH (≥ 0.50)** → `Verify`
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="User Query",
                placeholder="e.g., What is the capital of France?",
                lines=2,
            )
            response_input = gr.Textbox(
                label="LLM Response",
                placeholder="e.g., The capital of France is Paris.",
                lines=4,
            )
            context_input = gr.Textbox(
                label="Context (Optional Grounding Reference)",
                placeholder="e.g., France is a country in Western Europe...",
                lines=3,
            )
            detect_btn = gr.Button("🔍 Detect Hallucination", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Detection Results")
            with gr.Row():
                risk_output = gr.Textbox(label="Risk Level", scale=1)
                action_output = gr.Textbox(label="Next Action", scale=1)
                label_output = gr.Textbox(label="Predicted Label", scale=1)

            with gr.Row():
                prob_output = gr.Number(label="Hallucination Probability", precision=4)
                conf_output = gr.Number(label="Confidence Score", precision=4)

            gr.Markdown("### ⚙️ Telemetry & Diagnostics")
            diagnostics_output = gr.JSON(label="Execution Diagnostics")

    detect_btn.click(
        fn=detect_hallucination,
        inputs=[query_input, response_input, context_input],
        outputs=[
            prob_output,
            conf_output,
            label_output,
            risk_output,
            action_output,
            diagnostics_output,
        ],
    )

    gr.Examples(
        examples=[
            [
                "What is the capital of France?",
                "The capital of France is Paris.",
                "",
            ],
            [
                "Who wrote Romeo and Juliet?",
                "Romeo and Juliet was written by Albert Einstein in 1920 in Germany.",
                "",
            ],
            [
                "Summarize the following document.",
                "The president visited Paris on Monday.",
                "The president visited Paris on Monday to discuss international trade.",
            ],
        ],
        inputs=[query_input, response_input, context_input],
    )

if fastapi_app is not None:
    demo.app.include_router(fastapi_app.router)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
