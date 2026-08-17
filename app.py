"""HalluciGuard — Hugging Face Gradio ZeroGPU Space entry point.

Architecture:
  1. Top-level @spaces.GPU decorated function `verify_claim`
  2. Connected via button.click(fn=verify_claim, ...) inside gr.Blocks() as demo
  3. Gradio UI mounted at /ui on main FastAPI app so /health, /verify, /docs are 100% accessible
"""

import asyncio
import json
import os
from typing import Any, Dict

try:
    import spaces
except ImportError:
    class DummySpaces:
        @staticmethod
        def GPU(fn=None, **kwargs):
            if fn is None:
                return lambda f: f
            return fn
    spaces = DummySpaces()

import gradio as gr
import uvicorn
from orchestration.api import app as fastapi_app
from orchestration.graph import run_verification


@spaces.GPU
def verify_claim(user_query: str, domain: str = "general") -> str:
    """ZeroGPU decorated verification handler."""
    if not user_query or not user_query.strip():
        return "Please enter a statement or query to verify."
    try:
        result = asyncio.run(
            run_verification(
                user_query=user_query.strip(),
                llm_response="",
                domain=domain,
                generation_mode="normal",
            )
        )
        status = result.get("verification_status", "unverified").upper()
        terminal = result.get("terminal_status", "completed")
        final_resp = result.get("final_response") or result.get("draft_response") or "No response generated."
        
        detector_info = result.get("detector", {})
        verifier_info = result.get("verifier", {})
        
        output = [
            f"=== VERIFICATION RESULT: {status} (Status: {terminal}) ===",
            "",
            f"Final Answer:\n{final_resp}",
            "",
            "--- Agent Analysis ---",
        ]
        
        if detector_info:
            output.append(f"• Detector Risk Score: {detector_info.get('hallucination_risk_score', 'N/A')}")
        if verifier_info:
            output.append(f"• Verifier Verdict: {verifier_info.get('verdict', 'N/A')}")
            output.append(f"• Confidence: {verifier_info.get('confidence', 'N/A')}")
            
        return "\n".join(output)
    except Exception as exc:
        return f"Verification Error: {type(exc).__name__}: {exc}"


# Construct Gradio Blocks interface
with gr.Blocks(title="HalluciGuard Verification Engine") as demo:
    gr.Markdown("# 🛡️ HalluciGuard Verification Engine")
    gr.Markdown(
        "Production LangGraph Supervisor API powered by OpenRouter LLM, Detector, Verifier, and Memory Agents.\n\n"
        "REST API endpoints available: **/health**, **/health?deep=true**, **/verify**, **/docs** | UI at **/ui**"
    )
    
    with gr.Row():
        user_input = gr.Textbox(
            label="Claim / Question to Verify",
            placeholder="e.g. What is the capital of France? Or paste a claim to check for hallucinations...",
            lines=3,
        )
        domain_input = gr.Dropdown(
            choices=["general", "biomedical", "finance"],
            value="general",
            label="Domain Context",
        )
    
    verify_btn = gr.Button("Verify Claim", variant="primary")
    output_text = gr.Textbox(label="HalluciGuard Verification Result", lines=12)
    
    verify_btn.click(
        fn=verify_claim,
        inputs=[user_input, domain_input],
        outputs=output_text,
    )

# Mount Gradio UI at /ui on the FastAPI app so root REST endpoints (/health, /verify, /docs) work 100%
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
