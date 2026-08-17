"""HalluciGuard — Hugging Face Gradio Space entry point with ZeroGPU support.

Architecture:
  1. FastAPI app with REST endpoints (/health, /verify, /docs) is defined in orchestration.api
  2. Gradio UI is decorated with @spaces.GPU for ZeroGPU dynamic allocation
  3. Gradio UI is mounted onto the FastAPI app via gr.mount_gradio_app
"""

import asyncio
import json
import os
from typing import Any, Dict

import gradio as gr

# Try importing spaces for Hugging Face ZeroGPU environment
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    spaces = None

from orchestration.api import app as fastapi_app
from orchestration.graph import run_verification


def _sync_verify(user_query: str, domain: str = "general") -> str:
    """Synchronous helper executing the LangGraph verification pipeline."""
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
        # Format clean, readable output for Gradio UI
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


# Define the ZeroGPU-decorated Gradio callback function
if HAS_SPACES:
    @spaces.GPU
    def verify_claim_gradio(user_query: str, domain: str) -> str:
        """Gradio callback decorated with @spaces.GPU for ZeroGPU dynamic allocation."""
        return _sync_verify(user_query, domain)
else:
    def verify_claim_gradio(user_query: str, domain: str) -> str:
        """Gradio callback for CPU fallback mode."""
        return _sync_verify(user_query, domain)


# ---------------------------------------------------------------------------
# Gradio UI Interface
# ---------------------------------------------------------------------------
demo = gr.Interface(
    fn=verify_claim_gradio,
    inputs=[
        gr.Textbox(
            label="Claim / Question to Verify",
            placeholder="e.g. What is the capital of France? Or paste a claim to check for hallucinations...",
            lines=3,
        ),
        gr.Dropdown(
            choices=["general", "biomedical", "finance"],
            value="general",
            label="Domain Context",
        ),
    ],
    outputs=gr.Textbox(label="HalluciGuard Verification Result", lines=12),
    title="🛡️ HalluciGuard Verification Engine",
    description=(
        "Production LangGraph Supervisor API powered by OpenRouter LLM, Detector, Verifier, and Memory Agents.\n"
        "REST API endpoints available: **/health**, **/health?deep=true**, **/verify**, **/docs**"
    ),
    examples=[
        ["What is the capital of France?", "general"],
        ["Is aspirin safe for children with flu?", "biomedical"],
    ],
    allow_flagging="never",
)

# Mount Gradio onto the existing FastAPI app so REST endpoints & UI coexist
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
