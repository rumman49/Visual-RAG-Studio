"""Hugging Face Spaces entrypoint.

Runs the FastAPI backend on port 7860 (the port Spaces exposes) and mounts a
tiny Gradio status page at /ui so the Space satisfies the Gradio SDK.
"""

import os

os.environ.setdefault("VRS_ENV", "production")
os.environ.setdefault("VRS_DATA_PATH", "./data")
os.environ.setdefault("VRS_DATABASE_URL", "sqlite:///./data/visual_rag_studio.db")
os.environ.setdefault("VRS_CHROMA_PATH", "./data/chroma")
os.makedirs("./data", exist_ok=True)

import gradio as gr
import uvicorn

try:  # ZeroGPU hardware expects at least one @spaces.GPU function
    import spaces

    @spaces.GPU
    def _gpu_probe() -> str:
        return "ok"
except ImportError:
    pass

from app.main import app as api

with gr.Blocks(title="Visual RAG Studio API") as demo:
    gr.Markdown(
        "# Visual RAG Studio API\n"
        "Backend is running. Health check: [`/api/health/live`](/api/health/live)"
    )

app = gr.mount_gradio_app(api, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
