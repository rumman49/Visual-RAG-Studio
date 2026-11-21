---
title: Visual RAG Studio API
emoji: 🧩
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: space_app.py
pinned: false
---

# Visual RAG Studio – Backend

FastAPI backend for Visual RAG Studio, served from a Gradio Space.
API lives at the Space root (e.g. `/api/health/live`); a status page is at `/ui`.

Required Space secrets/variables (the app refuses to start without `VRS_API_KEY` because `space_app.py` runs in production mode):

| Name | Purpose |
| --- | --- |
| `VRS_API_KEY` | Required. Clients send it as `X-API-Key` on every `/api/v1` request. |
| `VRS_CORS_ORIGINS` | Required for a browser frontend: comma-separated origins such as your Vercel URL. |
| `GROQ_API_KEY` | Optional. Default generation provider. Other provider keys (`HF_TOKEN`, `OPENAI_API_KEY`, ...) are optional too. |

Dependencies come from `requirements.txt` (includes `gradio` and the pinned `ragspark` commit). The same app runs from `Dockerfile` (`python space_app.py`, port 7860). Check a deployment with `GET /api/health/live`, which returns `{"status":"alive"}`.

Set `GROQ_API_KEY` as a Space secret. Storage on a free Space is ephemeral:
indexed documents are lost on restart.
