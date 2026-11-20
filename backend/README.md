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

Set `GROQ_API_KEY` as a Space secret. Storage on a free Space is ephemeral:
indexed documents are lost on restart.
