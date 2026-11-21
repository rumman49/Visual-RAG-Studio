# Visual RAG Studio backend image. Runs space_app.py, the same entrypoint as the Hugging Face Gradio Space.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Spaces run containers as UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    VRS_ENV=production \
    VRS_DATA_PATH=/home/user/app/data \
    VRS_DATABASE_URL=sqlite:////home/user/app/data/visual_rag_studio.db \
    VRS_CHROMA_PATH=/home/user/app/data/chroma

WORKDIR /home/user/app

# CPU-only torch keeps the image ~2GB smaller than the default CUDA build
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
# requirements.txt is the single source for hosted installs (incl. gradio and the ragspark pin)
COPY --chown=user requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the embedding model into the image so startup doesn't download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY --chown=user app ./app
COPY --chown=user space_app.py ./space_app.py
RUN mkdir -p data

# Same entrypoint as the Gradio Space: FastAPI at / plus a status page at /ui, on port 7860.
# Requires VRS_API_KEY (and VRS_CORS_ORIGINS for a browser frontend) at runtime.
EXPOSE 7860
CMD ["python", "space_app.py"]
