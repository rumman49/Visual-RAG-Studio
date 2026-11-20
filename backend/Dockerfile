# Visual RAG Studio backend - Hugging Face Spaces (Docker SDK)
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
RUN pip install --no-cache-dir --user \
    "fastapi>=0.120,<1" "uvicorn[standard]>=0.38,<1" "pydantic>=2.12,<3" \
    "pydantic-settings>=2.11,<3" "python-dotenv>=1.1,<2" "sqlalchemy>=2.0,<3" \
    "chromadb>=1.0,<2" "python-multipart>=0.0.20,<1" "pymupdf>=1.26,<2" \
    "httpx>=0.28,<1" "sentence-transformers>=5.1,<6" \
    "ragspark @ git+https://github.com/Adeebkhann/ragspark.git@d1c2fedfb980d406265faaa6650fc322ffd2693a"

# Bake the embedding model into the image so startup doesn't download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY --chown=user app ./app
RUN mkdir -p data

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
