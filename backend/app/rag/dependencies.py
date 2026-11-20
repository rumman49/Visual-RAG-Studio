from functools import lru_cache

from app.config import get_settings
from app.rag.credentials import runtime_credentials
from app.rag.providers import (
    GroqProvider,
    HuggingFaceProvider,
    OllamaProvider,
    SentenceTransformersProvider,
)
from app.rag.storage import DocumentStore
from app.rag.vector_store import ChromaVectorStore


@lru_cache
def get_document_store() -> DocumentStore:
    settings = get_settings()
    return DocumentStore(settings.database_url, settings.data_path)


@lru_cache
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore(get_settings().chroma_path)


@lru_cache
def get_ollama() -> OllamaProvider:
    active = get_settings()
    return OllamaProvider(active.ollama_base_url, active.ollama_timeout_seconds)


@lru_cache
def get_sentence_transformers() -> SentenceTransformersProvider:
    return SentenceTransformersProvider(get_settings().sentence_transformers_local_files_only)


@lru_cache
def get_huggingface() -> HuggingFaceProvider:
    active = get_settings()
    token = active.hf_token.get_secret_value().strip() if active.hf_token else None
    token = token or None
    return HuggingFaceProvider(token, active.hf_timeout_seconds)


def get_groq() -> GroqProvider:
    active = get_settings()
    runtime_key = runtime_credentials.get_groq_api_key()
    environment_key = (
        active.groq_api_key.get_secret_value().strip() if active.groq_api_key else None
    )
    return GroqProvider(runtime_key or environment_key, active.groq_timeout_seconds)
