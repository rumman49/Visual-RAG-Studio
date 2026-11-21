"""Opt-in live generation checks for Ollama, Groq and Hugging Face (Story 1.6).

Run with `uv run pytest -m live_providers -rs`. A provider that is not configured is
skipped (reported as SKIPPED, never passed). Nothing here uses mocks: each test makes a
real provider call and runs one grounded RAG query against a fixture document through
the real API routes. Embeddings use the dependency-free `fast_local` provider so only
generation is under test.

Opt-in configuration (backend `.env` or environment):
- Groq: `GROQ_API_KEY`; model from `VRS_LIVE_GROQ_MODEL` or `VRS_GROQ_GENERATION_MODEL`.
- Hugging Face: `HF_TOKEN` and `VRS_LIVE_HF_MODEL` (or `VRS_HF_GENERATION_MODEL`).
- Ollama: a reachable server and `VRS_LIVE_OLLAMA_MODEL` (or `VRS_OLLAMA_GENERATION_MODEL`).
"""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.rag.dependencies import get_document_store, get_settings, get_vector_store
from app.rag.providers import GroqProvider, HuggingFaceProvider, OllamaProvider
from app.rag.storage import DocumentStore
from app.rag.vector_store import ChromaVectorStore
from app.registry import TEMPLATES_BY_ID

pytestmark = pytest.mark.live_providers

FIXTURE_TEXT = (
    b"Visual RAG Studio Live Verification Fixture\n\n"
    b"The Aurora cache stores embeddings in a Chroma collection named zephyr_index. "
    b"Citations must reference the exact chunk identifier that supported the answer.\n"
)
QUESTION = "Which Chroma collection does the Aurora cache use?"


@dataclass
class LiveProvider:
    name: str
    model: str
    generate: Callable[[], str]
    external: bool


def _first(*values: str | None) -> str | None:
    return next((v.strip() for v in values if v and v.strip()), None)


def _settings() -> Settings:
    return Settings()


def _groq() -> LiveProvider:
    s = _settings()
    key = s.groq_api_key.get_secret_value().strip() if s.groq_api_key else ""
    if not key:
        pytest.skip("Groq not configured: set GROQ_API_KEY")
    model = _first(os.environ.get("VRS_LIVE_GROQ_MODEL"), s.groq_generation_model)
    provider = GroqProvider(key, s.groq_timeout_seconds)
    return LiveProvider("groq", model, lambda: provider.generate(model, "Reply briefly.", "Say OK."), True)


def _huggingface() -> LiveProvider:
    s = _settings()
    token = s.hf_token.get_secret_value().strip() if s.hf_token else ""
    if not token:
        pytest.skip("Hugging Face not configured: set HF_TOKEN")
    model = _first(os.environ.get("VRS_LIVE_HF_MODEL"), s.hf_generation_model)
    if not model:
        pytest.skip("Hugging Face model not configured: set VRS_LIVE_HF_MODEL")
    provider = HuggingFaceProvider(token, s.hf_timeout_seconds)
    return LiveProvider(
        "huggingface", model,
        lambda: provider.generate(model, "Reply briefly.", "Say OK.", policy=s.hf_provider_policy),
        True,
    )


def _ollama() -> LiveProvider:
    s = _settings()
    model = _first(os.environ.get("VRS_LIVE_OLLAMA_MODEL"), s.ollama_generation_model)
    if not model:
        pytest.skip("Ollama model not configured: set VRS_LIVE_OLLAMA_MODEL")
    try:
        httpx.get(f"{s.ollama_base_url.rstrip('/')}/api/tags", timeout=3).raise_for_status()
    except httpx.HTTPError:
        pytest.skip(f"Ollama not reachable at {s.ollama_base_url}")
    provider = OllamaProvider(s.ollama_base_url, s.ollama_timeout_seconds)
    return LiveProvider("ollama", model, lambda: provider.generate(model, "Reply briefly.", "Say OK."), False)


def _grounded_query(live: LiveProvider, tmp_path) -> None:
    settings = _settings().model_copy(update={
        "database_url": f"sqlite:///{(tmp_path / 'live.db').as_posix()}",
        "data_path": tmp_path / "data",
        "chroma_path": tmp_path / "chroma",
        "generation_provider": live.name,
    })
    store = DocumentStore(settings.database_url, settings.data_path)
    vectors = ChromaVectorStore(settings.chroma_path)
    app.dependency_overrides.update({
        get_settings: lambda: settings,
        get_document_store: lambda: store,
        get_vector_store: lambda: vectors,
    })
    try:
        client = TestClient(app)
        graph = TEMPLATES_BY_ID["traditional-rag"].graph.model_dump(mode="json")
        for node in graph["nodes"]:
            if node["node_type_id"] == "embedding.create":
                node["config"] = {
                    "provider": "fast_local",
                    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "dimension": 384,
                }
        uploaded = client.post(
            "/api/v1/documents", files={"file": ("live.txt", FIXTURE_TEXT, "text/plain")}
        )
        assert uploaded.status_code == 201, uploaded.text
        doc_id = uploaded.json()["document_id"]
        indexed = client.post(f"/api/v1/documents/{doc_id}/ingest", json={"graph": graph})
        assert indexed.status_code == 200, indexed.text
        res = client.post("/api/v1/query", json={
            "question": QUESTION, "graph": graph, "document_ids": [doc_id],
            "generation_provider": live.name, "generation_model": live.model,
            "allow_external_processing": live.external,
        })
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["status"] == "completed", body.get("generation_error") or body
        assert body["abstained"] is False
        assert body["generation_provider"] == live.name
        assert body["citations"], "grounded answer must carry citations"
        assert body["citations"][0]["document_id"] == doc_id
        assert re.search(r"zephyr_index", body["answer"], re.IGNORECASE), body["answer"]
    finally:
        vectors.close()
        store.close()
        for dep in (get_settings, get_document_store, get_vector_store):
            app.dependency_overrides.pop(dep, None)


@pytest.mark.parametrize("factory", [_ollama, _groq, _huggingface], ids=["ollama", "groq", "huggingface"])
def test_live_provider_minimal_prompt_and_grounded_query(factory, tmp_path):
    live = factory()
    answer = live.generate()
    assert isinstance(answer, str) and answer.strip(), "minimal prompt returned no text"
    _grounded_query(live, tmp_path)
