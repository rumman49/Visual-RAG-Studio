import json

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.contracts import WorkflowValidationRequest
from app.main import app, validate_workflow
from app.rag.chunking import ChunkingError, chunk_document
from app.rag.dependencies import (
    get_document_store,
    get_groq,
    get_huggingface,
    get_ollama,
    get_sentence_transformers,
    get_vector_store,
)
from app.rag.execution import ExecutionTracker
from app.rag.graph import configured_provider_instances, validate_traditional_graph
from app.rag.parser import DocumentParseError, parse_document
from app.rag.pipeline import PipelineError, answer_query, index_document
from app.rag.providers import (
    AnthropicProvider,
    FastLocalEmbeddingProvider,
    GeminiProvider,
    GroqProvider,
    HuggingFaceProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    SentenceTransformersProvider,
)
from app.rag.recommendations import recommend_chunking
from app.rag.routes import _is_local_origin
from app.rag.storage import DocumentStore
from app.rag.vector_store import ChromaVectorStore, VectorStoreError
from app.registry import TEMPLATES_BY_ID


def test_markdown_profile_recommendation_and_offsets_are_real():
    source = b"# Overview\n\n  A useful paragraph. More detail here.  \n"
    parsed = parse_document("doc-1", "guide.md", source)
    chunk = chunk_document(parsed, "structure_aware", {"chunk_size": 100, "overlap": 10})[0]
    assert parsed.profile.file_type.value == "markdown"
    assert parsed.profile.structure.has_headings.value is True
    assert recommend_chunking(parsed.profile).recommended_strategy_id == "structure_aware"
    assert source[chunk.source_start : chunk.source_end].decode().strip() == chunk.text
    assert chunk.section_path == ("Overview",)


@pytest.mark.parametrize(
    "name,body", [("empty.txt", b" \n "), ("bad.txt", b"\xff"), ("unknown.docx", b"text")]
)
def test_parser_rejects_empty_invalid_and_unsupported_files(name, body):
    with pytest.raises(DocumentParseError):
        parse_document("doc", name, body)


def test_all_initial_chunkers_return_bounded_nonempty_chunks():
    parsed = parse_document("doc", "note.txt", ("Sentence one. Sentence two. " * 20).encode())
    for strategy in ("recursive", "fixed", "sentence", "structure_aware"):
        chunks = chunk_document(parsed, strategy, {"chunk_size": 64, "overlap": 8})
        assert chunks and all(chunk.text.strip() for chunk in chunks)
        assert all(chunk.document_id == "doc" and chunk.strategy_id == strategy for chunk in chunks)
    with pytest.raises(ChunkingError):
        chunk_document(parsed, "semantic")


def test_ollama_adapter_parses_real_api_shape_and_surfaces_provider_errors():
    def response(request):
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
        return httpx.Response(503, text="service unavailable")

    client = httpx.Client(transport=httpx.MockTransport(response))
    provider = OllamaProvider("http://ollama", client=client)
    assert provider.embed("nomic-embed-text", ["one", "two"]) == ([[0.1, 0.2], [0.3, 0.4]], 2)
    with pytest.raises(ProviderError, match="HTTP 503"):
        provider.generate("llama", "system", "question")
    client.close()


def test_multilingual_sentence_transformer_is_default_embedding_provider():
    settings = Settings()
    embedding = configured_provider_instances(settings)[0]
    assert settings.embedding_provider == "sentence_transformers"
    assert settings.sentence_transformers_model == (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    assert embedding.id == "embedding-provider"
    assert embedding.adapter_id == "sentence-transformers.embedding"
    with pytest.raises(ProviderError, match="non-empty text"):
        SentenceTransformersProvider().embed(settings.sentence_transformers_model, [""])


def test_huggingface_adapter_sends_grounded_messages_and_parses_chat_completion():
    captured = {}

    def response(request):
        captured.update(request=request, payload=__import__("json").loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "Grounded answer [chunk-1]"}}]}
        )

    client = httpx.Client(transport=httpx.MockTransport(response))
    provider = HuggingFaceProvider("hf-test-secret", client=client)
    output = provider.generate(
        "org/model",
        "System rules",
        "Retrieved context",
        max_tokens=64,
        temperature=0.1,
        policy="cheapest",
    )
    assert output == "Grounded answer [chunk-1]"
    assert captured["request"].headers["Authorization"] == "Bearer hf-test-secret"
    assert captured["payload"]["model"] == "org/model:cheapest"
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "System rules"},
        {"role": "user", "content": "Retrieved context"},
    ]
    assert captured["payload"]["max_tokens"] == 64
    client.close()


def test_groq_adapter_sends_secret_server_side_and_parses_completion():
    captured = {}

    def response(request):
        captured["authorization"] = request.headers.get("authorization")
        captured["payload"] = __import__("json").loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "Grounded [chunk-1]"}}]}
        )

    client = httpx.Client(transport=httpx.MockTransport(response))
    provider = GroqProvider("gsk-test-secret", client=client)
    assert (
        provider.generate("openai/gpt-oss-20b", "System rules", "Evidence", 64, 0.1)
        == "Grounded [chunk-1]"
    )
    assert captured["authorization"] == "Bearer gsk-test-secret"
    assert captured["payload"]["model"] == "openai/gpt-oss-20b"
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "System rules"},
        {"role": "user", "content": "Evidence"},
    ]
    assert captured["payload"]["include_reasoning"] is False
    client.close()


def test_groq_adapter_only_sends_reasoning_option_to_gpt_oss_models():
    captured = []

    def response(request):
        captured.append(__import__("json").loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    client = httpx.Client(transport=httpx.MockTransport(response))
    provider = GroqProvider("gsk-test-secret", client=client)
    provider.generate("llama-3.3-70b-versatile", "system", "prompt")
    assert "include_reasoning" not in captured[0]
    client.close()


def test_ollama_adapter_handles_thinking_tokens_gracefully():
    def response(request):
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "thinking": "Analysis shows the answer is 42.",
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(response))
    provider = OllamaProvider("http://ollama", client=client)
    res = provider.generate("qwen3:1.7b", "system", "question")
    assert res == "Analysis shows the answer is 42."
    client.close()


def test_openai_adapter_sends_bearer_token_and_parses_chat_completion():
    captured = {}

    def response(request):
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "OpenAI answer [chunk:1]"}}]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(response))
    provider = OpenAIProvider(api_key="sk-test-openai-123", client=client)
    answer = provider.generate("gpt-4o-mini", "system rules", "user prompt", max_tokens=100)
    assert answer == "OpenAI answer [chunk:1]"
    assert captured["auth"] == "Bearer sk-test-openai-123"
    assert captured["body"]["model"] == "gpt-4o-mini"
    assert captured["body"]["max_tokens"] == 100
    client.close()


def test_gemini_adapter_sends_key_in_url_and_parses_content():
    captured = {}

    def response(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Gemini answer [chunk:2]"}]
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(response))
    provider = GeminiProvider(api_key="AIzaSyTestKey123", client=client)
    answer = provider.generate("gemini-2.0-flash", "system prompt", "user query", max_tokens=80)
    assert answer == "Gemini answer [chunk:2]"
    assert "key=AIzaSyTestKey123" in captured["url"]
    assert captured["body"]["system_instruction"]["parts"][0]["text"] == "system prompt"
    assert captured["body"]["generationConfig"]["maxOutputTokens"] == 80
    client.close()


def test_anthropic_adapter_sends_x_api_key_and_parses_content():
    captured = {}

    def response(request):
        captured["api_key"] = request.headers.get("x-api-key")
        captured["version"] = request.headers.get("anthropic-version")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "Anthropic answer [chunk:3]"}
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(response))
    provider = AnthropicProvider(api_key="sk-ant-test-456", client=client)
    answer = provider.generate("claude-3-5-sonnet-20241022", "anthropic system", "user question", max_tokens=120)
    assert answer == "Anthropic answer [chunk:3]"
    assert captured["api_key"] == "sk-ant-test-456"
    assert captured["version"] == "2023-06-01"
    assert captured["body"]["model"] == "claude-3-5-sonnet-20241022"
    assert captured["body"]["max_tokens"] == 120
    client.close()


def test_generic_provider_credentials_endpoints():
    client = TestClient(app)
    res = client.get("/api/v1/providers/credentials/openai")
    assert res.status_code == 200
    assert res.json()["provider"] == "openai"
    assert res.json()["configured"] is False

    post_res = client.post("/api/v1/providers/credentials/openai", json={"api_key": "sk-runtime-key"})
    assert post_res.status_code == 200
    assert post_res.json()["configured"] is True
    assert post_res.json()["source"] == "session"

    check_res = client.get("/api/v1/providers/credentials/openai")
    assert check_res.json()["configured"] is True

    del_res = client.delete("/api/v1/providers/credentials/openai")
    assert del_res.status_code == 200
    assert del_res.json()["configured"] is False

    final_res = client.get("/api/v1/providers/credentials/openai")
    assert final_res.json()["configured"] is False
    client.close()


def test_groq_adapter_never_includes_key_in_provider_errors():
    secret = "gsk-never-echo-this"
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(401, text=secret)))
    with pytest.raises(ProviderError, match="rejected the API key") as error:
        GroqProvider(secret, client=client).generate("openai/gpt-oss-20b", "system", "prompt")
    assert secret not in str(error.value)
    client.close()


@pytest.mark.parametrize(
    "status,body,message",
    [
        (400, "bad model", "rejected the model"),
        (401, "invalid token", "rejected the token"),
        (403, "permission denied", "rejected the token"),
        (402, "no balance", "no available account credit"),
        (404, "not found", "model/provider is unavailable"),
        (429, "rate limit", "rate limit reached"),
        (429, "credit quota exceeded", "quota or credits are exhausted"),
        (503, "private provider body", "failed with HTTP 503"),
    ],
)
def test_huggingface_adapter_maps_service_errors_without_leaking_body(status, body, message):
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, text=body))
    )
    provider = HuggingFaceProvider("secret", client=client)
    with pytest.raises(ProviderError, match=message):
        provider.generate("org/model", "system", "prompt")
    client.close()


def test_huggingface_adapter_rejects_missing_token_invalid_model_and_malformed_completion():
    with pytest.raises(ProviderError, match="HF_TOKEN is missing"):
        HuggingFaceProvider(None).generate("org/model", "system", "prompt")
    with pytest.raises(ProviderError, match="valid Hugging Face model ID"):
        HuggingFaceProvider("secret").generate("not-a-model", "system", "prompt")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"choices": []}))
    )
    with pytest.raises(ProviderError, match="empty chat completion"):
        HuggingFaceProvider("secret", client=client).generate("org/model", "system", "prompt")
    client.close()


def test_huggingface_adapter_surfaces_timeout_without_fallback():
    def timeout(_):
        raise httpx.ReadTimeout("timed out")

    client = httpx.Client(transport=httpx.MockTransport(timeout))
    with pytest.raises(ProviderError, match="timed out"):
        HuggingFaceProvider("secret", client=client).generate("org/model", "system", "prompt")
    client.close()


def test_chroma_persists_and_retrieves_vectors_and_rejects_dimension_mismatch(tmp_path):
    first = ChromaVectorStore(tmp_path / "chroma")
    reopened = None
    try:
        parsed = parse_document("doc", "note.txt", b"Vector persistence evidence lives here.")
        chunk = chunk_document(parsed, "fixed", {"chunk_size": 128, "overlap": 0})
        first.upsert_document("doc", chunk, [[1.0, 0.0]], "test-model", 2)
        first.close()
        reopened = ChromaVectorStore(tmp_path / "chroma")
        rows = reopened.search([1.0, 0.0], 1, ["doc"])
        assert rows[0]["id"] == chunk[0].chunk_id
        assert rows[0]["text"] == chunk[0].text
        total, page = reopened.list_chunks(limit=10)
        assert total == 1 and page[0]["id"] == chunk[0].chunk_id
        assert page[0]["embedding_available"] is True and page[0]["embedding_dimension"] == 2
        with pytest.raises(VectorStoreError, match="dimension mismatch"):
            reopened.upsert_document("doc", chunk, [[1.0, 0.0, 0.0]], "test-model", 3)
    finally:
        first.close()
        if reopened:
            reopened.close()


def test_chroma_search_surfaces_corrupt_vector_index_cause(tmp_path, monkeypatch):
    vectors = ChromaVectorStore(tmp_path / "chroma")
    try:
        vectors.collection.modify(metadata={"embedding_dimension": 2})

        def fail_query(**_):
            raise RuntimeError("Internal error: Error finding id")

        monkeypatch.setattr(vectors.collection, "query", fail_query)
        with pytest.raises(VectorStoreError, match="local vector index is inconsistent"):
            vectors.search([1.0, 0.0], 1)
    finally:
        vectors.close()


def test_chroma_search_merges_document_scoped_candidates_by_distance(tmp_path, monkeypatch):
    vectors = ChromaVectorStore(tmp_path / "chroma")
    try:
        vectors.collection.modify(metadata={"embedding_dimension": 2})
        calls = []

        def query(*, where, **_):
            document_id = where["document_id"]["$in"][0]
            calls.append(document_id)
            distance = {"doc-a": 0.6, "doc-b": 0.2}[document_id]
            return {
                "ids": [[f"{document_id}-chunk"]],
                "documents": [[document_id]],
                "metadatas": [[{"document_id": document_id}]],
                "distances": [[distance]],
            }

        monkeypatch.setattr(vectors.collection, "query", query)
        results = vectors.search([1.0, 0.0], 1, ["doc-a", "doc-b"])
        assert calls == ["doc-a", "doc-b"]
        assert [row["id"] for row in results] == ["doc-b-chunk"]
    finally:
        vectors.close()


def test_traditional_rag_pipeline_uses_mock_provider_and_real_persistent_chroma(tmp_path):
    class DeterministicProvider:
        def embed(self, model, texts):
            assert model == "test-embed"
            return ([[0.1, 0.9] for _ in texts], 2)

        def generate(self, model, system, prompt, **_settings):
            import json

            evidence = json.loads(prompt)["evidence"]
            return f"The source states this fact. [{evidence[0]['chunk_id']}]"

    db = tmp_path / "metadata.db"
    store = DocumentStore(f"sqlite:///{db.as_posix()}", tmp_path / "data")
    vectors = ChromaVectorStore(tmp_path / "chroma")
    try:
        document_id = store.create(
            "evidence.txt",
            "txt",
            "text/plain",
            b"The project stores evidence in persistent local storage.",
        )
        parsed = parse_document(document_id, "evidence.txt", store.source_bytes(document_id))
        recommendation = recommend_chunking(parsed.profile)
        store.update(
            document_id,
            status="profiled",
            profile=parsed.profile.model_dump(mode="json"),
            recommendation=recommendation.model_dump(mode="json"),
            selection={"strategy_id": "fixed", "settings": {"chunk_size": 128, "overlap": 0}},
        )
        settings = Settings(
            generation_provider="ollama",
            embedding_provider="ollama",
            ollama_embedding_model="test-embed",
            ollama_generation_model="test-chat",
        )
        provider = DeterministicProvider()
        index_steps = []
        indexed = index_document(
            document_id,
            store,
            vectors,
            provider,
            settings,
            on_step=lambda node, state: index_steps.append((node, state)),
        )
        assert indexed["status"] == "ready" and indexed["chunk_count"] == 1
        assert [node for node, state in index_steps if state == "running"] == [
            "parser",
            "chunker",
            "embedder",
            "index",
        ]
        query_steps = []
        result = answer_query(
            "Where is evidence stored?",
            3,
            [document_id],
            vectors,
            provider,
            settings,
            store,
            on_step=lambda node, state: query_steps.append((node, state)),
        )
        assert not result["abstained"]
        assert result["citations"][0]["chunk_id"] == store.get(document_id)["chunks"][0]["chunk_id"]
        assert [node for node, state in query_steps if state == "running"] == [
            "retriever",
            "context",
            "answer",
        ]

        class FailingGenerator:
            def generate(self, *_args, **_kwargs):
                raise ProviderError("Test provider is unavailable.")

        failed = answer_query(
            "Where is evidence stored?",
            3,
            [document_id],
            vectors,
            provider,
            settings,
            store,
            generation_provider=FailingGenerator(),
            generation_provider_id="ollama",
        )
        assert failed["status"] == "generation_failed"
        assert failed["retrieved_chunks"] and failed["generation_error"] == "Test provider is unavailable."

        class FailingEmbedder:
            def embed(self, *_args):
                raise ProviderError("replacement embedding failed")

        store.update(
            document_id,
            selection={"strategy_id": "fixed", "settings": {"chunk_size": 128, "overlap": 0}},
        )
        with pytest.raises(PipelineError, match="replacement embedding failed"):
            index_document(document_id, store, vectors, FailingEmbedder(), settings)
        after_failed_reindex = store.get(document_id)
        assert after_failed_reindex["status"] == "ready"
        assert after_failed_reindex["error"] == "replacement embedding failed"
        assert vectors.get_document(document_id)
    finally:
        vectors.close()
        store.close()


def test_sqlite_metadata_and_upload_survive_reopening_store(tmp_path):
    db_path = tmp_path / "reopen.db"
    root = tmp_path / "app-data"
    first = DocumentStore(f"sqlite:///{db_path.as_posix()}", root)
    reopened = None
    try:
        document_id = first.create("saved.txt", "txt", "text/plain", b"Persisted source.")
        first.update(document_id, status="profiled", error=None)
        reopened = DocumentStore(f"sqlite:///{db_path.as_posix()}", root)
        row = reopened.get(document_id)
        assert row and row["status"] == "profiled"
        assert reopened.source_bytes(document_id) == b"Persisted source."
        assert reopened.storage_summary() == {"document_count": 1, "source_files_present": 1}
    finally:
        first.close()
        if reopened:
            reopened.close()


def test_failed_initial_index_never_marks_document_ready(tmp_path):
    class FailingEmbedder:
        def embed(self, *_args):
            raise ProviderError("embedding provider unavailable")

    store = DocumentStore(
        f"sqlite:///{(tmp_path / 'failed.db').as_posix()}", tmp_path / "source-data"
    )
    vectors = ChromaVectorStore(tmp_path / "failed-chroma")
    try:
        document_id = store.create("failure.txt", "txt", "text/plain", b"This remains unindexed.")
        store.update(
            document_id,
            status="profiled",
            selection={"strategy_id": "fixed", "settings": {"chunk_size": 128, "overlap": 0}},
        )
        settings = Settings(
            generation_provider="ollama",
            embedding_provider="ollama",
            ollama_embedding_model="test-embed",
            ollama_generation_model="test-chat",
        )
        with pytest.raises(PipelineError, match="embedding provider unavailable"):
            index_document(document_id, store, vectors, FailingEmbedder(), settings)
        assert store.get(document_id)["status"] == "failed"
        assert store.get(document_id)["error"] == "embedding provider unavailable"
    finally:
        vectors.close()
        store.close()


def test_execution_tracker_reports_real_node_transitions_and_failure():
    tracker = ExecutionTracker()
    tracker.start("run-1", ["source", "parser", "chunker"], ["source"])
    tracker.node("run-1", "parser", "running")
    tracker.fail("run-1", "provider failed")
    progress = tracker.get("run-1")
    assert progress == {
        "execution_id": "run-1",
        "status": "failed",
        "node_statuses": {"source": "completed", "parser": "failed", "chunker": "skipped"},
        "error": "provider failed",
    }


def test_registered_traditional_graph_is_validated_against_configured_adapters():
    graph = TEMPLATES_BY_ID["traditional-rag"].graph
    settings = Settings(
        generation_provider="ollama",
        embedding_provider="ollama",
        ollama_embedding_model="nomic-embed-text",
        ollama_generation_model="llama3.2",
    )
    report = validate_traditional_graph(graph, settings)
    assert report.valid and report.executable
    invalid = graph.model_copy(update={"connections": graph.connections[:-1]})
    assert (
        validate_traditional_graph(invalid, settings).issues[0].code == "unsupported_graph_topology"
    )
    bad_chunk_config = graph.model_copy(deep=True)
    chunk_node = next(node for node in bad_chunk_config.nodes if node.node_type_id == "chunk.split")
    chunk_node.config = {"strategy": "fixed", "chunk_size": 64, "overlap": 64}
    bad_report = validate_traditional_graph(bad_chunk_config, settings)
    assert not bad_report.valid
    assert any(issue.code == "invalid_chunking_config" for issue in bad_report.issues)


def test_validate_custom_user_composed_graph_with_custom_node_ids():
    template_graph = TEMPLATES_BY_ID["traditional-rag"].graph
    assert template_graph is not None

    id_mapping = {n.id: f"custom-node-{n.id}-uuid" for n in template_graph.nodes}
    custom_nodes = [
        n.model_copy(update={"id": id_mapping[n.id]})
        for n in template_graph.nodes
    ]
    custom_connections = [
        c.model_copy(update={
            "source_node_id": id_mapping[c.source_node_id],
            "target_node_id": id_mapping[c.target_node_id],
        })
        for c in template_graph.connections
    ]
    custom_graph = template_graph.model_copy(
        update={
            "id": "user-custom-workflow",
            "name": "My Custom RAG",
            "architecture_id": "custom-rag",
            "template_id": None,
            "nodes": custom_nodes,
            "connections": custom_connections,
        }
    )
    settings = Settings(
        generation_provider="ollama",
        embedding_provider="ollama",
        ollama_embedding_model="nomic-embed-text",
        ollama_generation_model="llama3.2",
    )
    report = validate_traditional_graph(custom_graph, settings)
    assert report.valid and report.executable, f"Custom graph failed validation: {report.issues}"


def test_groq_credential_endpoint_allows_local_and_rejects_remote_origins():
    assert _is_local_origin("http://127.0.0.1:5174")
    assert _is_local_origin("http://localhost:5173")
    assert not _is_local_origin("https://attacker.example")
    assert not _is_local_origin("https://127.0.0.1:5174")


def test_traditional_validation_request_accepts_selected_generation_provider():
    graph = TEMPLATES_BY_ID["traditional-rag"].graph
    request = WorkflowValidationRequest(
        graph=graph,
        generation_provider="groq",
        generation_model="openai/gpt-oss-20b",
        require_executable=False,
    )
    report = validate_workflow(request)
    assert {"valid", "executable", "issues", "topological_order"}.issubset(report)


def test_document_upload_index_and_query_api_with_mock_models_and_real_chroma(tmp_path):
    import json

    class DeterministicProvider:
        def embed(self, model, texts):
            return ([[0.2, 0.8] for _ in texts], 2)

        def generate(self, model, system, prompt, **_settings):
            if prompt == "OK":
                return "OK"
            evidence = json.loads(prompt)["evidence"]
            return f"The uploaded document contains this fact. [{evidence[0]['chunk_id']}]"

    settings = Settings(
        generation_provider="ollama",
        embedding_provider="ollama",
        database_url=f"sqlite:///{(tmp_path / 'api.db').as_posix()}",
        data_path=tmp_path / "data",
        chroma_path=tmp_path / "chroma",
        ollama_embedding_model="test-embed",
        ollama_generation_model="test-chat",
        hf_token="hf-test-token",
    )
    store = DocumentStore(settings.database_url, settings.data_path)
    vectors = ChromaVectorStore(settings.chroma_path)
    provider = DeterministicProvider()
    app.dependency_overrides.update(
        {
            get_settings: lambda: settings,
            get_document_store: lambda: store,
            get_vector_store: lambda: vectors,
            get_ollama: lambda: provider,
            get_huggingface: lambda: provider,
        }
    )
    try:
        client = TestClient(app)
        uploaded = client.post(
            "/api/v1/documents",
            files={
                "file": (
                    "source.txt",
                    b"Visual RAG stores vectors in persistent Chroma.",
                    "text/plain",
                )
            },
        )
        assert uploaded.status_code == 201
        doc = uploaded.json()
        assert doc["status"] == "profiled"
        recommendation = doc["recommendation"]
        choice = client.post(
            f"/api/v1/documents/{doc['document_id']}/chunking-selection",
            json={
                "selection": {
                    "document_id": doc["document_id"],
                    "strategy_id": recommendation["recommended_strategy_id"],
                    "choice_mode": "accept_recommendation",
                    "source_recommendation_id": recommendation["recommendation_id"],
                    "settings": recommendation["proposed_settings"],
                }
            },
        )
        assert choice.status_code == 200
        graph = TEMPLATES_BY_ID["traditional-rag"].graph.model_dump(mode="json")
        validation = client.post(
            "/api/v1/workflows/validate",
            json={
                "graph": graph,
                "generation_provider": "ollama",
                "generation_model": "test-chat",
                "require_executable": False,
            },
        )
        assert validation.status_code == 200, validation.text
        indexed = client.post(
            f"/api/v1/documents/{doc['document_id']}/ingest", json={"graph": graph}
        )
        assert indexed.status_code == 200, indexed.text
        assert indexed.json()["status"] == "ready"
        query = client.post(
            "/api/v1/query",
            json={
                "question": "Where are vectors stored?",
                "graph": graph,
                "document_ids": [doc["document_id"]],
            },
        )
        assert query.status_code == 200, query.text
        assert query.json()["abstained"] is False
        assert query.json()["citations"][0]["document_id"] == doc["document_id"]
        assert query.json()["execution_id"]
        assert query.json()["retrieved_chunks"][0]["text"]
        chunks = client.get(f"/api/v1/chunks?document_id={doc['document_id']}&limit=10")
        assert chunks.status_code == 200 and chunks.json()["total"] == 1
        assert chunks.json()["chunks"][0]["embedding_available"] is True
        storage = client.get("/api/v1/storage/summary")
        assert storage.status_code == 200
        assert storage.json()["metadata_document_count"] == 1
        assert storage.json()["source_files_present"] == 1
        assert storage.json()["vector_count"] == 1
        assert str(tmp_path) not in storage.text
        preview = client.get(f"/api/v1/documents/{doc['document_id']}/preview")
        assert preview.status_code == 200 and "persistent Chroma" in preview.json()["text"]
        repeated = client.post(
            f"/api/v1/documents/{doc['document_id']}/ingest", json={"graph": graph}
        )
        assert repeated.status_code == 200
        assert client.get("/api/v1/storage/summary").json()["vector_count"] == 1
        blocked_hf_query = client.post(
            "/api/v1/query",
            json={
                "question": "Where are vectors stored?",
                "graph": graph,
                "document_ids": [doc["document_id"]],
                "generation_provider": "huggingface",
                "generation_model": "org/model",
            },
        )
        assert blocked_hf_query.status_code == 200
        assert blocked_hf_query.json()["status"] == "authorization_required"
        assert blocked_hf_query.json()["retrieved_chunks"]
        assert blocked_hf_query.json()["provider_output"] is None
        hf_query = client.post(
            "/api/v1/query",
            json={
                "question": "Where are vectors stored?",
                "graph": graph,
                "document_ids": [doc["document_id"]],
                "generation_provider": "huggingface",
                "generation_model": "org/model",
                "allow_external_processing": True,
            },
        )
        assert hf_query.status_code == 200, hf_query.text
        assert hf_query.json()["citations"][0]["document_id"] == doc["document_id"]
        configuration = client.get("/api/v1/providers/config").json()
        assert configuration["huggingface_token_present"] is True
        assert "hf_token" not in configuration and "hf-test-token" not in str(configuration)
        tested = client.post(
            "/api/v1/providers/test", json={"provider": "huggingface", "model": "org/model"}
        )
        assert tested.status_code == 200 and tested.json()["ready"] is True
    finally:
        client.close()
        vectors.close()
        store.close()
        app.dependency_overrides.clear()


def test_traditional_rag_pdf_end_to_end_verification_flow(tmp_path):
    # 1. Generate real PDF with selectable text
    pdf_doc = pymupdf.open()
    pdf_page = pdf_doc.new_page()
    pdf_page.insert_text(
        (50, 72),
        "Visual RAG Studio Acceptance Verification\n\n"
        "Traditional RAG uses SQLite for document metadata and Chroma for dense vector retrieval.\n"
        "Citations must match the exact chunk identifier and preserve page provenance.\n"
        "Generation errors preserve retrieved evidence without silent fallbacks.",
    )
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    class MockPdfProvider:
        def embed(self, model, texts):
            assert model == "test-embed"
            return ([[0.3, 0.7] for _ in texts], 2)

        def generate(self, model, system, prompt, **_settings):
            if prompt == "OK":
                return "OK"
            evidence = json.loads(prompt)["evidence"]
            return f"According to the source, SQLite and Chroma are used. [{evidence[0]['chunk_id']}]"

    settings = Settings(
        generation_provider="ollama",
        embedding_provider="ollama",
        database_url=f"sqlite:///{(tmp_path / 'pdf_verify.db').as_posix()}",
        data_path=tmp_path / "data",
        chroma_path=tmp_path / "chroma",
        ollama_embedding_model="test-embed",
        ollama_generation_model="test-chat",
    )
    store = DocumentStore(settings.database_url, settings.data_path)
    vectors = ChromaVectorStore(settings.chroma_path)
    provider = MockPdfProvider()
    app.dependency_overrides.update(
        {
            get_settings: lambda: settings,
            get_document_store: lambda: store,
            get_vector_store: lambda: vectors,
            get_ollama: lambda: provider,
        }
    )

    try:
        client = TestClient(app)

        # 1. Upload supported PDF
        uploaded = client.post(
            "/api/v1/documents",
            files={"file": ("verification.pdf", pdf_bytes, "application/pdf")},
        )
        assert uploaded.status_code == 201, uploaded.text
        doc = uploaded.json()
        doc_id = doc["document_id"]
        assert doc["file_type"] == "pdf"
        assert doc["status"] == "profiled"

        # 2. Parse and profile it
        profile_res = client.get(f"/api/v1/documents/{doc_id}/profile")
        assert profile_res.status_code == 200
        profile = profile_res.json()
        assert profile["extraction_status"] == "succeeded"
        assert profile["meaningful_text_extracted"]["value"] is True
        assert profile["structure"]["page_count"]["value"] == 1

        rec_res = client.post(f"/api/v1/documents/{doc_id}/recommendation")
        assert rec_res.status_code == 200
        recommendation = rec_res.json()
        assert recommendation["recommended_strategy_id"] in {
            "recursive",
            "fixed",
            "sentence",
            "structure_aware",
        }

        # 3. Select or override chunking strategy (override to sentence strategy)
        choice = client.post(
            f"/api/v1/documents/{doc_id}/chunking-selection",
            json={
                "selection": {
                    "document_id": doc_id,
                    "strategy_id": "sentence",
                    "choice_mode": "explicit_override",
                    "settings": {"chunk_size": 150, "overlap": 20},
                }
            },
        )
        assert choice.status_code == 200

        graph = TEMPLATES_BY_ID["traditional-rag"].graph.model_dump(mode="json")

        # 4. Index chunks and embeddings
        indexed = client.post(f"/api/v1/documents/{doc_id}/ingest", json={"graph": graph})
        assert indexed.status_code == 200, indexed.text
        assert indexed.json()["status"] == "ready"
        assert indexed.json()["chunk_count"] >= 1

        # 5. Confirm document and chunk records in SQLite/Chroma as appropriate
        doc_record = client.get(f"/api/v1/documents/{doc_id}").json()
        assert doc_record["status"] == "ready"
        assert doc_record["chunk_count"] >= 1
        assert len(store.get(doc_id)["chunks"]) >= 1

        chunk_list = client.get(f"/api/v1/documents/{doc_id}/chunks").json()
        assert len(chunk_list["chunks"]) >= 1
        assert chunk_list["chunks"][0]["page_number"] == 1
        assert (
            "SQLite" in chunk_list["chunks"][0]["text"]
            or "Chroma" in chunk_list["chunks"][0]["text"]
        )

        storage = client.get("/api/v1/storage/summary").json()
        assert storage["metadata_document_count"] == 1
        assert storage["vector_count"] >= 1

        # 6. Restart or reinitialize storage
        vectors.close()
        store.close()

        store = DocumentStore(settings.database_url, settings.data_path)
        vectors = ChromaVectorStore(settings.chroma_path)
        app.dependency_overrides[get_document_store] = lambda: store
        app.dependency_overrides[get_vector_store] = lambda: vectors

        # 7. Confirm persisted records remain accessible after restart
        doc_reopened = client.get(f"/api/v1/documents/{doc_id}").json()
        assert doc_reopened["status"] == "ready"
        assert doc_reopened["chunk_count"] == doc_record["chunk_count"]
        assert len(store.get(doc_id)["chunks"]) == len(store.get(doc_id)["chunks"])

        storage_reopened = client.get("/api/v1/storage/summary").json()
        assert storage_reopened["vector_count"] == storage["vector_count"]
        assert storage_reopened["metadata_document_count"] == 1

        preview = client.get(f"/api/v1/documents/{doc_id}/preview").json()
        assert "Visual RAG Studio Acceptance Verification" in preview["text"]

        # 8. Run retrieval
        query_res = client.post(
            "/api/v1/query",
            json={
                "question": "What does Traditional RAG use?",
                "graph": graph,
                "document_ids": [doc_id],
            },
        )
        assert query_res.status_code == 200, query_res.text
        query_data = query_res.json()
        assert query_data["abstained"] is False
        assert query_data["status"] == "completed"

        # 9. Inspect actual retrieved chunks and source metadata
        retrieved = query_data["retrieved_chunks"]
        assert len(retrieved) >= 1
        assert retrieved[0]["document_id"] == doc_id
        assert retrieved[0]["document_name"] == "verification.pdf"
        assert retrieved[0]["page_number"] == 1
        assert retrieved[0]["text"]
        assert retrieved[0]["distance"] is not None

        # 10. Verify citation references against retrieved chunks
        citations = query_data["citations"]
        assert len(citations) == 1
        assert citations[0]["chunk_id"] == retrieved[0]["chunk_id"]
        assert citations[0]["page_number"] == 1
        assert citations[0]["document_id"] == doc_id

        # 11. Exercise empty-retrieval abstention
        empty_vector_store = ChromaVectorStore(tmp_path / "empty_chroma")
        app.dependency_overrides[get_vector_store] = lambda: empty_vector_store
        try:
            abstain_res = client.post(
                "/api/v1/query",
                json={
                    "question": "What is quantum gravity?",
                    "graph": graph,
                    "document_ids": [doc_id],
                },
            )
            assert abstain_res.status_code == 200
            abstain_data = abstain_res.json()
            assert abstain_data["abstained"] is True
            assert abstain_data["status"] == "abstained"
            assert abstain_data["citations"] == []
            assert "could not find enough retrieved evidence" in abstain_data["answer"]
        finally:
            empty_vector_store.close()
            app.dependency_overrides[get_vector_store] = lambda: vectors

        # 12. Exercise generation-provider errors
        class FailingGenerator:
            def embed(self, model, texts):
                return provider.embed(model, texts)

            def generate(self, *_args, **_kwargs):
                raise ProviderError("Upstream model service disconnected.")

        failing_provider = FailingGenerator()
        app.dependency_overrides[get_ollama] = lambda: failing_provider
        failing_query = client.post(
            "/api/v1/query",
            json={
                "question": "What does Traditional RAG use?",
                "graph": graph,
                "document_ids": [doc_id],
            },
        )
        assert failing_query.status_code == 200
        failing_data = failing_query.json()
        assert failing_data["status"] == "generation_failed"
        assert failing_data["abstained"] is True
        assert failing_data["generation_error"] == "Upstream model service disconnected."
        # Evidence is preserved!
        assert len(failing_data["retrieved_chunks"]) >= 1
        assert failing_data["retrieved_chunks"][0]["document_id"] == doc_id
        assert failing_data["citations"] == []
    finally:
        client.close()
        vectors.close()
        store.close()
        app.dependency_overrides.clear()


def test_provider_consent_privacy_and_credential_sanitization(tmp_path):
    sent_prompts = []

    class MockExternalProvider:
        def embed(self, model, texts):
            return ([[0.5, 0.5] for _ in texts], 2)

        def generate(self, model, system, prompt, **_settings):
            sent_prompts.append(prompt)
            if prompt == "OK":
                return "OK"
            evidence = json.loads(prompt)["evidence"]
            return f"External answer [{evidence[0]['chunk_id']}]"

    settings = Settings(
        generation_provider="groq",
        embedding_provider="ollama",
        database_url=f"sqlite:///{(tmp_path / 'privacy.db').as_posix()}",
        data_path=tmp_path / "data",
        chroma_path=tmp_path / "chroma",
        ollama_embedding_model="test-embed",
        groq_api_key="gsk-super-secret-token-12345",
        hf_token="hf-super-secret-token-67890",
    )
    store = DocumentStore(settings.database_url, settings.data_path)
    vectors = ChromaVectorStore(settings.chroma_path)
    provider = MockExternalProvider()

    app.dependency_overrides.update(
        {
            get_settings: lambda: settings,
            get_document_store: lambda: store,
            get_vector_store: lambda: vectors,
            get_ollama: lambda: provider,
            get_groq: lambda: provider,
            get_huggingface: lambda: provider,
        }
    )

    try:
        client = TestClient(app)

        # Upload & index document
        uploaded = client.post(
            "/api/v1/documents",
            files={
                "file": (
                    "confidential.txt",
                    b"Confidential business strategy document.",
                    "text/plain",
                )
            },
        ).json()
        doc_id = uploaded["document_id"]
        rec = uploaded["recommendation"]
        client.post(
            f"/api/v1/documents/{doc_id}/chunking-selection",
            json={
                "selection": {
                    "document_id": doc_id,
                    "strategy_id": rec["recommended_strategy_id"],
                    "choice_mode": "accept_recommendation",
                    "settings": rec["proposed_settings"],
                }
            },
        )
        graph = TEMPLATES_BY_ID["traditional-rag"].graph.model_dump(mode="json")
        client.post(f"/api/v1/documents/{doc_id}/ingest", json={"graph": graph})

        # 1. External generation blocked without consent
        sent_prompts.clear()
        blocked_res = client.post(
            "/api/v1/query",
            json={
                "question": "What is the strategy?",
                "graph": graph,
                "document_ids": [doc_id],
                "generation_provider": "groq",
                "generation_model": "openai/gpt-oss-20b",
                "allow_external_processing": False,
            },
        )
        assert blocked_res.status_code == 200
        blocked_data = blocked_res.json()
        assert blocked_data["status"] == "authorization_required"
        assert blocked_data["abstained"] is True
        assert blocked_data["provider_output"] is None
        assert "External generation is blocked" in blocked_data["answer"]
        # Crucial check: external provider received ZERO prompts with document content!
        assert len(sent_prompts) == 0

        # 2. When consent is given, query succeeds and prompt is sent
        allowed_res = client.post(
            "/api/v1/query",
            json={
                "question": "What is the strategy?",
                "graph": graph,
                "document_ids": [doc_id],
                "generation_provider": "groq",
                "generation_model": "openai/gpt-oss-20b",
                "allow_external_processing": True,
            },
        )
        assert allowed_res.status_code == 200
        allowed_data = allowed_res.json()
        assert allowed_data["status"] == "completed"
        assert len(sent_prompts) == 1
        assert "Confidential business strategy" in sent_prompts[0]

        # 3. Subsequent query without explicit consent is blocked (consent does NOT persist across queries)
        subsequent_res = client.post(
            "/api/v1/query",
            json={
                "question": "What is the strategy?",
                "graph": graph,
                "document_ids": [doc_id],
                "generation_provider": "groq",
                "generation_model": "openai/gpt-oss-20b",
                # allow_external_processing omitted -> defaults to False
            },
        )
        assert subsequent_res.status_code == 200
        assert subsequent_res.json()["status"] == "authorization_required"
        assert len(sent_prompts) == 1  # No second prompt sent to provider

        # 4. Token & secret privacy check: tokens never appear in API responses or config
        cfg = client.get("/api/v1/providers/config").json()
        assert cfg["groq_api_key_present"] is True
        assert cfg["huggingface_token_present"] is True
        assert "gsk-super-secret-token-12345" not in str(cfg)
        assert "hf-super-secret-token-67890" not in str(cfg)

        # 5. Provider test prompt is separate from grounded queries
        test_prompts = []

        class TestProbeProvider:
            def generate(self, model, system, prompt, **_kwargs):
                test_prompts.append((system, prompt))
                return "OK"

        probe = TestProbeProvider()
        app.dependency_overrides[get_groq] = lambda: probe
        app.dependency_overrides[get_huggingface] = lambda: probe
        test_res = client.post(
            "/api/v1/providers/test", json={"provider": "groq", "model": "openai/gpt-oss-20b"}
        )
        assert test_res.status_code == 200
        assert len(test_prompts) == 1
        # The probe prompt is generic ("OK"), contains zero document evidence
        assert "Confidential" not in test_prompts[0][1]
        assert test_prompts[0][1] == "OK"
        assert test_prompts[0][0] == "Reply with exactly OK."
    finally:
        client.close()
        vectors.close()
        store.close()
        app.dependency_overrides.clear()


def test_batch_document_upload_and_ingestion(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        data_path=str(tmp_path / "data"),
        chroma_path=str(tmp_path / "chroma"),
        embedding_provider="sentence_transformers",
        generation_provider="extractive",
    )
    store = DocumentStore(settings.database_url, settings.data_path)
    vectors = ChromaVectorStore(settings.chroma_path)
    embedder = FastLocalEmbeddingProvider()

    app.dependency_overrides.update(
        {
            get_settings: lambda: settings,
            get_document_store: lambda: store,
            get_vector_store: lambda: vectors,
            get_sentence_transformers: lambda: embedder,
            get_ollama: lambda: embedder,
        }
    )

    try:
        client = TestClient(app)

        # 1. Batch upload multiple documents
        files = [
            ("files", ("doc1.txt", b"First document about astronomy and black holes in deep space.", "text/plain")),
            ("files", ("doc2.txt", b"Second document about biology and cellular mitochondria powerhouse.", "text/plain")),
        ]
        batch_res = client.post("/api/v1/documents/batch", files=files)
        assert batch_res.status_code == 201
        batch_data = batch_res.json()
        assert len(batch_data) == 2
        doc_ids = [d["document_id"] for d in batch_data]

        # 2. Batch ingestion
        graph = TEMPLATES_BY_ID["traditional-rag"].graph.model_dump(mode="json")
        ingest_res = client.post(
            "/api/v1/documents/ingest-batch",
            json={
                "graph": graph,
                "document_ids": doc_ids,
                "generation_provider": "extractive",
            },
        )
        assert ingest_res.status_code == 200
        ingest_data = ingest_res.json()
        assert ingest_data["indexed_count"] == 2

        # 3. Verify both documents are ready
        for doc_id in doc_ids:
            doc_status = client.get(f"/api/v1/documents/{doc_id}").json()
            assert doc_status["status"] == "ready"
            assert doc_status["chunk_count"] > 0

        # 4. Query across all documents
        query_res = client.post(
            "/api/v1/query",
            json={
                "question": "What is mitochondria?",
                "graph": graph,
                "document_ids": doc_ids,
                "generation_provider": "extractive",
                "top_k": 3,
            },
        )
        assert query_res.status_code == 200
        query_data = query_res.json()
        assert query_data["status"] == "completed"
        assert len(query_data["retrieved_chunks"]) > 0
        assert any("mitochondria" in c["text"].lower() for c in query_data["retrieved_chunks"])
    finally:
        client.close()
        vectors.close()
        store.close()
        app.dependency_overrides.clear()


def test_dynamic_models_routes_and_ollama_thinking_suppression():
    from app.rag.providers import OllamaProvider
    import httpx

    # Test dynamic models fallback
    client = TestClient(app)
    groq_models = client.get("/api/v1/providers/groq/models").json()
    assert isinstance(groq_models, list)
    assert "llama-3.1-8b-instant" in groq_models

    openai_models = client.get("/api/v1/providers/openai/models").json()
    assert isinstance(openai_models, list)
    assert "gpt-4o-mini" in openai_models

    # Test Ollama thinking mode disabled and think tags stripped
    captured_payloads = []

    def mock_handler(request: httpx.Request):
        import json
        payload = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(payload)
        # Mock thinking tags in response
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "<think>Wait, let me deliberate on this question...</think>The BLEU score is 28.4."
                }
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    ollama = OllamaProvider(base_url="http://mock-ollama:11434", client=mock_client)

    result = ollama.generate("qwen3:1.7b", "System instructions", "User question")
    assert len(captured_payloads) == 1
    # Verify think: False is sent in the payload
    assert captured_payloads[0].get("think") is False
    # Verify <think> tags are stripped from the final output
    assert "<think>" not in result
    assert result.strip() == "The BLEU score is 28.4."

