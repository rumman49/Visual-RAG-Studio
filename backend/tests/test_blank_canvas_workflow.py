"""Tests for Blank Canvas workflow design and false blocker elimination."""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.contracts import WorkflowGraph, WorkflowValidationRequest
from app.main import app
from app.rag.graph import validate_traditional_graph
from app.registry import TEMPLATES_BY_ID


@pytest.fixture
def blank_canvas_rag_graph() -> WorkflowGraph:
    """Simulates a RAG workflow built from scratch on the blank canvas."""
    template_graph = TEMPLATES_BY_ID["traditional-rag"].graph
    assert template_graph is not None

    # Simulate blank canvas node creation where nodes get fresh IDs and provider_instance_id was omitted/null
    id_map = {node.id: f"blank_node_{node.id}_{idx}" for idx, node in enumerate(template_graph.nodes)}
    nodes = [
        node.model_copy(update={
            "id": id_map[node.id],
            "provider_instance_id": None,  # Nodes dropped on canvas historically had null provider_instance_id
        })
        for node in template_graph.nodes
    ]
    connections = [
        conn.model_copy(update={
            "source_node_id": id_map[conn.source_node_id],
            "target_node_id": id_map[conn.target_node_id],
        })
        for conn in template_graph.connections
    ]

    return WorkflowGraph(
        schema_version=1,
        id="draft-workflow",
        name="Untitled workflow",
        architecture_id="start-from-blank-canvas",
        template_id="start-from-blank-canvas",
        nodes=nodes,
        connections=connections,
    )


def test_blank_canvas_graph_validates_as_executable(blank_canvas_rag_graph: WorkflowGraph):
    """Ensure blank canvas graphs with complete RAG nodes validate with 0 errors and 0 blocking warnings."""
    settings = Settings(
        generation_provider="groq",
        embedding_provider="sentence_transformers",
        sentence_transformers_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    report = validate_traditional_graph(
        blank_canvas_rag_graph,
        settings,
        generation_provider="groq",
        generation_model="llama-3.3-70b-versatile",
        require_executable=True,
    )
    assert report.valid is True
    assert report.executable is True
    # Verify no false provider_not_selected warnings
    provider_warnings = [i for i in report.issues if i.code == "provider_not_selected"]
    assert not provider_warnings, f"Found false provider warnings: {provider_warnings}"


def test_blank_canvas_api_validation_endpoint(blank_canvas_rag_graph: WorkflowGraph):
    """Test /api/v1/workflows/validate endpoint handles blank canvas graphs without false blockers."""
    client = TestClient(app)
    req = WorkflowValidationRequest(
        graph=blank_canvas_rag_graph,
        generation_provider="extractive",
        generation_model="default",
        require_executable=False,
    )
    res = client.post("/api/v1/workflows/validate", json=req.model_dump(mode="json"))
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is True
    assert data["executable"] is True
    assert not any(i["code"] == "provider_not_selected" for i in data["issues"])


def test_blank_canvas_with_null_architecture_and_template_id(blank_canvas_rag_graph: WorkflowGraph):
    """Test that a workflow graph with architecture_id=None and template_id=None still validates cleanly."""
    graph_null_arch = blank_canvas_rag_graph.model_copy(
        update={"architecture_id": None, "template_id": None}
    )
    client = TestClient(app)
    req = WorkflowValidationRequest(
        graph=graph_null_arch,
        generation_provider="extractive",
        generation_model="default",
        require_executable=False,
    )
    res = client.post("/api/v1/workflows/validate", json=req.model_dump(mode="json"))
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is True
    assert data["executable"] is True
    assert not any(i["code"] == "provider_not_selected" for i in data["issues"])


def test_blank_canvas_partial_graph_does_not_emit_false_provider_errors():
    """When a user has only placed a few nodes on canvas, provider warnings should not be emitted."""
    client = TestClient(app)
    partial_graph = WorkflowGraph(
        schema_version=1,
        id="draft-workflow",
        name="Untitled workflow",
        architecture_id="start-from-blank-canvas",
        template_id="start-from-blank-canvas",
        nodes=[
            {
                "id": "node_1",
                "node_type_id": "document.source",
                "config": {},
                "provider_instance_id": None,
            },
            {
                "id": "node_2",
                "node_type_id": "embedding.create",
                "config": {"provider": "sentence_transformers", "model": "multilingual-e5-base", "dimension": 768},
                "provider_instance_id": None,
            },
        ],
        connections=[],
    )
    req = WorkflowValidationRequest(
        graph=partial_graph,
        generation_provider="groq",
        require_executable=False,
    )
    res = client.post("/api/v1/workflows/validate", json=req.model_dump(mode="json"))
    assert res.status_code == 200
    data = res.json()
    # While it won't be executable due to incomplete connections, it MUST NOT emit false provider_not_selected
    codes = [i["code"] for i in data["issues"]]
    assert "provider_not_selected" not in codes


def test_blank_canvas_ingest_and_query_e2e(tmp_path, blank_canvas_rag_graph: WorkflowGraph):
    """Verify that a document can be ingested and queried using a blank-canvas designed workflow."""
    from app.rag.dependencies import get_document_store, get_settings, get_vector_store
    from app.rag.storage import DocumentStore
    from app.rag.vector_store import ChromaVectorStore

    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'meta.db'}",
        data_path=str(tmp_path / "data"),
        chroma_path=str(tmp_path / "chroma"),
        generation_provider="extractive",
        embedding_provider="fast_local",
    )
    store = DocumentStore(settings.database_url, settings.data_path)
    vectors = ChromaVectorStore(settings.chroma_path)

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_document_store] = lambda: store
    app.dependency_overrides[get_vector_store] = lambda: vectors

    try:
        client = TestClient(app)
        # 1. Upload document
        upload = client.post(
            "/api/v1/documents",
            files={"file": ("aligarh_act.txt", b"The Aligarh Muslim University Act of 1920 establishes the University.", "text/plain")},
        )
        assert upload.status_code == 201
        doc_id = upload.json()["document_id"]

        # 2. Ingest document with blank-canvas graph
        ingest_payload = {
            "graph": blank_canvas_rag_graph.model_dump(mode="json"),
            "generation_provider": "extractive",
        }
        ingest = client.post(f"/api/v1/documents/{doc_id}/ingest", json=ingest_payload)
        assert ingest.status_code == 200, ingest.text
        assert ingest.json()["status"] == "ready"

        # 3. Query document with blank-canvas graph
        query_payload = {
            "question": "What does the 1920 Act establish?",
            "graph": blank_canvas_rag_graph.model_dump(mode="json"),
            "document_ids": [doc_id],
            "generation_provider": "extractive",
        }
        query = client.post("/api/v1/query", json=query_payload)
        assert query.status_code == 200, query.text
        res_data = query.json()
        assert res_data["abstained"] is False
        assert "University" in res_data["answer"] or "establishes" in res_data["answer"]
        assert len(res_data["citations"]) > 0
    finally:
        app.dependency_overrides.clear()
        vectors.close()
        store.close()

