"""Story 1.4: credentials must never live in workflow graphs (AD-9)."""

from fastapi.testclient import TestClient

from app.contracts import WorkflowGraph
from app.main import app
from app.rag.credentials import runtime_credentials
from app.rag.vector_store import VectorStoreError, get_vector_store_adapter
from app.registry import NODE_TYPES_BY_ID, TEMPLATES_BY_ID
from app.validation import secret_config_issues, validate_graph

SECRET = "pcsk-super-secret-value"


def _graph(**vector_config) -> WorkflowGraph:
    graph = TEMPLATES_BY_ID["traditional-rag"].graph
    assert graph is not None
    nodes = [
        node.model_copy(update={"config": {**node.config, **vector_config}})
        if node.node_type_id == "vector.index"
        else node
        for node in graph.nodes
    ]
    return graph.model_copy(update={"nodes": nodes})


def test_api_key_in_vector_node_is_rejected_with_actionable_issue():
    graph = _graph(backend="pinecone", api_key=SECRET)
    issues = secret_config_issues(graph)
    assert [i.code for i in issues] == ["secret_in_graph"]
    assert "PINECONE_API_KEY" in issues[0].message
    assert SECRET not in issues[0].message
    report = validate_graph(graph, NODE_TYPES_BY_ID)
    assert any(i.code == "secret_in_graph" for i in report.issues)
    assert report.valid is False


def test_url_with_embedded_credentials_is_rejected():
    issues = secret_config_issues(_graph(backend="mongodb", url="mongodb://user:pw@host:27017"))
    assert [i.code for i in issues] == ["secret_in_graph"]
    assert "pw" not in issues[0].message


def test_empty_key_and_host_only_url_are_accepted():
    assert secret_config_issues(_graph(api_key="", url="http://localhost:8000")) == []


def test_saving_a_graph_with_a_secret_is_refused_and_not_stored():
    client = TestClient(app)
    graph = _graph(backend="pinecone", api_key=SECRET).model_dump(mode="json")
    res = client.post("/api/v1/workflows", json={"name": "leaky", "graph": graph})
    assert res.status_code == 422
    assert res.json()["detail"]["code"] == "secret_in_graph"
    assert SECRET not in res.text
    assert SECRET not in client.get("/api/v1/workflows").text


def test_query_refuses_graph_with_secret_without_echoing_it():
    client = TestClient(app)
    graph = _graph(backend="pinecone", api_key=SECRET).model_dump(mode="json")
    res = client.post("/api/v1/query", json={"question": "hi", "graph": graph})
    assert res.status_code in {422}
    assert SECRET not in res.text


def test_pinecone_key_resolves_from_runtime_credentials_not_graph(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    try:
        get_vector_store_adapter(backend="pinecone", api_key=None)
    except VectorStoreError as exc:
        assert "PINECONE_API_KEY" in str(exc)
    runtime_credentials.set_key("pinecone", SECRET)
    try:
        try:
            get_vector_store_adapter(backend="pinecone", api_key=runtime_credentials.get_key("pinecone"))
        except VectorStoreError as exc:
            # Key was found; the failure is the (unimplemented) connection, not a missing key.
            assert "requires" not in str(exc)
            assert SECRET not in str(exc)
    finally:
        runtime_credentials.clear_key("pinecone")
