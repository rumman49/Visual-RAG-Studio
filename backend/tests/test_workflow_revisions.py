"""Story 2.1: every save creates a revision; older ones are kept and addressable."""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.main import app
from app.rag.dependencies import get_document_store
from app.rag.storage import DocumentStore, WorkflowRevisionRow, WorkflowRow
from app.registry import TEMPLATES_BY_ID


@pytest.fixture
def store(tmp_path):
    s = DocumentStore(f"sqlite:///{(tmp_path / 'wf.db').as_posix()}", tmp_path / "data")
    yield s
    s.close()


@pytest.fixture
def client(store):
    app.dependency_overrides[get_document_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.pop(get_document_store, None)


def _graph(label: str) -> dict:
    graph = TEMPLATES_BY_ID["traditional-rag"].graph.model_dump(mode="json")
    graph["name"] = label
    return graph


def test_each_save_adds_a_numbered_revision_and_keeps_the_previous(client):
    first = client.post("/api/v1/workflows", json={"id": "wf1", "name": "A", "graph": _graph("v1")})
    assert first.status_code == 200 and first.json()["revision"] == 1
    second = client.post("/api/v1/workflows", json={"id": "wf1", "name": "B", "graph": _graph("v2")})
    assert second.json()["revision"] == 2
    assert second.json()["created_at"] == first.json()["created_at"]
    assert second.json()["updated_at"] >= first.json()["updated_at"]

    listing = client.get("/api/v1/workflows/wf1/revisions").json()
    assert [r["revision"] for r in listing] == [2, 1]
    assert all(r["created_at"] for r in listing)
    # updated_at reflects the latest revision's timestamp
    assert client.get("/api/v1/workflows/wf1").json()["updated_at"] == listing[0]["created_at"]


def test_open_by_number_returns_exact_graph_and_latest_is_default(client):
    for label in ("v1", "v2", "v3"):
        client.post("/api/v1/workflows", json={"id": "wf1", "name": label, "graph": _graph(label)})
    rev1 = client.get("/api/v1/workflows/wf1/revisions/1").json()
    assert rev1["graph"]["name"] == "v1" and rev1["name"] == "v1" and rev1["revision"] == 1
    latest = client.get("/api/v1/workflows/wf1").json()
    assert latest["revision"] == 3 and latest["graph"]["name"] == "v3"
    assert client.get("/api/v1/workflows").json()[0]["revision"] == 3


def test_unknown_workflow_or_revision_is_404(client):
    client.post("/api/v1/workflows", json={"id": "wf1", "name": "A", "graph": _graph("v1")})
    assert client.get("/api/v1/workflows/nope/revisions").status_code == 404
    res = client.get("/api/v1/workflows/wf1/revisions/9")
    assert res.status_code == 404
    assert res.json()["detail"]["code"] == "workflow_revision_not_found"


def test_workflow_saved_before_revisions_becomes_revision_one(tmp_path):
    url = f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}"
    first = DocumentStore(url, tmp_path / "data")
    saved_at = datetime(2026, 1, 2, tzinfo=UTC)
    with first.sessions.begin() as session:
        session.add(WorkflowRow(
            id="old", name="Legacy", graph={"marker": "legacy"}, created_at=saved_at, updated_at=saved_at
        ))
        session.execute(delete(WorkflowRevisionRow))
    first.close()

    reopened = DocumentStore(url, tmp_path / "data")
    try:
        assert reopened.list_workflow_revisions("old")[0]["revision"] == 1
        loaded = reopened.get_workflow("old")
        assert loaded["revision"] == 1 and loaded["graph"] == {"marker": "legacy"}
        # idempotent: reopening again adds nothing
        reopened.close()
        again = DocumentStore(url, tmp_path / "data")
        assert len(again.list_workflow_revisions("old")) == 1
        again.close()
    finally:
        reopened.close()


def test_deleting_a_workflow_removes_its_revisions(client, store):
    client.post("/api/v1/workflows", json={"id": "wf1", "name": "A", "graph": _graph("v1")})
    assert client.delete("/api/v1/workflows/wf1").status_code == 200
    assert store.list_workflow_revisions("wf1") is None
    with store.sessions() as session:
        assert session.query(WorkflowRevisionRow).count() == 0
