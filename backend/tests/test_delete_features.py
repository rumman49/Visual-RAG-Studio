from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.chunking import TextChunk
from app.rag.dependencies import get_document_store, get_vector_store
from app.rag.storage import DocumentStore
from app.rag.vector_store import ChromaVectorStore


def make_test_chunk(chunk_id: str, document_id: str, document_name: str, ordinal: int = 0, text: str = "sample text"):
    return TextChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        ordinal=ordinal,
        text=text,
        strategy_id="fixed",
        settings={"chunk_size": 100, "overlap": 10},
        page_number=1,
        section_path=(),
        source_start=0,
        source_end=len(text),
        parser_id="plain_text",
    )


@pytest.fixture
def client(tmp_path: Path):
    db_file = tmp_path / "test.db"
    tmp_path / "uploads"
    chroma_dir = tmp_path / "chroma"
    test_store = DocumentStore(f"sqlite:///{db_file}", tmp_path)
    test_vs = ChromaVectorStore(chroma_dir)

    app.dependency_overrides[get_document_store] = lambda: test_store
    app.dependency_overrides[get_vector_store] = lambda: test_vs

    with TestClient(app) as test_client:
        yield test_client, test_store, test_vs

    test_store.close()
    test_vs.close()
    app.dependency_overrides.clear()


def test_delete_single_document_and_source_file(client):
    test_client, store, vector_store = client

    # 1. Create a document
    doc_id = store.create(
        filename="test_manual.md",
        file_type="md",
        content_type="text/markdown",
        content=b"# Header\n\nThis is test content for deletion testing.",
    )
    source_file = store._source_path(doc_id)
    assert source_file.is_file()

    # 2. Add indexed chunks to vector store
    chunk = make_test_chunk(
        chunk_id=f"chunk-{doc_id}-0",
        document_id=doc_id,
        document_name="test_manual.md",
        ordinal=0,
        text="This is test content for deletion testing.",
    )
    vector_store.upsert_document(
        document_id=doc_id,
        chunks=[chunk],
        embeddings=[[0.1] * 384],
        model="test-embedder",
        dimension=384,
    )
    store.update(doc_id, status="ready", chunks=[asdict(chunk)])

    # Verify chunk is in vector store
    assert len(vector_store.get_document(doc_id)) == 1

    # 3. Call DELETE /api/v1/documents/{doc_id}
    response = test_client.delete(f"/api/v1/documents/{doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["deleted"] is True
    assert data["document_id"] == doc_id
    assert data["chunks_removed"] == 1

    # 4. Verify source file on disk is removed
    assert not source_file.is_file()

    # 5. Verify database row is gone
    get_res = test_client.get(f"/api/v1/documents/{doc_id}")
    assert get_res.status_code == 404

    # 6. Verify chunks in vector store are gone
    assert len(vector_store.get_document(doc_id)) == 0


def test_delete_document_chunks_only(client):
    test_client, store, vector_store = client

    doc_id = store.create(
        filename="preserve_source.txt",
        file_type="txt",
        content_type="text/plain",
        content=b"Keep this text file, but clear its indexed vectors.",
    )
    source_file = store._source_path(doc_id)
    assert source_file.is_file()

    chunk = make_test_chunk(
        chunk_id=f"chunk-{doc_id}-0",
        document_id=doc_id,
        document_name="preserve_source.txt",
        ordinal=0,
        text="Keep this text file, but clear its indexed vectors.",
    )
    vector_store.upsert_document(
        document_id=doc_id,
        chunks=[chunk],
        embeddings=[[0.2] * 384],
        model="test-embedder",
        dimension=384,
    )
    store.update(doc_id, status="ready", chunks=[asdict(chunk)])

    # Call DELETE /api/v1/documents/{doc_id}/chunks
    response = test_client.delete(f"/api/v1/documents/{doc_id}/chunks")
    assert response.status_code == 200
    data = response.json()
    assert data["cleared"] is True
    assert data["chunks_removed"] == 1
    assert data["document"]["status"] == "uploaded"
    assert data["document"]["chunk_count"] == 0

    # Source file must STILL exist
    assert source_file.is_file()

    # Vector store must have 0 chunks for this doc
    assert len(vector_store.get_document(doc_id)) == 0


def test_bulk_delete_all_documents(client):
    test_client, store, _vector_store = client

    store.create("doc1.txt", "txt", "text/plain", b"Doc 1 text")
    store.create("doc2.txt", "txt", "text/plain", b"Doc 2 text")

    # Call DELETE /api/v1/documents
    response = test_client.delete("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["deleted_count"] == 2

    # Verify document list is empty
    list_res = test_client.get("/api/v1/documents")
    assert list_res.status_code == 200
    assert list_res.json() == []


def test_delete_single_chunk(client):
    test_client, store, vector_store = client

    doc_id = store.create("multi_chunk.txt", "txt", "text/plain", b"Chunk 1 and Chunk 2")
    c1 = make_test_chunk(chunk_id="chunk-c1", document_id=doc_id, document_name="multi_chunk.txt", ordinal=0, text="Chunk 1")
    c2 = make_test_chunk(chunk_id="chunk-c2", document_id=doc_id, document_name="multi_chunk.txt", ordinal=1, text="Chunk 2")
    vector_store.upsert_document(
        document_id=doc_id,
        chunks=[c1, c2],
        embeddings=[[0.1] * 384, [0.2] * 384],
        model="test-model",
        dimension=384,
    )
    store.update(doc_id, status="ready", chunks=[asdict(c1), asdict(c2)])

    # Delete single chunk
    response = test_client.delete("/api/v1/chunks/chunk-c1")
    assert response.status_code == 200
    assert response.json()["deleted"] is True

    # Check remaining chunks
    doc_chunks = vector_store.get_document(doc_id)
    assert len(doc_chunks) == 1
    assert doc_chunks[0]["id"] == "chunk-c2"


def test_clear_all_chunks(client):
    test_client, store, vector_store = client

    doc_id = store.create("clear_all.txt", "txt", "text/plain", b"Some content")
    c1 = make_test_chunk(chunk_id="ca-1", document_id=doc_id, document_name="clear_all.txt", ordinal=0, text="content")
    vector_store.upsert_document(
        document_id=doc_id,
        chunks=[c1],
        embeddings=[[0.3] * 384],
        model="test-model",
        dimension=384,
    )
    store.update(doc_id, status="ready", chunks=[asdict(c1)])

    # Clear all chunks
    response = test_client.delete("/api/v1/chunks")
    assert response.status_code == 200
    assert response.json()["cleared"] is True

    # Vector store should have 0 chunks
    assert len(vector_store.get_document(doc_id)) == 0


def test_storage_cleanup_orphaned_files(client):
    test_client, store, _ = client

    # Create a legitimate document
    doc_id = store.create("legit.txt", "txt", "text/plain", b"Legitimate document")

    # Manually drop an orphaned file into upload_root
    orphan_file = store.upload_root / "orphan_123.txt"
    orphan_file.write_bytes(b"I am an orphaned file with no database record")

    assert orphan_file.is_file()

    # Trigger cleanup
    response = test_client.post("/api/v1/storage/cleanup")
    assert response.status_code == 200
    data = response.json()
    assert data["cleaned_count"] == 1
    assert "orphan_123.txt" in data["cleaned_files"]

    # Verify orphan was deleted but legitimate file remains
    assert not orphan_file.is_file()
    assert store._source_path(doc_id).is_file()


def test_delete_nonexistent_document_returns_404(client):
    test_client, _, _ = client
    response = test_client.delete("/api/v1/documents/non-existent-uuid")
    assert response.status_code == 404
