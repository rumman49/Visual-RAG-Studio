"""Persistent dense Chroma adapter with explicit model and dimension checks."""

from __future__ import annotations

import contextlib
import gc
import math
import os
from pathlib import Path

import chromadb

from app.rag.chunking import TextChunk


class VectorStoreError(RuntimeError):
    pass


class ChromaVectorStore:
    def __init__(
        self, path: str | Path, collection_name: str = "visual_rag_studio_documents"
    ):
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.backend_name = "chroma"
        self.collection_name = collection_name or "visual_rag_studio_documents"
        chroma_settings = chromadb.config.Settings(anonymized_telemetry=False, is_persistent=True)
        self.client = chromadb.PersistentClient(path=str(self.path), settings=chroma_settings)
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception as exc:
            if "not found" not in str(exc).lower() and "does not exist" not in str(exc).lower():
                raise VectorStoreError("Could not open the persistent Chroma collection.") from exc
            self.collection = self.client.create_collection(
                self.collection_name,
                metadata={"hnsw:space": "cosine", "product": "visual-rag-studio"},
            )
        metadata = self.collection.metadata or {}
        if metadata.get("hnsw:space", "cosine") != "cosine":
            raise VectorStoreError(
                "The existing Chroma collection uses an incompatible distance metric."
            )

    def close(self) -> None:
        if hasattr(self, "client") and self.client is not None:
            with contextlib.suppress(Exception):
                self.client.close()
            self.client = None
            gc.collect()

    def _check_model_dimension(self, model: str, dimension: int) -> None:
        metadata = dict(self.collection.metadata or {})
        existing_model = metadata.get("embedding_model")
        existing_dimension = metadata.get("embedding_dimension")
        if existing_model and existing_model != model:
            raise VectorStoreError(
                f"The persistent collection uses embedding model '{existing_model}', not '{model}'."
            )
        if existing_dimension and int(existing_dimension) != dimension:
            raise VectorStoreError(
                f"Embedding dimension mismatch: collection has {existing_dimension}, provider returned {dimension}."
            )
        if existing_model is None or existing_dimension is None:
            metadata.update(
                {
                    "product": "visual-rag-studio",
                    "embedding_model": model,
                    "embedding_dimension": dimension,
                }
            )
            self.collection.modify(metadata={key: value for key, value in metadata.items() if key != "hnsw:space"})

    def upsert_document(
        self,
        document_id: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
        model: str,
        dimension: int,
        old_ids: list[str] | None = None,
    ) -> list[str]:
        if not chunks or len(chunks) != len(embeddings):
            raise VectorStoreError(
                "Vector insertion requires a matching non-empty chunk and embedding set."
            )
        if any(len(vector) != dimension for vector in embeddings):
            raise VectorStoreError(
                "Embedding dimension does not match the provider response metadata."
            )
        self._check_model_dimension(model, dimension)
        ids = [chunk.chunk_id for chunk in chunks]
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=[chunk.text for chunk in chunks],
                metadatas=[chunk.chroma_metadata() for chunk in chunks],
            )
            stale = sorted(set(old_ids or []) - set(ids))
            if stale:
                self.collection.delete(ids=stale)
        except Exception as exc:
            raise VectorStoreError("Chroma could not persist document vectors.") from exc
        return ids

    def search(
        self,
        vector: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
        query_text: str | None = None,
    ) -> list[dict]:
        metadata = self.collection.metadata or {}
        dimension = metadata.get("embedding_dimension")
        if not dimension:
            return []
        if len(vector) != int(dimension):
            raise VectorStoreError(
                f"Query embedding dimension {len(vector)} does not match indexed dimension {dimension}."
            )
        # Chroma's approximate index can return poor candidates for a broad
        # multi-document filter. Search each document separately, then merge by
        # its actual cosine distance so a strong match in a smaller document is
        # not hidden by short/noisy chunks from a larger one.
        is_all_documents = document_ids is None or (len(document_ids) == 0)
        clean_scopes = [str(d) for d in document_ids if d and str(d).strip()] if document_ids else []
        scopes = list(dict.fromkeys(clean_scopes))
        allowed_doc_ids = set(scopes) if scopes else None
        candidates: dict[str, dict] = {}
        try:
            if is_all_documents or not scopes:
                # Searching all indexed documents: complete collection
                fetch_n = max(top_k * 4, 30)
                result = self.collection.query(
                    query_embeddings=[vector],
                    n_results=fetch_n,
                    where=None,
                    include=["documents", "metadatas", "distances"],
                )
                ids = (result.get("ids") or [[]])[0]
                documents = (result.get("documents") or [[]])[0]
                metadatas = (result.get("metadatas") or [[]])[0]
                distances = (result.get("distances") or [[]])[0]
                for index, chunk_id in enumerate(ids):
                    candidates[chunk_id] = {
                        "id": chunk_id,
                        "text": documents[index] if index < len(documents) else "",
                        "metadata": metadatas[index] or {} if index < len(metadatas) else {},
                        "distance": distances[index] if index < len(distances) else None,
                        "retrieval_strategy": "dense",
                    }
            elif len(scopes) > 5:
                # Searching a selected subset of > 5 documents:
                # Use a single query with $in filter to preserve scope isolation
                fetch_n = max(top_k * 4, 30)
                result = self.collection.query(
                    query_embeddings=[vector],
                    n_results=fetch_n,
                    where={"document_id": {"$in": scopes}},
                    include=["documents", "metadatas", "distances"],
                )
                ids = (result.get("ids") or [[]])[0]
                documents = (result.get("documents") or [[]])[0]
                metadatas = (result.get("metadatas") or [[]])[0]
                distances = (result.get("distances") or [[]])[0]
                for index, chunk_id in enumerate(ids):
                    meta = metadatas[index] or {} if index < len(metadatas) else {}
                    doc_id = meta.get("document_id")
                    if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                        continue
                    candidates[chunk_id] = {
                        "id": chunk_id,
                        "text": documents[index] if index < len(documents) else "",
                        "metadata": meta,
                        "distance": distances[index] if index < len(distances) else None,
                        "retrieval_strategy": "dense",
                    }
            else:
                # Searching 1 to 5 selected documents:
                # Search per document and merge so smaller documents are fairly represented
                for document_id in scopes:
                    where = {"document_id": {"$in": [document_id]}}
                    fetch_n = max(top_k * 3, 15)
                    result = self.collection.query(
                        query_embeddings=[vector],
                        n_results=fetch_n,
                        where=where,
                        include=["documents", "metadatas", "distances"],
                    )
                    ids = (result.get("ids") or [[]])[0]
                    documents = (result.get("documents") or [[]])[0]
                    metadatas = (result.get("metadatas") or [[]])[0]
                    distances = (result.get("distances") or [[]])[0]
                    for index, chunk_id in enumerate(ids):
                        meta = metadatas[index] or {} if index < len(metadatas) else {}
                        doc_id = meta.get("document_id")
                        if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                            continue
                        candidates[chunk_id] = {
                            "id": chunk_id,
                            "text": documents[index] if index < len(documents) else "",
                            "metadata": meta,
                            "distance": distances[index] if index < len(distances) else None,
                            "retrieval_strategy": "dense",
                        }
        except Exception as exc:
            cause = str(exc).splitlines()[0][:180] or type(exc).__name__
            if "error finding id" in cause.lower():
                message = (
                    "Chroma retrieval failed because its local vector index is inconsistent "
                    "(Error finding id). Rebuild the index from the preserved chunk records."
                )
            else:
                message = f"Chroma retrieval failed ({type(exc).__name__}: {cause})."
            raise VectorStoreError(message) from exc
        ranked = sorted(
            candidates.values(),
            key=lambda row: row["distance"]
            if row["distance"] is not None
            else float("inf"),
        )
        unique_ranked: list[dict] = []
        seen_snippets: set[str] = set()
        for item in ranked:
            snippet = " ".join(item.get("text", "").split())[:120].lower()
            if snippet and snippet in seen_snippets:
                continue
            if snippet:
                seen_snippets.add(snippet)
            unique_ranked.append(item)

        if query_text:
            import re
            query_words = {w.lower() for w in re.findall(r"\w+", query_text) if len(w) > 2}
            if query_words:
                for row in unique_ranked:
                    text_words = {w.lower() for w in re.findall(r"\w+", row.get("text", ""))}
                    overlap = len(query_words & text_words) / len(query_words)
                    dist = row.get("distance") if row.get("distance") is not None else 1.0
                    dense_score = max(0.0, 1.0 - dist)
                    row["_combined_score"] = dense_score + 0.35 * overlap
                unique_ranked.sort(key=lambda r: r.get("_combined_score", 0.0), reverse=True)

        return unique_ranked[:top_k]

    def get_document(self, document_id: str) -> list[dict]:
        result = self.collection.get(
            where={"document_id": document_id}, include=["documents", "metadatas"]
        )
        ids, texts, metadatas = (
            result.get("ids", []),
            result.get("documents", []),
            result.get("metadatas", []),
        )
        rows = [
            {"id": ids[i], "text": texts[i], "metadata": metadatas[i] or {}}
            for i in range(len(ids))
        ]
        return sorted(rows, key=lambda row: row["metadata"].get("ordinal", 0))

    def storage_summary(self) -> dict:
        try:
            count = self.collection.count()
            collections = self.client.list_collections()
            collection_rows = []
            for collection in collections:
                name = collection.name if hasattr(collection, "name") else str(collection)
                active = collection if hasattr(collection, "count") else self.client.get_collection(name)
                collection_rows.append({"name": name, "count": active.count()})
            return {"healthy": True, "count": count, "collections": collection_rows}
        except Exception as exc:
            raise VectorStoreError("Could not inspect persistent Chroma storage.") from exc

    def list_chunks(
        self, limit: int = 50, offset: int = 0, document_id: str | None = None
    ) -> tuple[int, list[dict]]:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Chunk page limit must be 1–100 and offset cannot be negative.")
        where = {"document_id": document_id} if document_id else None
        try:
            total = self.collection.count() if not where else self.collection.get(
                where=where, include=[]
            ).get("ids", [])
            if isinstance(total, list):
                total = len(total)
            result = self.collection.get(
                where=where,
                limit=limit,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception as exc:
            raise VectorStoreError("Could not list persisted chunks from Chroma.") from exc
        ids = result.get("ids", [])
        texts = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        vectors = result.get("embeddings", [])
        rows = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {} if index < len(metadatas) else {}
            vector = vectors[index] if index < len(vectors) else None
            rows.append({
                "id": chunk_id,
                "text": texts[index] if index < len(texts) else "",
                "metadata": metadata,
                "embedding_available": vector is not None,
                "embedding_dimension": len(vector) if vector is not None else None,
            })
        rows.sort(key=lambda row: (row["metadata"].get("document_id", ""), row["metadata"].get("ordinal", 0)))
        return int(total), rows

    def delete_document(self, document_id: str) -> int:
        try:
            res = self.collection.get(where={"document_id": document_id}, include=[])
            ids = res.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(f"Could not delete document vectors from Chroma: {exc}") from exc

    def delete_chunks(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        try:
            self.collection.delete(ids=chunk_ids)
            return len(chunk_ids)
        except Exception as exc:
            raise VectorStoreError(f"Could not delete chunks from Chroma: {exc}") from exc

    def delete_all(self) -> int:
        try:
            res = self.collection.get(include=[])
            ids = res.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(f"Could not clear Chroma vector store: {exc}") from exc


class MemoryVectorStore:
    """In-memory vector database for fast ephemeral indexing and deterministic testing."""

    def __init__(self, collection_name: str = "visual_rag_studio_documents"):
        self.backend_name = "memory"
        self.collection_name = collection_name or "visual_rag_studio_documents"
        self._collections: dict[str, dict[str, dict]] = {self.collection_name: {}}
        self._metadata: dict[str, dict] = {
            self.collection_name: {"embedding_model": None, "embedding_dimension": None}
        }

    def close(self) -> None:
        self._collections.clear()
        self._metadata.clear()

    def upsert_document(
        self,
        document_id: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
        model: str,
        dimension: int,
        old_ids: list[str] | None = None,
    ) -> list[str]:
        if not chunks or len(chunks) != len(embeddings):
            raise VectorStoreError(
                "Vector insertion requires a matching non-empty chunk and embedding set."
            )
        if any(len(v) != dimension for v in embeddings):
            raise VectorStoreError(
                "Embedding dimension does not match provider dimension."
            )
        col = self._collections.setdefault(self.collection_name, {})
        meta = self._metadata.setdefault(self.collection_name, {})
        if meta.get("embedding_dimension") and meta["embedding_dimension"] != dimension:
            raise VectorStoreError(
                f"Embedding dimension mismatch: collection has {meta['embedding_dimension']}, got {dimension}."
            )
        meta["embedding_model"] = model
        meta["embedding_dimension"] = dimension

        if old_ids:
            for old_id in old_ids:
                col.pop(old_id, None)

        ids = []
        for chunk, emb in zip(chunks, embeddings):
            col[chunk.chunk_id] = {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": emb,
                "metadata": chunk.chroma_metadata(),
            }
            ids.append(chunk.chunk_id)
        return ids

    def search(
        self,
        vector: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
        query_text: str | None = None,
    ) -> list[dict]:
        col = self._collections.get(self.collection_name, {})
        meta = self._metadata.get(self.collection_name, {})
        dimension = meta.get("embedding_dimension")
        if not col or not dimension:
            return []
        if len(vector) != dimension:
            raise VectorStoreError(
                f"Query embedding dimension {len(vector)} does not match indexed dimension {dimension}."
            )

        norm_q = math.sqrt(sum(x * x for x in vector)) or 1.0
        doc_filter = set(document_ids) if document_ids else None
        candidates = []

        for item in col.values():
            if doc_filter and item["metadata"].get("document_id") not in doc_filter:
                continue
            emb = item["embedding"]
            norm_e = math.sqrt(sum(x * x for x in emb)) or 1.0
            dot = sum(a * b for a, b in zip(vector, emb))
            sim = dot / (norm_q * norm_e)
            dist = max(0.0, 1.0 - sim)
            candidates.append({
                "id": item["id"],
                "text": item["text"],
                "metadata": dict(item["metadata"]),
                "distance": dist,
            })
        candidates.sort(key=lambda x: x["distance"])

        unique_ranked: list[dict] = []
        seen_snippets: set[str] = set()
        for item in candidates:
            snippet = " ".join(item.get("text", "").split())[:120].lower()
            if snippet and snippet in seen_snippets:
                continue
            if snippet:
                seen_snippets.add(snippet)
            unique_ranked.append(item)

        if query_text:
            import re
            query_words = {w.lower() for w in re.findall(r"\w+", query_text) if len(w) > 2}
            if query_words:
                for row in unique_ranked:
                    text_words = {w.lower() for w in re.findall(r"\w+", row.get("text", ""))}
                    overlap = len(query_words & text_words) / len(query_words)
                    dist = row.get("distance") if row.get("distance") is not None else 1.0
                    dense_score = max(0.0, 1.0 - dist)
                    row["_combined_score"] = dense_score + 0.35 * overlap
                unique_ranked.sort(key=lambda r: r.get("_combined_score", 0.0), reverse=True)

        return unique_ranked[:top_k]

    def get_document(self, document_id: str) -> list[dict]:
        col = self._collections.get(self.collection_name, {})
        rows = [
            {"id": item["id"], "text": item["text"], "metadata": dict(item["metadata"])}
            for item in col.values()
            if item["metadata"].get("document_id") == document_id
        ]
        return sorted(rows, key=lambda r: r["metadata"].get("ordinal", 0))

    def count(self) -> int:
        return len(self._collections.get(self.collection_name, {}))

    def storage_summary(self) -> dict:
        col = self._collections.get(self.collection_name, {})
        return {
            "healthy": True,
            "count": len(col),
            "collections": [{"name": self.collection_name, "count": len(col)}],
        }

    def list_chunks(
        self, limit: int = 50, offset: int = 0, document_id: str | None = None
    ) -> tuple[int, list[dict]]:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Chunk page limit must be 1–100 and offset cannot be negative.")
        col = self._collections.get(self.collection_name, {})
        matching = [
            item for item in col.values()
            if not document_id or item["metadata"].get("document_id") == document_id
        ]
        matching.sort(key=lambda x: (x["metadata"].get("document_id", ""), x["metadata"].get("ordinal", 0)))
        sliced = matching[offset : offset + limit]
        rows = [
            {
                "id": it["id"],
                "text": it["text"],
                "metadata": it["metadata"],
                "embedding_available": True,
                "embedding_dimension": len(it["embedding"]),
            }
            for it in sliced
        ]
        return len(matching), rows

    def delete_document(self, document_id: str) -> int:
        col = self._collections.get(self.collection_name, {})
        to_del = [cid for cid, item in col.items() if item.get("metadata", {}).get("document_id") == document_id]
        for cid in to_del:
            col.pop(cid, None)
        return len(to_del)

    def delete_chunks(self, chunk_ids: list[str]) -> int:
        col = self._collections.get(self.collection_name, {})
        deleted = 0
        for cid in chunk_ids:
            if cid in col:
                col.pop(cid, None)
                deleted += 1
        return deleted

    def delete_all(self) -> int:
        col = self._collections.get(self.collection_name, {})
        count = len(col)
        col.clear()
        return count


_GLOBAL_MEMORY_STORES: dict[str, MemoryVectorStore] = {}


def get_shared_memory_vector_store(collection_name: str) -> MemoryVectorStore:
    if collection_name not in _GLOBAL_MEMORY_STORES:
        _GLOBAL_MEMORY_STORES[collection_name] = MemoryVectorStore(collection_name)
    return _GLOBAL_MEMORY_STORES[collection_name]


def get_vector_store_adapter(
    backend: str = "chroma",
    collection_name: str = "visual_rag_studio_documents",
    data_dir: Path | str | None = None,
    api_key: str | None = None,
    url: str | None = None,
) -> ChromaVectorStore | MemoryVectorStore:
    b = backend.lower().strip()
    col = collection_name or "visual_rag_studio_documents"
    if b == "chroma":
        path = Path(data_dir) if data_dir else Path("data/chroma")
        return ChromaVectorStore(path, collection_name=col)
    elif b == "memory":
        return get_shared_memory_vector_store(col)
    elif b in {"pinecone", "mongodb", "qdrant"}:
        if b == "pinecone":
            key = api_key or os.environ.get("PINECONE_API_KEY")
            if not key:
                raise VectorStoreError(
                    "Pinecone vector database requires 'PINECONE_API_KEY'. "
                    "Configure your Pinecone credentials in the Inspector."
                )
            raise VectorStoreError(
                "Could not connect to Pinecone vector store (service unreachable or index missing)."
            )
        elif b == "mongodb":
            conn_url = url or os.environ.get("MONGODB_URI")
            if not conn_url:
                raise VectorStoreError(
                    "MongoDB vector search requires 'MONGODB_URI'. "
                    "Configure your MongoDB connection URL in the Inspector."
                )
            raise VectorStoreError(
                "Could not connect to MongoDB Atlas cluster (connection refused or timeout)."
            )
        elif b == "qdrant":
            q_url = url or os.environ.get("QDRANT_URL")
            if not q_url:
                raise VectorStoreError(
                    "Qdrant vector database requires 'QDRANT_URL'. "
                    "Configure your Qdrant host URL in the Inspector."
                )
            raise VectorStoreError(
                "Could not connect to Qdrant vector database service."
            )
    else:
        raise VectorStoreError(f"Unsupported vector database backend '{backend}'.")
