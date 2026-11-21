"""Rebuild the local Chroma vector segment from its preserved chunk records.

Migrate the default legacy ``data/chroma`` directory to the configured path
(``data/chroma-recovered`` by default). Run from ``backend`` with the API stopped:
    uv run python -m scripts.repair_chroma_index

The script reads chunk text and metadata from the existing local Chroma SQLite
metadata segment and re-embeds them locally. The old index remains untouched.
It never calls a generation provider.
"""

from __future__ import annotations

import sqlite3
import sys

from app.config import get_settings
from app.rag.chunking import DEFAULT_SETTINGS, TextChunk
from app.rag.dependencies import get_document_store
from app.rag.providers import SentenceTransformersProvider
from app.rag.vector_store import ChromaVectorStore


def main() -> int:
    settings = get_settings()
    data_path = settings.data_path.resolve()
    if not data_path.exists():
        print(f"The local data directory does not exist: {data_path}")
        return 1
    source_path = data_path / "chroma"
    target_path = settings.chroma_path.resolve()
    if source_path == target_path:
        print("The configured path is still the legacy path; set VRS_CHROMA_PATH to a new directory first.")
        return 1
    if target_path.exists():
        print(f"Recovery target already exists; it was not changed: {target_path}")
        return 1
    source_vectors = ChromaVectorStore(source_path)
    count = source_vectors.collection.count()
    if not count:
        print("The existing Chroma collection is empty; nothing was changed.")
        return 1

    staging_vectors: ChromaVectorStore | None = None
    try:
        staging_vectors = ChromaVectorStore(target_path)
        embedder = SentenceTransformersProvider(local_files_only=True)

        # Chroma's metadata segment still serves IDs, text, and provenance even
        # when its HNSW vector files are inconsistent. Keep the text in process.
        chroma_db = source_path / "chroma.sqlite3"
        with sqlite3.connect(f"file:{chroma_db.as_posix()}?mode=ro", uri=True) as connection:
            ids = [row[0] for row in connection.execute("SELECT embedding_id FROM embeddings")]
        if len(ids) != count:
            raise RuntimeError(f"Chroma ID count mismatch: expected {count}, found {len(ids)}.")
        rows_by_document: dict[str, list[dict]] = {}
        batch_size = 128
        for start in range(0, len(ids), batch_size):
            records = source_vectors.collection.get(
                ids=ids[start : start + batch_size], include=["documents", "metadatas"]
            )
            for chunk_id, text, metadata in zip(
                records["ids"], records["documents"], records["metadatas"], strict=True
            ):
                if not text or not metadata or not metadata.get("document_id"):
                    raise RuntimeError("An existing chunk is missing its text or provenance metadata.")
                section_path = tuple(
                    part.strip() for part in (metadata.get("section_path") or "").split("/") if part.strip()
                )
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    document_id=str(metadata["document_id"]),
                    document_name=str(metadata.get("document_name") or "Unknown source"),
                    ordinal=int(metadata.get("ordinal") or 0),
                    text=text,
                    strategy_id=str(metadata.get("strategy_id") or "recursive"),
                    settings=DEFAULT_SETTINGS.copy(),
                    page_number=metadata.get("page_number"),
                    section_path=section_path,
                    source_start=metadata.get("source_start"),
                    source_end=metadata.get("source_end"),
                    parser_id=str(metadata.get("parser_id") or "unknown"),
                )
                rows_by_document.setdefault(chunk.document_id, []).append(
                    {"chunk": chunk, "text": text}
                )

        model = settings.sentence_transformers_model
        total = 0
        for position, (document_id, rows) in enumerate(rows_by_document.items(), start=1):
            chunks = [row["chunk"] for row in rows]
            vectors: list[list[float]] = []
            dimension = None
            for start in range(0, len(chunks), 128):
                embedded, batch_dimension = embedder.embed(
                    model, [chunk.text for chunk in chunks[start : start + 128]]
                )
                if dimension is not None and dimension != batch_dimension:
                    raise RuntimeError("The local embedding model returned inconsistent dimensions.")
                dimension = batch_dimension
                vectors.extend(embedded)
                print(f"Embedding {chunks[0].document_name}: {min(start + 128, len(chunks))}/{len(chunks)}", flush=True)
            if dimension is None:
                continue
            staging_vectors.upsert_document(
                document_id, chunks, vectors, model, dimension
            )
            total += len(chunks)
            print(f"Rebuilt document {position}/{len(rows_by_document)}: {chunks[0].document_name} ({len(chunks)} chunks)", flush=True)

        actual = staging_vectors.collection.count()
        if actual != count or total != count:
            raise RuntimeError(f"Staged vector count mismatch: expected {count}, got {actual}.")

        # Smoke-test the actual persisted vector path using a local embedding only.
        query_vectors, _ = embedder.embed(
            settings.sentence_transformers_model, ["local index recovery check"]
        )
        ready_ids = [
            row["document_id"]
            for row in get_document_store().list()
            if row["status"] == "ready"
        ]
        smoke_hits = staging_vectors.search(query_vectors[0], 5, ready_ids)
        if not smoke_hits:
            raise RuntimeError("The rebuilt index returned no results in the retrieval smoke test.")

        staging_vectors.close()
        staging_vectors = None
        source_vectors.close()

        print(
            f"Recovery complete: {actual} vectors across {len(rows_by_document)} documents.\n"
            f"Recovered collection: {target_path}\n"
            f"Original collection retained: {source_path}"
        )
        return 0
    except Exception as exc:
        print(f"Recovery stopped safely: {type(exc).__name__}: {exc}")
        print("The active Chroma directory was not replaced.")
        return 1
    finally:
        if staging_vectors is not None:
            staging_vectors.close()
        source_vectors.close()
        # Keep an incomplete target for inspection; never overwrite it silently.


if __name__ == "__main__":
    sys.exit(main())
