"""SQLite metadata and private local source-file storage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Integer, String, Text, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(240), nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(200))
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="uploaded")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    profile: Mapped[dict | None] = mapped_column(JSON)
    recommendation: Mapped[dict | None] = mapped_column(JSON)
    plan: Mapped[dict | None] = mapped_column(JSON)
    selection: Mapped[dict | None] = mapped_column(JSON)
    chunks: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    stale_chunks: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    embedding_model: Mapped[str | None] = mapped_column(String(200))
    embedding_dimension: Mapped[int | None] = mapped_column(Integer)
    vector_backend: Mapped[str | None] = mapped_column(String(64))
    collection_name: Mapped[str | None] = mapped_column(String(128))
    error: Mapped[str | None] = mapped_column(Text)


class WorkflowRow(Base):
    __tablename__ = "saved_workflows"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(240), nullable=False)
    graph: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DocumentStore:
    def __init__(self, database_url: str, data_root: str | Path):
        self.data_root = Path(data_root).resolve()
        self.upload_root = self.data_root / "uploads"
        self.upload_root.mkdir(parents=True, exist_ok=True)
        if database_url.startswith("sqlite:///") and ":memory:" not in database_url:
            db_part = database_url.removeprefix("sqlite:///")
            db_path = Path(db_part)
            if not db_path.is_absolute():
                db_path = Path.cwd() / db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = (
            {"connect_args": {"check_same_thread": False}}
            if database_url.startswith("sqlite")
            else {}
        )
        if database_url.endswith(":memory:"):
            kwargs["poolclass"] = StaticPool
        self.engine = create_engine(database_url, **kwargs)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        if database_url.startswith("sqlite") and ":memory:" not in database_url:
            with self.engine.connect() as conn:
                try:
                    result = conn.execute(text("PRAGMA table_info(documents)"))
                    existing_cols = {row[1] for row in result.fetchall()}
                    if existing_cols:
                        if "vector_backend" not in existing_cols:
                            conn.execute(text("ALTER TABLE documents ADD COLUMN vector_backend VARCHAR(64)"))
                        if "collection_name" not in existing_cols:
                            conn.execute(text("ALTER TABLE documents ADD COLUMN collection_name VARCHAR(128)"))
                        if "plan" not in existing_cols:
                            conn.execute(text("ALTER TABLE documents ADD COLUMN plan JSON"))
                        conn.commit()
                except Exception:
                    pass

    def close(self) -> None:
        if hasattr(self, "engine") and self.engine is not None:
            self.engine.dispose()

    def create(
        self, filename: str, file_type: str, content_type: str | None, content: bytes
    ) -> str:
        self.upload_root.mkdir(parents=True, exist_ok=True)
        document_id = str(uuid4())
        target = self.upload_root / f"{document_id}.{file_type}"
        target.write_bytes(content)
        with self.sessions.begin() as session:
            session.add(
                DocumentRow(
                    id=document_id,
                    filename=filename,
                    file_type=file_type,
                    content_type=content_type,
                    source_path=str(target),
                    status="uploaded",
                    created_at=datetime.now(UTC),
                    chunks=[],
                    stale_chunks=[],
                )
            )
        return document_id

    def get(self, document_id: str) -> dict | None:
        with self.sessions() as session:
            row = session.get(DocumentRow, document_id)
            return self._serialize(row) if row else None

    def list(self) -> list[dict]:
        with self.sessions() as session:
            rows = session.scalars(
                select(DocumentRow).order_by(DocumentRow.created_at.desc())
            ).all()
            return [self._serialize(row) for row in rows]

    def storage_summary(self) -> dict:
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        documents = self.list()
        present = 0
        for row in documents:
            try:
                source = self._source_path(row["document_id"])
                present += int(source.is_file())
            except (OSError, RuntimeError, KeyError):
                continue
        return {
            "document_count": len(documents),
            "source_files_present": present,
        }

    def _source_path(self, document_id: str) -> Path:
        with self.sessions() as session:
            row = session.get(DocumentRow, document_id)
            if row is None:
                raise KeyError(document_id)
            path = Path(row.source_path).resolve()
        if not path.is_relative_to(self.upload_root.resolve()):
            raise RuntimeError("Stored document path is outside the managed upload directory.")
        return path

    def source_bytes(self, document_id: str) -> bytes:
        return self._source_path(document_id).read_bytes()

    def update(self, document_id: str, **fields) -> dict:
        allowed = {
            "status",
            "profile",
            "recommendation",
            "plan",
            "selection",
            "chunks",
            "stale_chunks",
            "embedding_model",
            "embedding_dimension",
            "vector_backend",
            "collection_name",
            "error",
        }
        if set(fields) - allowed:
            raise ValueError("Unsupported document metadata field.")
        with self.sessions.begin() as session:
            row = session.get(DocumentRow, document_id)
            if row is None:
                raise KeyError(document_id)
            for key, value in fields.items():
                setattr(row, key, value)
        result = self.get(document_id)
        assert result is not None
        return result

    def delete(self, document_id: str) -> bool:
        """Delete a document row and its uploaded source file."""
        with self.sessions.begin() as session:
            row = session.get(DocumentRow, document_id)
            if row is None:
                return False
            try:
                path = Path(row.source_path).resolve()
                if path.is_file():
                    path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                for match in self.upload_root.glob(f"{document_id}.*"):
                    if match.is_file():
                        match.unlink(missing_ok=True)
            except Exception:
                pass
            session.delete(row)
            return True

    def delete_all(self) -> int:
        """Delete all document records and all uploaded files."""
        with self.sessions.begin() as session:
            rows = session.scalars(select(DocumentRow)).all()
            count = len(rows)
            for row in rows:
                try:
                    path = Path(row.source_path).resolve()
                    if path.is_file():
                        path.unlink(missing_ok=True)
                except Exception:
                    pass
                session.delete(row)
        try:
            if self.upload_root.exists():
                for item in self.upload_root.iterdir():
                    if item.is_file():
                        item.unlink(missing_ok=True)
        except Exception:
            pass
        return count

    def clear_chunks(self, document_id: str) -> dict | None:
        """Clear indexed chunks and reset document status back to 'uploaded'."""
        with self.sessions.begin() as session:
            row = session.get(DocumentRow, document_id)
            if row is None:
                return None
            row.chunks = []
            row.stale_chunks = []
            row.status = "uploaded"
            row.embedding_model = None
            row.embedding_dimension = None
            row.error = None
        return self.get(document_id)

    def remove_chunk_from_document(self, document_id: str, chunk_id: str) -> bool:
        """Remove a specific chunk from the document's stored chunks list."""
        with self.sessions.begin() as session:
            row = session.get(DocumentRow, document_id)
            if row is None:
                return False
            updated_chunks = []
            for c in (row.chunks or []):
                cid = c.get("chunk_id") if isinstance(c, dict) else getattr(c, "chunk_id", None)
                if cid != chunk_id:
                    updated_chunks.append(c)
            row.chunks = updated_chunks
            row.stale_chunks = [cid for cid in (row.stale_chunks or []) if cid != chunk_id]
            if not updated_chunks and row.status == "ready":
                row.status = "uploaded"
            return True

    def cleanup_orphan_files(self) -> dict:
        """Remove any files in upload_root not tracked in SQLite."""
        with self.sessions() as session:
            rows = session.scalars(select(DocumentRow)).all()
            tracked_paths = {Path(r.source_path).resolve() for r in rows}
            tracked_ids = {r.id for r in rows}
        cleaned: list[str] = []
        if self.upload_root.exists():
            for item in self.upload_root.iterdir():
                if item.is_file():
                    is_tracked = item.resolve() in tracked_paths or any(
                        item.name.startswith(f"{doc_id}.") for doc_id in tracked_ids
                    )
                    if not is_tracked:
                        try:
                            item.unlink(missing_ok=True)
                            cleaned.append(item.name)
                        except Exception:
                            pass
        return {"cleaned_count": len(cleaned), "cleaned_files": cleaned}

    def save_workflow(self, workflow_id: str, name: str, graph: dict) -> dict:
        now = datetime.now(UTC)
        with self.sessions.begin() as session:
            existing = session.get(WorkflowRow, workflow_id)
            if existing:
                existing.name = name
                existing.graph = graph
                existing.updated_at = now
                row_id = existing.id
                row_name = existing.name
                row_graph = existing.graph
                row_created = existing.created_at.isoformat()
                row_updated = existing.updated_at.isoformat()
            else:
                new_row = WorkflowRow(
                    id=workflow_id,
                    name=name,
                    graph=graph,
                    created_at=now,
                    updated_at=now,
                )
                session.add(new_row)
                row_id = new_row.id
                row_name = new_row.name
                row_graph = new_row.graph
                row_created = new_row.created_at.isoformat()
                row_updated = new_row.updated_at.isoformat()
            return {
                "id": row_id,
                "name": row_name,
                "graph": row_graph,
                "created_at": row_created,
                "updated_at": row_updated,
            }

    def get_workflow(self, workflow_id: str) -> dict | None:
        with self.sessions() as session:
            row = session.get(WorkflowRow, workflow_id)
            if not row:
                return None
            return {
                "id": row.id,
                "name": row.name,
                "graph": row.graph,
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
            }

    def list_workflows(self) -> list[dict]:
        with self.sessions() as session:
            rows = session.scalars(select(WorkflowRow).order_by(WorkflowRow.updated_at.desc())).all()
            return [
                {
                    "id": r.id,
                    "name": r.name,
                    "graph": r.graph,
                    "created_at": r.created_at.isoformat(),
                    "updated_at": r.updated_at.isoformat(),
                }
                for r in rows
            ]

    def delete_workflow(self, workflow_id: str) -> bool:
        with self.sessions.begin() as session:
            row = session.get(WorkflowRow, workflow_id)
            if row:
                session.delete(row)
                return True
            return False

    @staticmethod
    def _serialize(row: DocumentRow) -> dict:
        return {
            "document_id": row.id,
            "filename": row.filename,
            "file_type": row.file_type,
            "content_type": row.content_type,
            "status": row.status,
            "created_at": row.created_at,
            "profile": row.profile,
            "recommendation": row.recommendation,
            "plan": row.plan,
            "selection": row.selection,
            "chunks": row.chunks or [],
            "stale_chunks": row.stale_chunks or [],
            "embedding_model": row.embedding_model,
            "embedding_dimension": row.embedding_dimension,
            "vector_backend": row.vector_backend,
            "collection_name": row.collection_name,
            "error": row.error,
        }
