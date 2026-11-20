# Story 4 — Real Traditional RAG Execution

**Status:** Implemented vertical slice; live-provider acceptance remains unverified.
**Date:** 2026-10-25

## Delivered

- Upload and list `.txt`, `.md`, `.markdown`, and selectable-text `.pdf` files with a 10 MiB default limit. Files and SQLite metadata are stored below ignored `data/`; parser failures are persisted as failed documents with a visible reason.
- UTF-8/Markdown and PyMuPDF parsing; evidence-based document profile; deterministic recommendation; explicit strategy selection/override; provenance-preserving recursive, fixed, sentence-aware, and structure-aware chunking.
- Explicit Ollama embedding/chat adapter with health, configured-model detection, timeouts, response/dimension checks, and no downloads or fallback.
- Persistent Chroma cosine collection, model/dimension compatibility checks, dense Top-K retrieval, stale-vector cleanup after upsert, context limit that retains complete chunks, deterministic empty-evidence abstention, and citation-ID validation.
- Fixed deterministic Traditional RAG runner boundary that validates the registered graph topology and configured provider instances. Other architecture families stay in the catalog as planned.
- Existing React Flow builder loads the registered Traditional RAG graph and preserves provider instance IDs. The playground now uploads, displays profile/recommendation, accepts chunk strategy/config, indexes, queries, displays actual citation excerpts, checks Ollama readiness, and surfaces API errors.
- Updated library usage ledger and setup/testing/limitations in the README.

## Files

Backend implementation: `backend/app/rag/{contracts,parser,chunking,providers,recommendations,storage,vector_store,pipeline,dependencies,graph,routes}.py`; updated `backend/app/{config,main,registry,validation}.py`, `.env.example`, and graph/runtime tests.

Frontend: `frontend/src/VisualBuilder.tsx`, `frontend/src/builder.css`, and updated `frontend/src/VisualBuilder.test.tsx`.

Docs: `README.md`, `docs/library-usage.md`, and this report.

## Providers and dependencies

Reused PyMuPDF for text PDFs, SQLite/SQLAlchemy for metadata, Chroma `PersistentClient` for local vectors, and `httpx` for Ollama. RagSpark remains not integrated after its exact source/API/fallback review; LangChain, LangGraph, Langflow, and Dify are not installed. See `docs/library-usage.md` for per-capability evidence and limits.

Versions imported in this workspace: Python 3.14.8, FastAPI 0.162.0, SQLAlchemy 2.1.6, Chroma 1.5.9, PyMuPDF 1.28.2, httpx 0.28.1, Node 24.21.0, npm 11.19.0, React Flow 12.12.0.

## API

- `GET /api/v1/providers/health`
- `POST /api/v1/documents`, `GET /api/v1/documents`, `GET /api/v1/documents/{id}`
- `GET /api/v1/documents/{id}/profile`, `POST /api/v1/documents/{id}/recommendation`
- `POST /api/v1/documents/{id}/chunking-selection`, `POST /api/v1/documents/{id}/ingest`
- `GET /api/v1/documents/{id}/chunks`, `POST /api/v1/query`

## Verification

- `uv run pytest -q`: **30 passed**. Includes API upload → profile/recommendation → selection → indexing → query using deterministic fake model responses and real temporary Chroma; this is not live Ollama evidence.
- `npm test`: **12 passed**.
- `npm run build`: **passed** (TypeScript and Vite production build).
- `uv run ruff check app tests --ignore B008`: **passed**. B008 is ignored because it flags FastAPI's standard `Depends(...)` and `File(...)` defaults.
- Chroma persistent reopen, dense retrieval, and dimension mismatch were tested against actual local Chroma with temporary data. A complete parser→chunker→mock embedding/generation→real Chroma→retrieval/citation runner test and API integration test use deterministic fake providers; they are not model integration tests.

## Unverified acceptance / limits

- Ollama service was unavailable during environment inspection, and model variables were unset. No live Ollama embedding or generation run occurred. No model was downloaded. Accordingly the gallery template remains marked `planned`; the frontend does not present the unavailable provider as ready. A live configured Ollama end-to-end run is required before marking it tested/executable.
- The API upload/index/query test uses fake model adapters and real local Chroma. A live Ollama-backed API run remains outstanding.
- PDF OCR, image-only PDFs, complex layout/table reconstruction, language detection, semantic/hierarchical chunking, authentication, arbitrary graph execution, graph persistence, and other architecture runners are not included.
- Chunk sizes are characters. SQLite schema migration and multi-user/concurrent production operation are not addressed.
