# Story 6 — Chunks, Persistence, Storage Explorer, and RAG Trace

**Status:** Implemented and locally tested. Live Groq/Hugging Face generation remains unverified and requires per-query external-processing consent.
**Date:** 2026-10-26

## Findings and storage ownership

- Uploaded TXT/Markdown/PDF source bytes are stored below the configured application data directory's `uploads/` folder.
- SQLite through SQLAlchemy owns document identity, filename/type, upload time, ingestion state, profile/recommendation/selection, chunk ID/ordinal index, embedding model/dimension, and errors. The stored chunk list is metadata only; full chunk text is not duplicated into SQLite.
- Persistent Chroma owns the embedding vectors and the retrievable chunk text/metadata, including document ID/name, ordinal, strategy, page and section provenance where available.
- Query executions and answers are not persisted. Execution progress is short-lived in process memory; the query response returns trace data to the caller.
- The configured database and Chroma paths are resolved from the backend configuration and process working directory. The API reveals a managed local location description, not arbitrary absolute server paths.

## Implementation

- Reused the existing stable `chunk.split` node. Its schema now offers only the four implemented strategies: recursive, fixed, sentence-aware, and structure-aware; chunk size/overlap are documented as characters and constrained to the runtime's limits.
- Traditional RAG validation checks the cross-field chunk-size/overlap rules. Ingestion reads the chunk strategy and settings from the serialized graph and synchronizes them to the document's recorded selection before execution.
- Existing deterministic chunk IDs plus Chroma upserts make retries idempotent. Replacement vector IDs are cleaned only after the new vector set is upserted. On failed reindex, prior ready metadata remains ready with an error; new incomplete documents are not reported ready.
- Added `GET /api/v1/storage/summary`, `GET /api/v1/chunks?limit=&offset=&document_id=`, and `GET /api/v1/documents/{id}/preview`. Existing document list/detail/chunk endpoints remain in use. Chunk responses are bounded and include actual text, source provenance, and vector availability/dimension.
- Added navigation views for Workflow, Documents, Chunks, and Storage without replacing the canvas. Documents can be searched, selected, and previewed; chunk pages can be filtered and navigated; storage summary reports actual row/source/vector counts and collection counts.
- Query responses now carry execution/provider/model metadata, status, ranked retrieved chunks with full text and Chroma distance (labeled as distance), context preview/truncation, raw provider output, validated citations, and generation failure detail. Provider failures preserve retrieved evidence. Query history remains unpersisted.
- Groq and Hugging Face requests containing retrieved document context are blocked unless the user explicitly checks the per-question external-processing consent box. Consent resets after a query and when changing providers. Ollama stays a local generation option. The generic provider connectivity check remains separate and sends no document content.
- Groq GPT-OSS behavior from the preceding fix is preserved: `include_reasoning: false` only for `openai/gpt-oss-*` models.

## API and user-interface status

The Storage summary exposes only SQLite/SQLAlchemy and Chroma labels plus managed-path descriptions. It does not accept SQL or filesystem paths. The chunk explorer reads persisted Chroma text rather than synthesizing previews. A document source preview reparses the persisted upload and caps its response at 20,000 characters.

The query trace distinguishes retrieved records from answer citations and returns the exact model output separately from the citation-validated final answer. It does not expose provider reasoning fields.

## Tests and verification

- `backend\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-all`: **50 passed**. Covers the full backend suite, including SQLite store reopen/source persistence; Chroma persistence/reopen/vector dimensions; chunk page/embedding metadata; real temporary-Chroma upload/index/query; reindex without duplicate vectors; failure status; external-provider authorization blocking; and schema cross-field validation. Providers are deterministic test doubles, not live model evidence.
- `backend\.venv\Scripts\ruff.exe check app tests --ignore B008`: **passed**.
- `backend\.venv\Scripts\python.exe -m compileall -q app tests`: **passed**.
- `frontend: npm test`: **21 passed** across 2 files, including Storage/Documents/Chunks navigation and API-backed empty states.
- `frontend: npm run build`: **passed**, including TypeScript and Vite production build.
- Chroma emitted a Python 3.16 deprecation warning from its telemetry dependency; no test failed.

No live Groq or Hugging Face grounded query was run. The API and UI now require explicit external processing consent before any retrieved document evidence is included in a hosted provider request.

## Limitations

- Only the registered Traditional RAG graph executes; other catalog families remain planned.
- Only the four implemented character-based chunking strategies are selectable. Semantic, hierarchical, Markdown-aware, and code-aware strategy entries remain planned.
- One fixed Chroma collection is currently configured. Collection filters are not user-selectable because the runtime currently provisions a single product collection.
- PDF support is selectable text only; OCR and rich layout/table reconstruction are not implemented.
- SQLite migrations, multi-user auth, production concurrency, persistent workflow definitions, and saved query history are out of scope.
- Live Groq/Hugging Face end-to-end generation and model availability were not verified during this implementation.
