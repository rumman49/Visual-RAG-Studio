# Story 7 — Reliability and Acceptance Verification

**Status:** Completed. All backend and frontend test suites pass cleanly. Traditional RAG end-to-end PDF lifecycle and external provider consent/privacy boundaries verified with deterministic fixtures. Live provider acceptance remains truthfully marked `planned`.  
**Date:** 2026-10-26

---

## 1. Executive Summary

Story 7 investigated test suite reliability, eliminated backend test stalling, verified the complete Traditional RAG pipeline end-to-end using selectable-text PDF documents, audited provider consent and privacy boundaries, resolved the query history persistence decision, and performed full regression testing across backend and frontend environments.

All 52 backend tests, 22 frontend unit/integration tests, Ruff linting, Python bytecode compilation, TypeScript type-checking, and Vite production bundle builds pass with zero failures.

---

## 2. Test Suite Investigation and Root Cause of Stalls

### Root Cause Analysis
During previous backend test runs, pytest executions on Windows occasionally stalled or failed during temporary directory teardown. Investigation revealed two root causes:

1. **Windows SQLite and Rust File Locking:**  
   Chroma 1.5.9 utilizes a Rust/C++ bindings engine managing `chroma.sqlite3`, and SQLAlchemy manages SQLite connections. On Windows, file handles held by active client instances prevent directory removal (`PermissionError: [WinError 32]`). When tests did not wrap store creation in `try...finally` blocks or failed before teardown, handles remained open. On subsequent pytest invocations, pytest's `tmp_path` fixture attempted to clean up old temp directories in `%TEMP%\pytest-of-<user>`, hanging or failing on locked files.
2. **Chroma Background Telemetry:**  
   Chroma's default configuration spawned OpenTelemetry / PostHog background network tasks and emitted Python 3.14 deprecation warnings (`asyncio.iscoroutinefunction`). In environments with firewalls, restricted DNS, or offline execution, background network retries caused thread timeouts and test stalls.

### Fixes Applied
- **Telemetry Disabled:** In `app/rag/vector_store.py`, `chromadb.PersistentClient` is now explicitly configured with `chromadb.config.Settings(anonymized_telemetry=False, is_persistent=True)`.
- **Explicit Resource Teardown:** Added explicit `close()` methods with garbage collection (`gc.collect()`) to `ChromaVectorStore` and `engine.dispose()` to `DocumentStore`.
- **Guaranteed Cleanup in Tests:** Wrapped all database and vector store tests in `try...finally` blocks in `backend/tests/test_traditional_rag_runtime.py`, ensuring client and engine handles are always disposed regardless of test outcome.

---

## 3. Traditional RAG End-to-End Verification

The complete 12-step Traditional RAG flow was verified using a real in-memory selectable-text PDF created via PyMuPDF in `test_traditional_rag_pdf_end_to_end_verification_flow`:

1. **PDF Upload:** Real PDF document with selectable text generated using PyMuPDF and uploaded to `POST /api/v1/documents`.
2. **Parsing & Profiling:** Extracted text verified (`file_type == "pdf"`, `page_count == 1`, `meaningful_text_extracted == True`, extraction succeeded).
3. **Chunking Recommendation & Override:** Recommended strategy evaluated; explicit override to `sentence` chunking (`chunk_size=150`, `overlap=20`) recorded via `POST /api/v1/documents/{id}/chunking-selection`.
4. **Ingestion & Indexing:** Executed via `POST /api/v1/documents/{id}/ingest`; generated chunks and vectors indexed with status `ready`.
5. **Storage Record Confirmation:** Validated SQLite document records (`chunk_count >= 1`), Chroma vector persistence, and source page provenance (`page_number == 1` in chunk metadata).
6. **Storage Restart & Reinitialization:** Closed SQLite engine and Chroma client, created fresh store instances pointing to the existing paths, and rebound dependency overrides to simulate application restart.
7. **Post-Restart Verification:** Confirmed SQLite metadata, extracted text preview, Chroma vector counts, and paginated chunks remained accessible after storage reopen.
8. **Retrieval Execution:** Executed query via `POST /api/v1/query`; confirmed `abstained == False` and status `completed`.
9. **Metadata Inspection:** Verified retrieved chunks contained document name, document ID, page number 1, source text, and cosine distance.
10. **Citation Verification:** Validated that citation IDs and page numbers strictly corresponded to retrieved source chunks.
11. **Empty-Retrieval Abstention:** Verified that when no matching chunks exist, the pipeline deterministically abstains (`abstained == True`, status `abstained`, zero citations, standard abstention message).
12. **Generation Provider Error Handling:** Verified that when a generation provider fails, the response reports `status == "generation_failed"`, surfaces the sanitized error message, and preserves all retrieved source evidence.

---

## 4. Provider Consent, Privacy, and Security Boundaries

Verified in `test_provider_consent_privacy_and_credential_sanitization`:

- **External Provider Consent Enforcement:** When querying external providers (Groq or Hugging Face) without `allow_external_processing=True`, generation is blocked (`status == "authorization_required"`). Crucially, zero document text or prompt content is sent to the provider. Retrieved chunks are returned for user inspection.
- **Per-Action Authorization:** When `allow_external_processing=True` is provided, the query executes. Subsequent queries without the flag are immediately blocked again; consent is strictly per-action and never persistent or sticky across requests.
- **Credential Protection:** API keys and tokens (`GROQ_API_KEY`, `HF_TOKEN`) never appear in `GET /api/v1/providers/config`, API responses, error details, or serialized workflow graphs.
- **Probe vs. Grounded Query Isolation:** Provider connectivity tests (`POST /api/v1/providers/test`) send minimal probe messages ("OK") and never transmit user document excerpts.

---

## 5. Decision on Execution History Persistence

**Decision: Query history persistence is deliberately deferred.**

### Rationale
- **Product Scope:** The PRD specifies M1 as a local-first, single-user developer baseline with in-memory execution traces returned directly in `QueryResponse`.
- **Privacy & Security Alignment:** Storing full query prompts, retrieved evidence spans, and model responses to disk creates unnecessary data retention risks without a formal multi-user authentication, tenant isolation, and GDPR/deletion lifecycle boundary.
- **Current Trace Availability:** Callers receive complete query traces (retrieved chunks, distances, context preview, execution ID, timings, citations, provider output) ephemerally in the API response.
- **Deferred Requirements:** If implemented in a future milestone, query history should include execution ID, timestamp, provider/model ID, collection ID, timings, chunk IDs, and error status—without persisting credentials, with user opt-in, and with explicit retention/deletion controls.

---

## 6. Capability Registry Audit

The capability registry was audited in `backend/app/registry.py`:
- All 37 architecture families retain their presence in the catalog with status `ImplementationStatus.PLANNED`.
- Traditional RAG has a fully implemented vertical slice, but because live outbound provider access (Ollama/Groq/Hugging Face) with active real-world credentials has not been executed in this environment, it remains truthfully marked `planned`.
- Advanced architectures (Hybrid, Reranking, Multilingual, Agentic, Multimodal RAG) remain marked `planned` with no unverified claims of execution.

---

## 7. Command Execution and Verification Ledger

| Check | Command | Result |
| :--- | :--- | :--- |
| **Backend Health Tests** | `uv run pytest tests/test_health.py -v` | **2 passed** (2.80s) |
| **Backend Registry Tests** | `uv run pytest tests/test_capability_graph_contracts.py -v` | **18 passed** (1.70s) |
| **Backend Runtime & Verification Tests** | `uv run pytest tests/test_traditional_rag_runtime.py -v` | **32 passed** (6.28s) |
| **Full Backend Test Suite** | `uv run pytest -v` | **52 passed** (5.15s) |
| **Python Linting** | `uv run ruff check app tests --ignore B008` | **All checks passed** |
| **Python Bytecode Compilation** | `uv run python -m compileall -q app tests` | **Passed** (0 errors) |
| **Frontend Tests** | `npm test` (vitest run) | **22 passed** across 2 files (10.63s) |
| **Frontend TypeScript & Production Build** | `npm run build` (`tsc -b && vite build`) | **Passed** (built in 7.34s) |

---

## 8. Incomplete Acceptance Criteria and Next Steps

### Outstanding Verification Gaps
1. **Live Model Execution:** A live integration run with a real, reachable Ollama service or active Groq/Hugging Face API key has not been performed in this offline/local development session. The gallery templates must remain marked `planned` until verified against live models.
2. **Authentication Boundary:** Development binding remains restricted to loopback (`127.0.0.1`). Shared or non-loopback network deployments require an authentication and session boundary.

### Recommended Next Milestone
- **Milestone 2 (M2): Live Provider Integration & Workflow Persistence:**
  - Execute a live model verification pass with configured local Ollama or cloud inference credentials.
  - Implement exportable/importable declarative JSON workflow definitions with strict schema validation.
  - Support multi-document ingestion and collection selection on the canvas.
