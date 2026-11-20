# Story 2 — Capability and Graph Contracts

**Status:** implemented and verified for Story 2's declared scope. **Scope:** registry, typed contracts, deterministic validation, API exposure, and focused tests. No workflow runner or RAG provider was added.

## Delivered

- Added a backend-owned Pydantic registry containing the existing 37 architecture families and the conceptual blank-canvas item. Traditional RAG and Naive RAG remain distinct.
- Added the six priority template records (Traditional, Hybrid, Reranking, Multilingual, Agentic, Multimodal RAG). They have no graph and remain `planned`.
- Added schema-only reusable node types for document source/parsing, chunking, embeddings, vector indexing, query input, retrieval, reranking, context assembly, and answer generation. Eight chunking strategy records live in the versioned registry snapshot. No fake runtime identifiers or provider adapters are registered.
- Added typed workflow, node, port, configuration, execution, validation-issue, registry, provider-instance, document-profile, evidence, chunking recommendation, selection, and execution-record contracts. Evidence can be observed, inferred, or unavailable. Recommendation, user choice, and actual execution are separate records.
- Added deterministic graph validation for duplicate IDs/edges, unknown nodes/ports, missing endpoints, incompatible ports, cardinality, missing required inputs/config, invalid config, providers when provided, planned-node execution requests, cycles, and control-flow bounds. Validation never invokes an LLM or executes a graph.
- Added catalog, registry, node-type, and workflow-validation API endpoints. No TypeScript registry copy was introduced.
- Inspected RagSpark source and documented source facts and integration decision in the library ledger. RagSpark remains not installed and not integrated.

## Status truth

`planned` means metadata only; `experimental` means a limited implementation exists; `implemented` means runtime wiring exists but required execution evidence is incomplete; `tested` requires test evidence. Catalog presence does not imply a graph or runtime. A structurally valid graph can still have `executable: false`. All 37 catalog entries and all six priority templates currently remain planned. The blank canvas is also planned pending Story 3's editor.

## Verification

- `git status --short --branch` → unavailable: workspace is not a Git repository (`fatal: not a git repository`). Files were inspected before edits and no pre-existing Story 2 implementation files were overwritten.
- From `backend` in PowerShell: `$env:UV_CACHE_DIR = 'E:\Visual RAG Studio V-4\backend\.uv-cache'; uv run ruff check app tests` → **passed**.
- From `backend`: `$env:UV_CACHE_DIR = 'E:\Visual RAG Studio V-4\backend\.uv-cache'; uv run python -m compileall -q app tests` → **passed**.
- From `backend`: `$env:UV_CACHE_DIR = 'E:\Visual RAG Studio V-4\backend\.uv-cache'; uv run pytest -q` (elevated local socket access because Starlette `TestClient` stalls in the restricted sandbox) → **passed, 20 tests**.
- Registry import smoke check printed `38 6`: 37 architecture entries plus blank canvas and six priority templates.
- The earlier restricted-sandbox full test run stalled due local socket limitations. The elevated rerun above is the recorded result; no test is reported as passed from the stalled attempt.
- No frontend source changed in Story 2, so the frontend test/build was not rerun in this story. Story 1 report records the previously successful frontend tests and production build.

Story 1's environment issue was later resolved: Python dependencies are installed in `backend/.venv`, backend health endpoints and frontend checks/build were run and passed in the previous implementation turn. Therefore Story 1 foundation acceptance is **not pending**; its report documents the exact commands/results. This does not imply any RAG workflow is complete.

## Deferred

- React Flow editor, Gallery UI, shared/generated frontend contracts (Story 3).
- Traditional RAG runtime, document upload/ingestion, embeddings, vector retrieval, generation, citations, persistence and end-to-end execution (Story 4).
- Full document classification/recommendation engine and chunking strategy implementations.
- Optional LangGraph orchestration and all external framework integrations.

## Limitations

The registry's generic descriptions are intentionally conservative and do not claim verified framework support. Dependencies/provider availability are only checked when a node definition declares requirements; the initial nodes declare none because no adapter was selected. Versioned persistence/migration and graph import/export are future work. Stateful control-flow bounds are represented and validated, but such graphs are not executable in this story.
