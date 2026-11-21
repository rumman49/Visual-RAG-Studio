<!-- bmad:context -->
<!-- Verified 2026-11-21 against ebc8ec6. Managed by bmad-project-context; edits inside this block are replaced on refresh. Keep anything you want preserved outside the markers. -->

## Visual RAG Studio

Local-first visual RAG workflow builder. React/TypeScript/Vite + React Flow in `frontend/`; Python/FastAPI in `backend/` (SQLite, Chroma, Sentence Transformers; generation via Ollama, Groq or Hugging Face). Planning docs live in `docs/` (`PRD.md`, `architecture.md`, `security.md`, `epics/`); BMad output goes to `_bmad-output/`.

## Policy

- Never commit `.env`, API keys or tokens; use `.env.example` for new settings. Keep credentials out of graph JSON, traces and logs.
- Never execute code, shell, SQL or package installs from imported graphs; only allow-listed node types (`docs/security.md`).
- Never add silent provider fallback or implicit mocks; a failed provider surfaces its error.
- Never mark a capability or template executable without recorded end-to-end evidence; unverified stays "planned" (`docs/capability-registry.md`).
- Bind the backend to loopback unless `VRS_API_KEY` is set (production mode requires it); CORS is still open until Story 1.3.

## Where things are

- Registry, graph contracts, validation: `backend/app/registry.py`, `contracts.py`, `validation.py`
- RAG runtime (parsing, chunking, vector store, providers, hybrid, agent, planner): `backend/app/rag/`
- Frontend editor: `frontend/src/VisualBuilder.tsx`; API client in `frontend/src/api.ts`
- Backend tests: `backend/tests/`; frontend tests are `*.test.ts(x)` beside the source
- Deploy: `vercel.json` (frontend), `backend/Dockerfile` and `backend/space_app.py` (Hugging Face Space)

## Running and verifying

- Run backend commands from `backend/` with `uv run` (e.g. `uv run pytest`, `uv run ruff check`); bare `pytest` skips the project environment.
- Backend serves on port 8001; Vite proxies `/api` to it, so start the backend before `npm run dev`.
- Python `>=3.11,<3.15`, Node 20.19+ or 22.12+ (see `README.md` for Windows venv repair).
- Report real test results; do not infer passing from mocked provider tests.

## Known pitfalls

- Do not edit `backend/tmp-close-test/`, `data/` or `frontend/dist/`; they are ignored scratch or build output.

<!-- /bmad:context -->
