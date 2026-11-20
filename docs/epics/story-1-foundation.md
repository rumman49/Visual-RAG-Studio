# Story 1 Report — Project Foundation

**Status:** Implemented and verified for the foundation in the current environment. RAG functionality is explicitly out of scope for this story.

## Summary

Added a clean FastAPI backend and React/TypeScript/Vite frontend foundation, Pydantic settings with loopback-only binding, truthful liveness/readiness endpoints, a same-origin API proxy, generated-data/secrets ignore rules, `.env.example`, initial smoke-test source, and setup/security/limitations instructions in README. The landing surface clearly says RAG execution is not configured. No graph UI, registry API, RAG runtime, upload path, model calls, or mock answers were added.

## Files

- Root: `.gitignore`, `.env.example`, `README.md`.
- Backend: `backend/pyproject.toml`, `backend/app/{__init__,config,main}.py`, `backend/tests/test_health.py`.
- Frontend: `frontend/package.json`, `index.html`, `vite.config.ts`, TypeScript configs, `src/main.tsx`, `src/FoundationStatus.tsx`, `src/styles.css`, `src/test-setup.ts`, `src/main.test.tsx`.
- Planning adjustments: `docs/PRD.md`, `docs/architecture.md`, `docs/UX-specification.md`, `docs/capability-registry.md`, `docs/library-usage.md`, `docs/api-contracts.md`, `docs/security.md`, `docs/testing-strategy.md`, `docs/epics/README.md`.

## Commands and results

- `git status --short --branch` → failed: `fatal: not a git repository`; workspace was not initialized as Git.
- `python --version` → Windows Store shim could not start. Located and ran `C:\Users\rumma\AppData\Local\Programs\Python\Python314\python.exe --version` → Python 3.14.8.
- `node --version` → v24.21.0; `npm --version` → 11.19.0; `uv --version` → 0.12.24.
- `py -0p` → `No installed Pythons found!` (launcher not registered, despite direct interpreter existing).
- Python import check for FastAPI/SQLAlchemy/Pydantic/Chroma → failed at `ModuleNotFoundError: No module named 'fastapi'`.
- Initial `python.exe -m pip install fastapi ... pymupdf ...` → failed after retries: `NameResolutionError ... Failed to resolve 'pypi.org'`; no versions found. Pip cache reported no locally built wheels. A later approved package-registry run succeeded through uv.
- Ollama executable exists (client 0.40.2), but `ollama list` failed to connect to `127.0.0.1:11434`; no model/provider verification.
- `npm create vite@latest frontend -- --template react-ts` → stalled with no output under blocked network; interrupted. No npm package cache entries for React/React Flow/Vite were found.
- Initial `npm install --ignore-scripts --no-audit --no-fund --fetch-retries=0 --fetch-timeout=10000` → failed: `ENOTFOUND registry.npmjs.org`. Packages were later present in `frontend/node_modules` and the lockfile; frontend checks then ran successfully.
- Dependency-free Python syntax/TOML/JSON check → **passed** for backend Python source and manifests.
- Recreated `backend/.venv` with the directly installed Python 3.14.8; `uv sync --python ... --group dev` → installed 96 packages, including FastAPI 0.162.0, Chroma 1.5.9, SQLAlchemy 2.1.6 and pytest 8.4.2. `backend/uv.lock` was generated.
- `npm test` → **passed**, 2 tests. `npm run build` → **passed**, Vite production bundle generated.
- Initial `uv run pytest` exposed missing pytest import-path configuration (`ModuleNotFoundError: No module named 'app'`); adding `pythonpath = ["."]` fixed test discovery.
- `uv run pytest -q` under the restricted sandbox stalled because Starlette's test client could not create its local socket pair. Re-running with approved unsandboxed local test access → **passed**, 2 tests.
- `uv run python -c "import fastapi,chromadb,sqlalchemy,pydantic; ..."` → **passed**.
- Backend started with `uv run uvicorn app.main:app --host 127.0.0.1 --port 8000`; both `/api/health/live` and `/api/health/ready` returned expected truthful JSON. Frontend started with `npm run dev -- --host 127.0.0.1`; the UI was opened and showed **API CONNECTED**.

## Known limitations and next story

The original `.venv` pointed to a managed Python 3.12 interpreter that was unavailable/blocked, although Python 3.14.8 was installed directly. Recreating `.venv` against the installed interpreter resolved backend dependency imports. This agent shell still blocks direct launching of generated venv executables; `uv run` works when `UV_CACHE_DIR` is placed in the workspace. README includes recovery steps. Readiness only reports configuration validity; DB/vector/provider health remains `not_checked`. The UI is a Story 1 status shell, not the workflow builder.
