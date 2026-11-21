# Epic 1 Context: Hardened, Verified Deployment

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

A developer can deploy the app to a server without exposing an open API or leaking keys, and knows the live providers actually work. The hosted Space currently binds all interfaces with permissive CORS and no auth; this epic closes that gap and adds automated verification.

## Stories

- Story 1.1: CI pipeline
- Story 1.2: API key required for non-loopback access
- Story 1.3: CORS limited to configured origins
- Story 1.4: Secrets kept out of workflow graphs
- Story 1.5: Reproducible hosted image
- Story 1.6: Verified live providers

## Requirements & Constraints

- Secrets must never appear in graphs, traces, storage, logs or responses; new settings go in `.env.example`, never committed `.env`.
- Backend stays loopback-only unless an authentication boundary exists.
- Automated tests must run in CI; live-provider evidence is reported separately from mocked tests, and a failed provider surfaces its error (no silent fallback or implicit mocks).
- Capabilities are marked executable only with recorded end-to-end evidence.

## Technical Decisions

- Backend: Python >=3.11,<3.15, run via `uv run` from `backend/` (`pytest`, `ruff check`). Frontend: Node 20.19+/22.12+, `npm test` (vitest) and `npm run build` (`tsc -b && vite build`).
- `ragspark` is a git dependency; pin it consistently between `pyproject.toml` and the Dockerfile.
- Dockerfile must declare `gradio`; entrypoint choice between `space_app` and `app.main` is open.
- Vector-node `api_key`/`url` config can leak into graph JSON (AD-9).
- Windows-specific test-teardown fixes exist and must not cause failures or stalls on Linux.

## Cross-Story Dependencies

- 1.1 establishes the CI that later stories (1.2-1.6, other epics) are verified through.
- 1.6 introduces the `live_providers` test marker that 1.1 must exclude by default.
