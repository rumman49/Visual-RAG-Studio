# Story 3 — Visual Builder UI

**Status:** implemented and verified for the visual graph-editing scope. Graphs are in-memory drafts; Story 3 does not execute RAG.

## Implemented

- Dark desktop builder with searchable/category-filtered palette, interactive React Flow canvas, schema-driven Inspector, template selector, validation/playground panel, and restrained execution-unavailable state.
- Registry and catalog information come from `GET /api/v1/registry`; no 37-entry TypeScript catalog copy exists. The frontend uses returned node ports, schemas, statuses, dependencies, limitations, template metadata, and chunking-strategy metadata.
- Supports adding/selecting/moving/deleting nodes, compatible handle connections, selecting/deleting edges, config editing, workflow naming, blank-canvas reset with confirmation, and backend validation diagnostics. Deleting a node removes its connected edges. Graph serialization excludes canvas coordinates and UI-only data.
- Input/output handle compatibility is derived from backend port data types and `accepts`. The backend remains authoritative. Rejected connections display a warning.
- Draft graph validation uses `POST /api/v1/workflows/validate`, debounced by graph/config changes; position-only moves do not send another request. Out-of-order validation responses are ignored. API failures remain separate from validation results; node and edge issue styling plus the issue list show backend diagnostics.
- Priority planned templates and the full catalog are loaded from the registry. Template entries with no graph are explicitly marked planned and offer draft composition only; no template graph is fabricated. Traditional and Naive catalog identities remain distinct.
- The chunking panel displays strategy metadata/status and explicitly says profiling and recommendations are unavailable. The Run button is disabled and the lower panel reports that execution is unavailable.

## Endpoints

- `GET /api/v1/registry`
- `POST /api/v1/workflows/validate`

No registry/validation mock is used by the application. Frontend tests use clearly isolated registry/validation fixtures.

## State and validation

React Flow controlled node/edge arrays are the single authoritative editable graph state. Node definitions are carried as presentation data and stripped from the serialized payload; each node submits only ID, backend node type ID, config, and provider reference. Connections serialize source/target IDs and port IDs. The graph contract submits deterministic DAG mode. Validation occurs after structural/config edits or explicit user action, not during canvas dragging.

## Verification

- Installed canvas dependency: `npm list @xyflow/react --depth=0` → **@xyflow/react 12.12.0**.
- From `frontend`: `npm test -- --reporter=dot` → **passed, 2 files and 11 tests**.
- From `frontend`: `npm run build` → **passed**, TypeScript project build and Vite production bundle.
- Frontend tests cover palette search/category filtering, node addition, Inspector configuration edits, serialization to the backend graph shape, compatible/incompatible port types, blank-canvas reset confirmation, template status, validation diagnostics/API errors, planned execution state, and chunking strategy metadata. React Flow's `ResizeObserver` is stubbed only for jsdom tests.
- From `backend`: `uv run pytest -q` → **passed, 20 tests**; `uv run ruff check app tests` → **passed**; `uv run python -m compileall -q app tests` → **passed**.
- Direct localhost browser smoke check was unavailable in the restricted shell (`socket access forbidden` for ports 8000 and 5173). API integration behavior is covered by Story 2 backend tests and frontend request/response tests; interactive browser rendering was not manually verified in this turn.
- Workspace is not a Git repository (`git status --short --branch` reports `fatal: not a git repository`), so no Git diff is available.

## Limitations and Story 4 requirements

- No workflow persistence, import/export, backend template instantiation, or saved configuration comparisons.
- Planned templates contain no runnable graph; UI draft composition does not claim executable semantics.
- No document upload/profile/recommendation endpoint exists. Story 4 must add real document profile evidence, recommendation response and provenance, an explicit saved strategy selection, and a separate actual-ingestion strategy record. Unsupported strategies/dependencies must surface errors.
- No provider configuration, document ingestion, embeddings, retrieval, generation, citation, run event/trace, or agentic execution exists. Story 4 must connect a verified Traditional RAG graph to real components and return traceable retrieved sources; the UI must continue to keep execution state distinct from validation.
