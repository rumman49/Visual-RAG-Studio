# API Contracts

Story 2 implements the registry and graph-validation endpoints below. The document upload/ingestion, grounded query, persistence, execution, and template-publishing routes remain planned. For M1, prioritize those endpoints only after their providers and execution behavior are real and tested.

- `GET /api/v1/catalog?category=&status=&q=` → implemented catalog listing with category, status, and text filters.
- `GET /api/v1/catalog/{id}` → implemented catalog metadata lookup; returns 404 for unknown IDs.
- `GET /api/v1/registry` → implemented versioned registry snapshot including templates, node types, and provider adapters.
- `GET /api/v1/node-types` → implemented node-definition list with schema-driven configuration and typed ports.
- `POST /api/v1/workflows/validate` → implemented deterministic contract validation; reports structural validity separately from executable status. It never runs a graph.
- `POST /api/v1/workflows` / `GET /api/v1/workflows/{id}` → planned persistence.
- `POST /api/v1/workflows` / `GET /api/v1/workflows/{id}` → persist/retrieve workflow definition and revision metadata.
- `POST /api/v1/workflows/{id}/runs` → validate then start bounded execution; returns run ID.
- `POST /api/v1/documents` → upload one supported, size-limited document; returns document ID and ingestion status/provenance metadata.
- `POST /api/v1/documents/{id}/ingest` → execute the configured ingestion/index nodes with explicit embedding provider; returns indexed source/chunk metadata or a visible error.
- `POST /api/v1/query` → accept workflow ID and question; execute configured retrieval/context/generation nodes and return answer, retrieved source spans, provenance IDs and node events, or an explicit provider/validation error.
- `GET /api/v1/runs/{id}` and `/events` → redacted status, node outputs/errors and trace events; citations point to stored provenance IDs.
- `POST /api/v1/templates` → save a validated user graph as a reusable template with capability/provider metadata.

Graph payload includes schema version, stable workflow/node/edge IDs, node type and config (secret references only). Story 2 Pydantic contracts reject malformed payload fields; graph validation reports unknown types, invalid config, connections, missing inputs, provider availability when supplied, unsupported cycles, and planned nodes when execution is requested. Graph execution and persistence are not part of Story 2.

## Story 3 frontend integration

The visual builder loads the versioned `GET /api/v1/registry` response as its sole source for node definitions, catalog entries, planned templates, and chunking-strategy metadata. It sends the graph portion of the Pydantic `WorkflowGraph` contract to `POST /api/v1/workflows/validate` with `require_executable: false`. Frontend graph positions and React Flow presentation fields are omitted. Validation requests are debounced on graph structure/configuration changes; position-only changes do not trigger validation. API failures are displayed separately from graph issues. These routes do not start workflow execution.
