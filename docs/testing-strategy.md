# Testing Strategy

Story 2 adds backend registry, graph contract, and document/chunking contract unit tests. M1 still requires ingestion/execution tests, frontend component/interaction tests, and a production frontend build; report actual results and provider-gated tests separately.

- **Registry/schema unit checks:** all catalog IDs unique, categories/tags valid, status fields independent, all 37 required display names present, every provider status backed by evidence, executable entries have graph references.
- **Contract tests:** node config JSON Schemas, typed ports, required dependency/config checks, modality/provenance compatibility, embedding dimensions and index settings, invalid-cycle/loop constraints.
- **Graph tests:** valid baseline topology, invalid edge rejection, safe import/export round trips, version migration, unknown-node and code-field rejection.
- **Adapter integration tests:** actual package API with pinned optional dependency, deterministic local fixtures, provider errors, no silent fallback, provenance retention. External provider tests must be explicitly opt-in.
- **Execution tests:** baseline ingest/query, node output trace, abstention on insufficient evidence, resource bounds, cancellation/timeouts, redaction, bounded agent loop where applicable.
- **Frontend tests:** Gallery search/filter/status and planned gating; blank canvas; instantiate/edit/replace/branch/compare/save; actionable validation and execution errors; accessible keyboard navigation.
- **Release gates:** every executable template passes contract and end-to-end tests in supported runtime; catalog planned entries never claim execution. Report exact commands, versions and results; do not infer passing from mocks alone.

Story 2 tests are in `backend/tests/test_capability_graph_contracts.py`; they exercise metadata-only strategy behavior and do not claim strategy execution. API tests should cover search/filter semantics and 404 behavior as the API surface evolves.
