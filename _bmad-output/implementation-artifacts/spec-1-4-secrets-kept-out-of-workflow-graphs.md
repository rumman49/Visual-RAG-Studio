---
title: '1-4 Secrets kept out of workflow graphs'
type: 'feature'
created: '2026-11-21'
status: 'done'
baseline_commit: '7fc65d5'
route: 'oneshot'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** A `vector.index` node's `api_key` and `url` config could carry credentials into saved or exported graph JSON (AD-9).

**Approach:** Reject non-empty `api_key` and credential-bearing `url` in vector node config at validation, save and run time with an actionable issue; resolve vector-store keys server-side from `RuntimeCredentials` or env.

</frozen-after-approval>

## Implementation Notes

- `backend/app/validation.py`: `secret_config_issues()` (code `secret_in_graph`), called by `validate_graph`, so both validate paths report it.
- `backend/app/rag/routes.py`: `_reject_graph_secrets()` returns 422 `{detail:{code,message}}` on save, ingest, ingest-batch, query and agent-run; vector keys now come from `runtime_credentials.get_key(backend)` (adapter still falls back to env such as `PINECONE_API_KEY`). The message never echoes the secret.
- `backend/app/registry.py`: `api_key` field description now says it is not accepted in graphs. The field stays in the schema so empty defaults from templates and the frontend keep validating.
- Decision: a `url` with `user:pass@` is treated as a secret too (MongoDB URIs). Pinecone/MongoDB/Qdrant connections are still unimplemented stubs, so only key resolution was changed.
- Not done: the frontend inspector can still show an `api_key` input for vector nodes; typing a key there now yields a validation error rather than being hidden.
- Verified: ruff clean, backend pytest 208 passed (`tests/test_secrets_in_graphs.py` new). Review layers skipped (subagents not authorized).
