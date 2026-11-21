---
title: '2-1 Workflow revisions'
type: 'feature'
created: '2026-11-22'
status: 'done'
baseline_commit: '6050f2b'
route: 'oneshot'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `/workflows` stored one graph per record and overwrote it on every save, so earlier versions were lost.

**Approach:** Each save appends an immutable revision (number and timestamp) while the workflow record mirrors the latest; revisions can be listed and opened by number; pre-existing workflows become revision 1 at startup.

</frozen-after-approval>

## Implementation Notes

- `backend/app/rag/storage.py`: new `workflow_revisions` table (`WorkflowRevisionRow`, primary key `workflow_id` + `revision`). `save_workflow` always adds revision N+1 (first save is 1), keeps `saved_workflows` as the latest copy, and `updated_at` equals the latest revision's timestamp. `_backfill_workflow_revisions` runs on every store open and inserts revision 1 for workflows without revisions (idempotent, no data loss). `get_workflow(id, revision=None)`; `list_workflow_revisions`; delete removes revisions. Timestamps are now returned as UTC-aware ISO strings (SQLite returns naive values).
- API (`backend/app/rag/routes.py`, `contracts.py`): `WorkflowResponse` gains `revision`; new `GET /api/v1/workflows/{id}/revisions` (newest first) and `GET /api/v1/workflows/{id}/revisions/{n}` (404 `workflow_not_found` / `workflow_revision_not_found`). `GET /workflows/{id}` stays the latest default. Saving through `POST /workflows` still rejects graphs with secrets (Story 1.4).
- Decisions: every save creates a revision even if the graph is unchanged; restoring an old revision is a new save (no destructive rollback endpoint); no revision cap or pruning.
- Not done: the editor does not show revisions yet (Story 2.2). Existing frontend code ignores the new `revision` field.
- Verified: ruff clean; backend pytest 217 passed (`tests/test_workflow_revisions.py` new: numbering, previous kept, exact open by number, latest default, legacy backfill and idempotency, 404s, delete cascade); frontend tests unchanged and passing. Review layers skipped (subagents not authorized).
