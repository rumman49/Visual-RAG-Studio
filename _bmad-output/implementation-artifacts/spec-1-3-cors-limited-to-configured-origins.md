---
title: '1-3 CORS limited to configured origins'
type: 'feature'
created: '2026-11-21'
status: 'done'
baseline_commit: 'fd9fcc9'
route: 'oneshot'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `main.py` allowed any origin with credentials (`allow_origin_regex=^https?://.*`), so any site could call a hosted instance.

**Approach:** Restrict CORS to `VRS_CORS_ORIGINS` (comma-separated); default to `127.0.0.1` and `localhost` on port 5173; remove the regex.

</frozen-after-approval>

## Implementation Notes

- `backend/app/config.py`: `cors_origins` string setting plus `cors_allowed_origins` property (trailing slashes stripped). Wildcards and scheme-less entries fail validation, since wildcards cannot be combined with credentials. `main.py` uses `allow_origins=`.
- `.env.example`, `docs/architecture.md` (AD-11), `README.md`, `AGENTS.md` updated. The architecture spine copy was not touched because it has uncommitted edits from another session.
- Tests: `backend/tests/test_cors_origins.py` (allowed, rejected, default, wildcard rejection, real app).
- Deploy impact: hosted instances must set `VRS_CORS_ORIGINS` to the frontend URL or browsers will be rejected.
- Verified: ruff clean, backend pytest 202 passed. Review layers skipped (subagents not authorized).
