---
title: '1-2 API key required for non-loopback access'
type: 'feature'
created: '2026-11-21'
status: 'done'
baseline_commit: '135691b3ff2841b2e9817f26504b8374b3267724'
route: 'dispatch'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The hosted backend (Docker/Space bind 0.0.0.0) exposes every `/api/v1` route with no authentication.

**Approach:** Backend checks `X-API-Key` against `VRS_API_KEY` on all `/api/v1/*` routes (health stays open); production mode refuses to start without a key. Frontend prompts for the key, holds it in memory only, sends it on every request, and shows a clear 401 state with retry.

## Boundaries & Constraints

**Always:** Constant-time key comparison; key never logged, returned or stored in `localStorage`/`VITE_`; 401 body is `{detail:{code,message}}`; `/api/health/*` stays open; add `VRS_API_KEY` to `.env.example`.

**Never:** No CORS changes (Story 1.3); no per-user sessions; no silent bypass in production.

## Decisions

- Key is required when `VRS_ENV=production` or `VRS_HOST` is non-loopback. Non-loopback `VRS_HOST` is allowed only when `VRS_API_KEY` is set. Production without a key fails at startup naming `VRS_API_KEY`.
- Add a small dismissible UI notice in loopback dev with no key; `/api/health/ready` gains an `auth` field (`required` | `none`).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected | Error |
|---|---|---|---|
| Key required, none sent | `/api/v1/*` no header | 401 `api_key_required` | `{detail:{code,message}}` |
| Wrong key | bad `X-API-Key` | 401 `api_key_invalid` | same shape |
| Correct key | matching header | request proceeds | N/A |
| Health | `/api/health/live` no key | 200 | N/A |
| Dev, no key | development, loopback | open; ready reports `auth: none` | N/A |
| Production, no key | `VRS_ENV=production` | startup error naming `VRS_API_KEY` | refuses to start |

</frozen-after-approval>

## Code Map

- `backend/app/config.py` -- add `api_key: SecretStr | None` (`VRS_API_KEY`); extend `require_protection_outside_loopback`.
- `backend/app/main.py` -- auth dependency on `rag_router` and `/api/v1/*` routes defined here; `auth` in readiness.
- `backend/app/rag/routes.py` -- `router = APIRouter(prefix="/api/v1")`.
- `frontend/src/api.ts` -- `apiFetch` attaches in-memory key; 401 signalling.
- `frontend/src/VisualBuilder.tsx` -- key prompt, 401 state with retry, no-auth notice.
- `backend/tests/`, `frontend/src/api.test.ts` -- new tests.

## Tasks & Acceptance

**Execution:**
- [ ] `backend/app/config.py` -- add `api_key`, startup validation -- production without key refuses to start
- [ ] `backend/app/auth.py` -- `require_api_key` dependency (hmac.compare_digest) -- typed 401s
- [ ] `backend/app/main.py` -- apply dependency to all `/api/v1` routes; readiness `auth` field
- [ ] `.env.example` -- document `VRS_API_KEY`
- [ ] `backend/tests/test_api_key_auth.py` -- cover every matrix row
- [ ] `frontend/src/api.ts` -- in-memory key, `X-API-Key` header, 401 handling
- [ ] `frontend/src/VisualBuilder.tsx` -- key prompt, 401 retry state, no-auth notice
- [ ] frontend tests -- header attached, key not in storage, 401 state

**Acceptance Criteria:**
- Given production mode with a key, when a request lacks or mis-sends it, then 401 `{detail:{code,message}}` and health stays open
- Given the right key, when sent as `X-API-Key`, then accepted
- Given the key is entered in the UI, then it lives only in memory and is sent on every request

## Implementation Notes
- Backend: `app/auth.py` installs an HTTP middleware (not a per-route dependency) so every `/api/v1/*` route, including ones in `main.py` and the RAG router, is covered. It is registered before CORS so 401s carry CORS headers; OPTIONS preflights pass through. `config.py` gains `VRS_API_KEY`; production or non-loopback host without a key fails validation naming `VRS_API_KEY`. `/api/health/ready` reports `auth`.
- Frontend: `api.ts` holds the key in module memory and adds `X-API-Key` only to `/api/v1` requests; `AuthBoundary.tsx` wraps `VisualBuilder`, prompts on 401, remounts the builder on retry, and shows a dismissible no-auth notice.
- Deploy impact: `backend/Dockerfile` sets `VRS_ENV=production`, so the Space now refuses to start until a `VRS_API_KEY` secret is set. Docs (AD-11 / AD-12 text, `AGENTS.md` loopback policy line) were not updated.
- Verified: ruff clean; backend pytest 196 passed; frontend 39 tests passed, build ok. Review layers skipped (subagents not authorized).
