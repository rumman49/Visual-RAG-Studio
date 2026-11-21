---
title: '1-5 Reproducible hosted image'
type: 'chore'
created: '2026-11-21'
status: 'done'
baseline_commit: 'c84e822'
route: 'oneshot'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Hosting was split: `backend/README.md` declared a Gradio Space (`space_app.py`, installs from `requirements.txt`, which lacked `gradio`), while the Dockerfile ran `app.main` without gradio and repeated its own dependency list; `ragspark` was unpinned in `pyproject.toml`.

**Approach:** Make the Gradio Space entrypoint (`space_app.py`) the single supported one (decision: Gradio Space). Declare every dependency in `requirements.txt`, build the Dockerfile from it and run the same entrypoint, pin `ragspark` to one commit across `pyproject.toml`, `uv.lock`, `requirements.txt`, and document deployment.

</frozen-after-approval>

## Implementation Notes

- `backend/requirements.txt`: added `gradio>=5.50,<6`, aligned ranges with `pyproject.toml`, pinned `ragspark`. `backend/Dockerfile`: installs from `requirements.txt`, copies and runs `space_app.py`. `backend/pyproject.toml`: `ragspark` source gains `rev` (lock updated, two lines).
- Decision: gradio is deliberately not in `pyproject.toml`/`uv.lock`; adding it downgraded shared packages (starlette, pydantic, pillow, huggingface-hub), which would change the dev/CI environment.
- Docs: `backend/README.md` (secrets table) and root `README.md` (deployment steps). `backend/tests/test_deploy_manifests.py` guards pin agreement, gradio declaration, Dockerfile entrypoint and front matter.
- Verified: ruff clean; backend pytest passes; `space_app.py` run locally with gradio on port 7860 returned `{"status":"alive"}` on `/api/health/live` and 401 on `/api/v1/registry` without a key.
- Not verified: the Docker image build and container start (no Docker on this machine) and an actual Hugging Face deployment. Review layers skipped (subagents not authorized).
