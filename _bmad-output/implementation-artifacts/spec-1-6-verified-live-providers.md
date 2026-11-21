---
title: '1-6 Verified live providers'
type: 'feature'
created: '2026-11-21'
status: 'done'
baseline_commit: '6cb842a'
route: 'oneshot'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Provider adapters for Ollama, Groq and Hugging Face were marked implemented with only mocked tests, so provider status in the registry had no live evidence.

**Approach:** Add opt-in `live_providers` tests that make a real minimal-prompt call and run one grounded RAG query against a fixture document per provider; skip unconfigured providers; record status and verification date in the registry and a story report.

</frozen-after-approval>

## Implementation Notes

- `backend/tests/test_live_providers.py` (new, marked `live_providers`): per provider, a real minimal prompt then a real upload, ingest (dependency-free `fast_local` embeddings) and `/api/v1/query` with citation and answer assertions. Unconfigured providers call `pytest.skip` with a reason (shown by `-rs`); a configured provider that fails fails loudly. Opt-in env: `GROQ_API_KEY` (+ `VRS_LIVE_GROQ_MODEL`), `HF_TOKEN` + `VRS_LIVE_HF_MODEL`, `VRS_LIVE_OLLAMA_MODEL` (+ reachable server).
- `backend/pyproject.toml`: bare `pytest` now excludes `live_providers` by default (`-m 'not live_providers'`); `.github/workflows/ci.yml` opt-in run overrides it.
- Registry: `ProviderAdapterDefinition` gains `verification_status` and `evidence`; `verified` requires dated evidence (validator). `groq.chat` and `ollama.chat` are `verified`; `huggingface.chat` stays `unverified` with the failure recorded.

## Live Verification Report (2026-11-21)

| Provider | Model | Result |
| --- | --- | --- |
| Ollama (local) | `gemma3:4b` | Passed: minimal prompt and grounded query with citations |
| Groq | `openai/gpt-oss-20b` | Passed: minimal prompt and grounded query with citations |
| Hugging Face | tried `Qwen/Qwen2.5-7B-Instruct` and 7 others | Not verified: the account has no Inference Providers credit; the remaining models were rejected as unavailable. The test fails with that error, it is not skipped or mocked. |

- Command: `VRS_LIVE_OLLAMA_MODEL=gemma3:4b VRS_LIVE_HF_MODEL=Qwen/Qwen2.5-7B-Instruct uv run pytest -m live_providers -rs`. Keys were read from the environment/`.env` and never printed.
- Default suite (`uv run pytest`): 212 passed, 3 live tests deselected. Ruff clean.
- Not done: `docs/library-usage.md` still says live access is unverified for Groq/Hugging Face (and `docs/capability-registry.md` has uncommitted edits from another session, so it was left alone). Rerun the Hugging Face test after adding credit to flip its status. Review layers skipped (subagents not authorized).
