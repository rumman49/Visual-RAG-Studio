# Story 5 — Hugging Face Generation Provider

**Status:** Adapter and UI implemented; live inference unverified.
**Date:** 2026-10-25

## Delivered

- Added an `httpx`-based Hugging Face Inference Providers chat adapter using the official OpenAI-compatible chat-completions endpoint. Supports explicit model IDs, `fastest`/`cheapest`/`preferred` routing policy, system/user messages, max tokens, temperature, timeout handling, typed service errors, and response-shape validation.
- Retained Ollama as a separately selectable generation provider and as the only embedding provider in this milestone. Model/provider selection is explicit per request; there is no fallback between them.
- Added `HF_TOKEN` as backend-only `SecretStr` configuration; APIs expose presence/configuration only, never the token. Added a deliberate **Test provider** endpoint/UI action that performs a small real completion. Readiness remains `not_tested` until that action succeeds.
- Added query request model/provider selection, graph validation for the chosen adapter, and generation parameters. Retrieval, empty-evidence abstention, and citation ID validation remain in the shared pipeline.
- Traditional RAG now loads its registered graph on initial frontend load when available, instead of opening to a blank canvas. The template picker continues to include the catalog entry.

## Hugging Face API and cost notes

The adapter targets `https://router.huggingface.co/v1/chat/completions` and supports provider policy suffixes as documented by Hugging Face. A public model identifier is not treated as proof that hosted inference is available; the explicit provider test checks the selected model/provider now. Monthly hosted inference credits are limited and subject to change; usage beyond available credits can require paid usage. See the official [Inference Providers overview](https://huggingface.co/docs/inference-providers/en/index), [chat completion task](https://huggingface.co/docs/inference-providers/en/tasks/chat-completion), and [pricing and billing](https://huggingface.co/docs/inference-providers/pricing).

## Verification

Commands executed: `uv run pytest -q` (**41 passed**), `uv run ruff check app tests --ignore B008` (**passed**), `uv run python -m compileall -q app` (**passed**), `npm test` (**14 passed**), and `npm run build` (**passed**, including TypeScript and Vite).

The suite covers controlled chat response parsing, preserved system/user prompts and parameters, malformed/empty completions, missing token, invalid model IDs, authentication, permission, credit/quota, rate-limit, unavailable model, unexpected server errors, timeout, and no silent provider fallback. The API end-to-end test uses mocked models and real temporary Chroma. It does not establish live Ollama or Hugging Face inference.

The workspace had no `HF_TOKEN` in the process environment or `.env`, and Ollama model IDs were unset; Ollama was previously observed unavailable. Therefore no live-provider call was made. Configure the secret locally in `.env`, choose a model in the UI, and click **Test provider** to verify it without putting the token in the browser or chat.

The gallery template remains marked planned pending a successful real provider-backed Traditional RAG query. See [Story 4](story-4-traditional-rag.md) for parsing, chunking, Chroma, and mock-provider pipeline evidence.
