# Story 7 — Playground recovery and RagSpark compatibility gate

**Status:** Playground diagnosis verified; frontend fix implemented. RagSpark adapter work is blocked on package/runtime compatibility evidence and remains unimplemented.
**Date:** 2026-10-26

## Playground diagnosis

- Inspected the live local backend without retrieving uploaded document text. `/api/v1/providers/config` reported Groq configured with `openai/gpt-oss-20b`; `/api/v1/providers/health` reported the multilingual Sentence Transformers model available.
- Called the app's provider test endpoint. Groq returned success for its minimal `Reply with exactly OK.` prompt. This sent no user document content and verifies provider access at that moment only.
- `/api/v1/documents` returned no document records. The playground previously rendered its Ask form only when at least one document was indexed and ready, making an empty backend look like an LLM failure.
- The hosted-provider approval control was below the upload/index controls, obscured in the independently scrolling short playground pane. Without authorization the backend deliberately skips hosted generation.
- No grounded query was run. There were no indexed sources in the active backend, and the provider-only test is not evidence of a RAG answer.

## Changes

- Moved generation provider/model/key setup and provider test into the `generation.answer` node Inspector. **Configure Generate Answer** selects that node or adds it to a blank graph.
- Retained server-only, session-memory Groq credential behavior; the key never enters graph JSON or browser storage.
- Moved hosted-provider authorization immediately beside the Ask area and disabled Ask until the user explicitly authorizes sending retrieved document excerpts. Provider testing is also required before query execution.
- Kept the question field visible when the backend has no indexed documents and explained the upload/index prerequisite rather than silently hiding the form.
- Increased the playground pane's available height and provided a distinct, styled scrollbar.

## RagSpark source audit and decision

Rechecked the canonical GitHub default-branch source, including package metadata and parser, chunker, embedder, vector store, reranker, prompt, LLM, and RAG modules. Actual exposed API lists and known compatibility gaps are recorded in `docs/library-usage.md`.

RagSpark is not installed or locked in this app. The app currently runs Python 3.14.8; RagSpark declares `>=3.10`, but its classifiers name only 3.10–3.12. Its compatibility with this app's runtime and current dependency versions is unverified. Current repository tests could not be retrieved via the connected GitHub repository search, and no package tests or integrations were run. As required by the compatibility gate, no dependency was installed and no RagSpark functionality is presented as integrated. The next implementation story should isolate/test the chunker adapter first; expanding other subsystems before checking that boundary would be unsafe.

## Verification

- Live local provider config/health: Groq configured; Sentence Transformers model available.
- Live provider test: **passed** with a short non-document prompt.
- Frontend unit tests: **22 passed** after moving settings into the Inspector and updating the empty-doc experience.
- Frontend production build: **passed** with TypeScript and Vite.
- Backend code was not changed in this story. Full backend suite: **52 passed**; Ruff: **passed**. One Chroma telemetry dependency emits a Python deprecation warning.

## Remaining work

- Upload and index a supported document in the active backend, then explicitly allow Groq processing for a question to verify a real grounded end-to-end answer.
- Verify RagSpark tests and a narrow adapter under the target Python/dependency versions before adding the library dependency or advertising its APIs as executable.

## Follow-up workspace layout (2026-10-26)

The playground no longer contains provider status/settings or upload/indexing controls. Those belong to the Generate Answer and Document Source node inspectors respectively. The playground retains graph validation, the question form, external-data consent, and the answer trace. A keyboard- and pointer-resizable horizontal separator controls the playground height, making the canvas larger by default while allowing users to expand either region. The no-document prompt opens/selects the Document Source node.

**Verification:** frontend tests: **22 passed**; TypeScript/Vite production build: **passed**. Backend APIs and execution code were unchanged in this layout follow-up.
