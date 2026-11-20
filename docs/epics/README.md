# Initial BMAD Epics and Stories

Planning baseline with implementation progress recorded in the linked story reports. Stories 1–3 are implemented and verified. Story 4 has a tested local vertical slice; Story 5 adds Hugging Face generation configuration and UI; Story 6 adds persistence/storage inspection and query trace; Story 7 diagnoses the live playground and moves provider credentials into the answer-node Inspector. See the reports for exact live-provider and compatibility evidence.

## Epic 1 — Capability catalog and truthful status

- Story 1.1: Define versioned registry schema and validate required metadata/unique IDs.
- Story 1.2: Populate all 37 architecture families/patterns as searchable planned entries with categories, facets, limitations and provider status.
- Story 1.3: Build Gallery search/filter/detail/preview; gate execution by verified status.
- Story 1.4: Record provider/API/license evidence and verification date for each implemented capability.

## Epic 2 — Workflow contract and composition validation

- Story 2.1: Define declarative graph, typed ports and schema-version migration.
- Story 2.2: Validate configs, dependencies, compatibility, provenance and control-flow semantics.
- Story 2.3: Safely import/export graphs and reject code/unknown components.
- Story 2.4: Allow compatible replacement and branch/fork configuration comparison.

## Epic 3 — Editor and editable templates

Story 3 visual-builder scope is implemented; see [Story 3 report](story-3-visual-builder-ui.md). Workflow persistence, editable template graphs, and reusable-template publishing remain planned.

- Story 3.1: Create blank Custom canvas and registry-driven node palette/inspector.
- Story 3.2: Instantiate and edit Traditional RAG template with source template version retained.
- Story 3.3: Save/reopen workflow revisions and publish user graph as a reusable template.

## Epic 4 — Verified deterministic baseline execution

Story 4 implementation and test evidence: [Real Traditional RAG execution](story-4-traditional-rag.md). The API integration path is tested with mock model adapters and real temporary Chroma; live Ollama remains a required verification step.

## Story 5 — Hosted generation provider

Hugging Face adapter and provider configuration UI: [Hugging Face provider report](story-5-huggingface-provider.md). Live access remains unverified until a user-configured token/model succeeds in the explicit provider test.

- Story 4.1: Define adapter protocol and explicit provider configuration.
- Story 4.2: Implement minimal real ingestion/retrieval/generation path using selected, pinned libraries/providers.
- Story 4.3: Run end-to-end with provenance, abstention, errors, budgets and node trace.
- Story 4.4: Compare two validated graph configurations against the same fixture.

## Epic 6 — Storage and execution transparency

Story 6 adds the document/chunk/storage explorer, enforces graph-backed chunk settings, and returns query evidence with explicit external-processing consent. See [Chunks, persistence, and RAG trace](story-6-storage-trace.md) for implementation details and verification.

## Epic 7 — Customizable RagSpark building blocks

[Story 7](story-7-playground-ragspark-first.md) records the playground diagnosis and a current-source RagSpark compatibility gate. Groq connectivity was verified; the active backend has no indexed documents, so no grounded query was run. RagSpark APIs are audited but not installed or integrated because Python 3.14 compatibility remains unverified.

## Epic 7 — Advanced Retrieval Architectures

- [Story 8: Hybrid RAG & Reranking RAG](story-8-hybrid-rag.md) — Dense + BM25 reciprocal rank fusion (RRF) and Flash cross-encoder reranking.
- [Story 9: Multilingual RAG](story-9-multilingual-rag.md) — Cross-lingual retrieval, verified multilingual embedding models, and localized zero-hallucination abstention.
- [Story 10: Generic Agent Node, Autonomous Ingestion Planner, and Agent UX](story-10-agent-node-and-ingestion-planner.md) — Bounded agent orchestration, allowlisted tools, 4-mode autonomous ingestion planner, multiline resizable agent UX, adaptive query planning, multi-strategy chunking, and 16 regression scenarios.

## M1 definition of done

M1 stories are sequenced as a thin vertical slice:

1. **Foundation and decisions:** create frontend/backend structure, lock supported runtime/dependencies, add documented local setup, SQLite/Chroma/Ollama adapters, and explicit loopback-only development mode.
2. **Registry and graph contract:** encode all 37 catalog records and truthful statuses; define node config schemas and typed ports; implement graph validation plus correct/incorrect connection tests.
3. **Product shell and editor:** implement the dark app shell, searchable categorized palette, working React Flow canvas and schema-driven inspector connected to real registry/node definitions.
4. **Traditional RAG execution:** upload one small supported document, parse/chunk/index via configured real components, execute retrieval and generation via configured providers, and return source provenance and visible node/provider errors.
5. **Verification and handoff:** add backend/frontend automated tests, run the frontend production build, document exact setup/configuration/limitations/test commands in README, and keep the ledger current.

All Epic 1 essentials, graph validation, blank canvas, editable Traditional baseline, and at least one completely verified real execution path meet the acceptance criteria in `docs/PRD.md`. If provider setup is unavailable, do not label the template executable; report the blocker and retain a planned state. LangGraph is not an M1 dependency unless the chosen workflow's semantics require its stateful orchestration; Langflow and Dify remain separate optional evaluations.
