# Visual RAG Studio — Initial Architecture

**Status:** Phase 0 recommendation, 2026-10-25. No runtime has been implemented.

## Context and boundary

Use a React/TypeScript/Vite client with React Flow for the product's primary editor. A Python/FastAPI service owns catalog and capability APIs, graph validation, persistence, provider references, and bounded execution. Keep UI graph representation separate from runtime adapter interfaces. The initial runtime can execute a deterministic DAG/topological path; introduce LangGraph only for semantics that need state, branching, loops, streaming, checkpoints, or approvals. M1 defaults: SQLite via SQLAlchemy for local workflow/document metadata; Chroma persistent local storage via a narrow vector-store adapter; Ollama as the configurable local embedding and generation provider. These are replaceable defaults, not hard-coded graph semantics. Development is local-first with separate frontend/backend processes and a same-origin development proxy. Initial deployment target is a single-user self-hosted instance; no public multi-tenant deployment in M1. Authentication is disabled only for loopback development, with an explicit warning; add an API-key/session boundary before binding to non-loopback interfaces. Confirm security/deployment choices before any shared or public deployment.

Frameworks are optional implementation adapters rather than the application contract. No arbitrary code nodes, imported code execution, hidden provider failover, automatic model download, or implicit mocks. Catalog and executable graph contracts are separate. Traditional RAG is the complete transparent loader→chunk→embed/index→retrieve→context→generate path; Naive RAG is a simpler direct top-k retrieve→generate preset with fewer query/context transformations. Both may use shared adapters, while only Traditional RAG is in the M1 runnable scope.

## Proposed structure

```text
AGENTS.md
README.md
docs/{PRD,architecture,UX-specification,capability-registry,library-usage,security,testing-strategy,api-contracts}.md
docs/epics/
frontend/   # React/Vite editor, gallery, inspector, execution views
backend/    # FastAPI, schemas, registry, validation, execution, adapters, persistence, security
tests/      # contracts, backend integration, frontend flows, fixtures
scripts/
deployment/
```

Create only the documents needed for Phase 0 now. The directory was empty and is not a Git repository; this structure is proposed, not scaffolded.

## Key data flows

1. Gallery: UI requests catalog → registry returns metadata/status/provider evidence → UI filters/details/preview; planned status never yields an executable graph.
2. Instantiate/edit: selected template graph (or empty Custom graph) → client edits versioned declarative graph → validation checks schema, typed ports, semantics and configs → save creates a workflow revision.
3. Ingest: validated source metadata/file → constrained parser → chunk records with source/page/offset provenance → explicit embedding adapter → index, recording model/dimension/index config.
4. Query: query input → optional query processors → retrievers → compatible fusion/rerank/context nodes → explicitly configured generator → answer and citations mapped to retrieved provenance; unmet evidence threshold yields abstention.
5. Execution: API validates graph and budgets → deterministic executor or selected orchestration adapter → node events/results/errors → redacted trace; external secrets referenced through secure provider configuration, never serialized into graph exports.

## Registry-driven runtime contract

Nodes declare stable type/version, config schema, typed ports, dependency/provider requirements, execution semantics, modality/language claims with evidence, validation rules, and limitations. Adapters implement the contract; graph validation does not infer support from names. Workflow documents are declarative and strict; allow-listed node types only. Template executable state requires real adapter bindings, compatible graph, configuration, and recorded end-to-end evidence.

## Main risks and decisions

Graph version evolution and adapter compatibility; provider/license sprawl; Python dependency conflicts; private data and prompt injection; provenance preservation; costs and resource bounds; agent loops; time-sensitive catalog claims. M1 defaults above close the local database/vector/provider/development choices sufficiently to begin implementation. Exact model names, supported document formats, deployment authentication beyond loopback, and production hosting remain deferred. Traditional-vs-Naive wording is defined above. RagSpark is identified but remains under evaluation pending source/metadata/runtime inspection.
