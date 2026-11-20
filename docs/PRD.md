# Visual RAG Studio — Product Requirements

**Status:** Phase 0 baseline, 2026-10-25  
**Scope:** Product definition and first implementation milestone. This document records requirements, not shipped functionality.

## Product goal

Visual RAG Studio is a visual, extensible RAG architecture builder for developers and learners. Users select and customize verified architecture templates or assemble workflows on a blank canvas, configure components, execute workflows, and inspect evidence and intermediate results. Reuse proven open-source functions where their actual APIs and semantics fit; own the visual editor, graph contract, compatibility model, registry, product persistence, validation, and execution boundary.

## Architecture Gallery requirement

The Gallery is a catalog of architecture families and composable patterns. Entries are searchable, filterable by category and capability, and explain their flow, purpose, required components/dependencies, modalities/languages, configuration, validation, limitations, implementation provider, and integration/execution status. A preview is distinct from an executable template. Only validated, runnable templates may be instantiated as executable workflows. Planned or unverified entries stay visibly marked **Planned**.

Initial catalog (37 named entries; five foundational, fifteen retrieval-enhanced, seven agentic/corrective, eight multilingual/specialized, and seven production/advanced patterns):

| Category | Catalog entries |
|---|---|
| Foundational | Traditional RAG, Naive RAG, Advanced RAG, Modular RAG, Custom RAG |
| Retrieval-enhanced | Hybrid RAG, Reranking RAG, Sentence-Window RAG, Parent-Document RAG, Hierarchical RAG, Multi-Query RAG, Query-Routing RAG, Metadata-Filtered RAG, Graph RAG, Late-Interaction RAG |
| Agentic and corrective | Agentic RAG, Corrective RAG (CRAG), Self-RAG, Adaptive RAG, Iterative RAG, Feedback-Driven RAG, Multi-Agent RAG |
| Multilingual and specialized | Multilingual RAG, Cross-Lingual RAG, Multimodal RAG, Vision RAG, Table RAG, Code RAG, Temporal RAG, Domain-Specific RAG |
| Production and advanced patterns | Long-Context RAG, Contextual RAG, Knowledge-Graph-Enhanced RAG, SQL/Text-to-SQL RAG, Federated RAG, Streaming RAG, RAG with Verification |

These labels may overlap or denote alternative design patterns rather than exclusive whole-system architectures. Represent composition as verified component capabilities and execution semantics, not as claims that every pair is compatible. Custom RAG opens a blank canvas. Traditional RAG is the explanatory baseline: an end-to-end sequence of loading, chunking, embedding/indexing, retrieval, context construction, and grounded generation. Naive RAG is a separate simpler pattern: direct top-k retrieval followed by generation, with minimal query/context processing. The two can share nodes and adapters, but their explanations and graph presets must expose the distinction. User workflows can branch, compare settings, replace compatible components, and be saved as reusable templates.

## Product requirements

1. Discover catalog entries by search and category/capability/status filters; show previews, explanations, requirements, limitations and provider status.
2. Instantiate a blank canvas or an executable template; preserve template identity and allow edits.
3. Add, connect, configure, move, remove, branch and compare components. Validate port and semantic compatibility before execution.
4. Save and reopen user workflows and templates; import/export only strictly validated declarative graph documents.
5. Execute supported graphs through registered implementations with explicit provider configuration, bounded resources, per-node outputs/errors and traceable source provenance.
6. Support local and cloud model adapters without silent provider substitution; never execute code supplied by imported workflows.
7. Show absent or incomplete implementations as planned. Do not assert a library supports a pattern based on its general RAG positioning.
8. Preserve original sources and provenance through retrieval and answer citation.

## First milestone (M1): registry and working baseline

Establish a schema-versioned catalog/capability registry covering all 37 entries and a small, fully verified executable baseline set. M1's vertical slice is Traditional RAG. Keep Naive RAG as a distinct planned pattern using the definition above, not a duplicate executable template. Custom RAG starts blank and becomes executable only after valid nodes are assembled. The initial runtime may support a minimal deterministic linear pipeline. All other entries are **Planned** unless verified and implemented.

### Acceptance criteria

- All 37 entries are present with stable IDs, categories, searchable descriptions and explicit Planned/Preview/Executable state.
- Registry schema describes required components, dependencies, execution model, input/output contracts, modalities, languages, config schema, validation rules, provider/integration evidence, and known limitations.
- Gallery supports category/search/status filtering, informative details and a topology preview for every catalog entry; planned entries cannot be launched as executable graphs.
- Traditional baseline instantiates as a valid graph; blank Custom canvas is available.
- Every executable template validates and runs end-to-end against a deterministic fixture or configured real provider; evidence is recorded. No test result is presumed by this document.
- Workflow edits can replace compatible nodes and fork a configuration for comparison; invalid composition reports actionable incompatibilities.
- Saves and loads versioned declarative graph definitions; import rejects unknown node types, invalid schemas, and executable-code fields.
- Provider and integration status is recorded per capability and reflects tested reality.

## Later scope

Broader retrieval, agentic, multilingual, multimodal, evaluation, trace comparison, deployment, access control and multi-provider integrations are incremental milestones, selected by demonstrated demand and verified implementations. No target completion date is implied.

## Assumptions and unresolved decisions

- The user-provided Phase 0 brief and follow-up clarifications are authoritative product direction; no application code or existing data was found in the target directory.
- M1 working defaults: SQLite metadata, persistent local Chroma, Ollama embedding and generation adapters, local separate frontend/backend processes, single-user self-hosted deployment assumption. Loopback development may omit auth with warning; non-loopback/shared use requires an authentication boundary. See `docs/architecture.md` and `docs/security.md`.
- Traditional RAG and Naive RAG are separate concepts, with M1 definitions in this document and registry. Only Traditional RAG is targeted as runnable in M1.
- RagSpark identity is resolved to `Adeebkhann/ragspark`; README/reference claims are recorded, but package metadata/source behavior remain unverified, so it is not integrated pending audit.
- Exact provider model identifiers and any shared/production deployment model remain deferred; neither blocks the local M1 contract.
- Current source and package versions change rapidly; pin only after resolving the target Python runtime and checking compatible dependency sets.
