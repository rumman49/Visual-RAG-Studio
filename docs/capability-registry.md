# Capability Registry — Specification and Initial Catalog

**Assessment date:** 2026-10-25. **Status:** specification; no capabilities are implemented by this artifact.

## Registry contract

Catalog entries describe architecture families and composable patterns. An architecture label is not an implementation or a mutually exclusive workflow. Each entry has a stable `id`, display name, category, purpose, implementation status, composition facets, and optional template graph reference.

### Separate registry concepts and status meanings

- **Catalog entry:** explanation and composition metadata for an architecture family. This does not imply a workflow exists.
- **Architecture template:** an instantiable graph definition, which may remain planned until its graph and configs are complete.
- **Node type:** reusable component contract, including ports and config schema.
- **Provider adapter:** product-owned binding to an actual provider API and dependency.
- **Configured provider instance:** user/environment-specific adapter configuration; secret values are never stored in graph config.
- **Implemented capability:** a real runtime implementation is registered for the contract.
- **Tested executable template:** a complete compatible graph with required settings whose execution behavior has recorded test evidence.

Statuses are independent of catalog presence: `planned` means metadata/design only; `experimental` means a partial or limited implementation exists and must not be treated as generally executable; `implemented` means runtime code and provider/dependency bindings exist but execution evidence is incomplete; `tested` means the capability passed its specified tests with evidence recorded. A planned/experimental node cannot pass executable validation. A template is executable only when its graph validates and every required node/provider is implemented and verified. The API may return `valid: true` for a structurally valid planned graph while `executable: false`.

Each capability/provider binding records:

- required component capability IDs and graph input/output port schemas (type, modality, cardinality, provenance guarantees);
- dependencies with distribution/module, version constraints, license/source, optionality, and installation/integration state;
- execution model (`deterministic`, `branching`, `looping`, `streaming`, `agentic`, or explicit combination), state/checkpoint requirements, side effects, and resource bounds;
- supported input/output modalities and languages, with evidence scope (generation language is not cross-lingual retrieval evidence);
- JSON Schema configuration, defaults, secret references (never secret values), and validation rules;
- provider identity, API/function actually invoked, verification evidence and date, status, and known limitations;
- template graph contract, required configuration, and executed verification evidence before status can be executable.

Compatibility is a relation between concrete ports and execution semantics, not between catalog labels. A validator must check type/modality/cardinality, provenance needs, embedding dimensions, storage metric/index compatibility, dependency availability, config constraints, and control-flow termination/state requirements. Unknown evidence means unsupported/unverified, not compatible by assumption. Separate `catalog_status`, `provider_status`, and `template_status`.

## Catalog coverage and status

All entries below are **Planned** pending implementation and API verification. Provider is `TBD / unverified`; no framework is assigned by association. Category tags support multi-facet filtering. Detailed per-entry requirements and limitations are discovered as capabilities are selected for milestones.

| Stable ID | Name | Category | Initial composition facets |
|---|---|---|---|
| traditional-rag | Traditional RAG | Foundational | ingest, chunk, embed, retrieve, generate |
| naive-rag | Naive RAG | Foundational | direct top-k retrieve, generate; minimal query/context transforms |
| advanced-rag | Advanced RAG | Foundational | composable query/retrieval/context |
| modular-rag | Modular RAG | Foundational | replaceable components |
| custom-rag | Custom RAG | Foundational | blank canvas, user composition |
| hybrid-rag | Hybrid RAG | Retrieval-enhanced | dense+sparse, fusion |
| reranking-rag | Reranking RAG | Retrieval-enhanced | retrieve, rerank |
| sentence-window-rag | Sentence-Window RAG | Retrieval-enhanced | sentence index, window expansion |
| parent-document-rag | Parent-Document RAG | Retrieval-enhanced | child lookup, parent expansion |
| hierarchical-rag | Hierarchical RAG | Retrieval-enhanced | hierarchy, traversal |
| multi-query-rag | Multi-Query RAG | Retrieval-enhanced | query fan-out, merge |
| query-routing-rag | Query-Routing RAG | Retrieval-enhanced | classification, conditional routing |
| metadata-filtered-rag | Metadata-Filtered RAG | Retrieval-enhanced | validated metadata predicates |
| graph-rag | Graph RAG | Retrieval-enhanced | graph retrieval (definition/provider TBD) |
| late-interaction-rag | Late-Interaction RAG | Retrieval-enhanced | token-level/multi-vector ranking |
| agentic-rag | Agentic RAG | Agentic and corrective | bounded agent control flow |
| corrective-rag | Corrective RAG (CRAG) | Agentic and corrective | retrieval assessment, correction |
| self-rag | Self-RAG | Agentic and corrective | self-critique/reflection gates |
| adaptive-rag | Adaptive RAG | Agentic and corrective | complexity routing |
| iterative-rag | Iterative RAG | Agentic and corrective | bounded retrieve/refine loop |
| feedback-driven-rag | Feedback-Driven RAG | Agentic and corrective | feedback capture/update policy |
| multi-agent-rag | Multi-Agent RAG | Agentic and corrective | multi-actor orchestration |
| multilingual-rag | Multilingual RAG | Multilingual and specialized | language-aware end-to-end path |
| cross-lingual-rag | Cross-Lingual RAG | Multilingual and specialized | cross-language retrieval |
| multimodal-rag | Multimodal RAG | Multilingual and specialized | multiple modalities |
| vision-rag | Vision RAG | Multilingual and specialized | visual source parsing/retrieval |
| table-rag | Table RAG | Multilingual and specialized | structured table extraction/query |
| code-rag | Code RAG | Multilingual and specialized | code parsing/chunk/retrieval |
| temporal-rag | Temporal RAG | Multilingual and specialized | time-aware metadata/retrieval |
| domain-specific-rag | Domain-Specific RAG | Multilingual and specialized | domain constraints/schema |
| long-context-rag | Long-Context RAG | Production and advanced patterns | context sizing/selection |
| contextual-rag | Contextual RAG | Production and advanced patterns | contextual enrichment (definition TBD) |
| knowledge-graph-enhanced-rag | Knowledge-Graph-Enhanced RAG | Production and advanced patterns | graph + document retrieval |
| sql-text-to-sql-rag | SQL/Text-to-SQL RAG | Production and advanced patterns | structured query generation/execution |
| federated-rag | Federated RAG | Production and advanced patterns | multi-source retrieval/security |
| streaming-rag | Streaming RAG | Production and advanced patterns | incremental events/tokens |
| rag-with-verification | RAG with Verification | Production and advanced patterns | answer/evidence verification |

This table contains all 37 requested names across the five categories. The originating user list contains 37 names (5 foundational + 15 retrieval-enhanced + 7 agentic/corrective + 8 multilingual/specialized + 7 production/advanced). For M1, Traditional RAG means a transparent end-to-end ingestion-to-generation workflow. Naive RAG is a separate, simpler direct top-k retrieve-to-generate pattern; shared nodes are allowed, but distinct copy and topology should make the distinction visible. Only Traditional RAG is targeted as executable in M1.

## Initial component capability families

Ingestion/parsing; OCR/table/metadata extraction; chunking; embedding; vector and sparse indexes; dense/sparse retrieval; fusion; reranking; query rewrite/translation/classification/routing; metadata filtering; context expansion/assembly; graph and SQL retrieval; generation; citation/provenance; verification; evaluation; observability; control-flow (branch, merge, loop, retry, stream, checkpoint, human approval).

## Story 2 registry implementation status

`backend/app/registry.py` is the source of truth for the 37 architecture entries, six priority template records, blank-canvas catalog item, and initial reusable node contracts. All architecture and template statuses are `planned`; node definitions have no runtime identifiers; provider adapter list is empty. Chunking strategy identifiers are catalog metadata only. This story adds no execution behavior.

## M1 status gate

Mark a capability or template executable only after its graph schema and config validate, required real dependencies/providers are available, compatible inputs and outputs are checked, and execution behavior is verified with recorded evidence. Provider absence/configuration must be explicit; test doubles are labeled and isolated. A preview may show an illustrative flow but is never runnable. A blank Custom canvas is a starting graph, not a working RAG template.
