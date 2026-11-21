"""Product-owned capability catalog and node schemas (metadata only in Story 2)."""

from __future__ import annotations

from app.contracts import (
    ArchitectureCatalogEntry,
    ArchitectureTemplateDefinition,
    CapabilityEvidence,
    CatalogEntryKind,
    ChunkingStrategyDefinition,
    ConfigFieldDefinition,
    ConfigValueType,
    ExecutionContract,
    ExecutionMode,
    GraphConnection,
    ImplementationStatus,
    InputPortDefinition,
    NodeConfigSchema,
    NodeDefinition,
    NodeInstance,
    OutputPortDefinition,
    PortCardinality,
    PortDataType,
    ProviderAdapterDefinition,
    RegistrySnapshot,
    WorkflowGraph,
)

_CATALOG = [
    (
        "traditional-rag",
        "Traditional RAG",
        "Foundational",
        "ingest, chunk, embed, retrieve, generate",
    ),
    (
        "naive-rag",
        "Naive RAG",
        "Foundational",
        "direct top-k retrieve, generate; minimal query/context transforms",
    ),
    ("advanced-rag", "Advanced RAG", "Foundational", "composable query/retrieval/context"),
    ("modular-rag", "Modular RAG", "Foundational", "replaceable components"),
    ("custom-rag", "Custom RAG", "Foundational", "blank canvas, user composition"),
    ("hybrid-rag", "Hybrid RAG", "Retrieval-enhanced", "dense+sparse, fusion"),
    ("reranking-rag", "Reranking RAG", "Retrieval-enhanced", "retrieve, rerank"),
    (
        "sentence-window-rag",
        "Sentence-Window RAG",
        "Retrieval-enhanced",
        "sentence index, window expansion",
    ),
    (
        "parent-document-rag",
        "Parent-Document RAG",
        "Retrieval-enhanced",
        "child lookup, parent expansion",
    ),
    ("hierarchical-rag", "Hierarchical RAG", "Retrieval-enhanced", "hierarchy, traversal"),
    ("multi-query-rag", "Multi-Query RAG", "Retrieval-enhanced", "query fan-out, merge"),
    (
        "query-routing-rag",
        "Query-Routing RAG",
        "Retrieval-enhanced",
        "classification, conditional routing",
    ),
    (
        "metadata-filtered-rag",
        "Metadata-Filtered RAG",
        "Retrieval-enhanced",
        "validated metadata predicates",
    ),
    ("graph-rag", "Graph RAG", "Retrieval-enhanced", "graph retrieval"),
    (
        "late-interaction-rag",
        "Late-Interaction RAG",
        "Retrieval-enhanced",
        "token-level/multi-vector ranking",
    ),
    ("agentic-rag", "Agentic RAG", "Agentic and corrective", "bounded agent control flow"),
    (
        "corrective-rag",
        "Corrective RAG (CRAG)",
        "Agentic and corrective",
        "retrieval assessment, correction",
    ),
    ("self-rag", "Self-RAG", "Agentic and corrective", "self-critique/reflection gates"),
    ("adaptive-rag", "Adaptive RAG", "Agentic and corrective", "complexity routing"),
    ("iterative-rag", "Iterative RAG", "Agentic and corrective", "bounded retrieve/refine loop"),
    (
        "feedback-driven-rag",
        "Feedback-Driven RAG",
        "Agentic and corrective",
        "feedback capture/update policy",
    ),
    ("multi-agent-rag", "Multi-Agent RAG", "Agentic and corrective", "multi-actor orchestration"),
    (
        "multilingual-rag",
        "Multilingual RAG",
        "Multilingual and specialized",
        "language-aware end-to-end path",
    ),
    (
        "cross-lingual-rag",
        "Cross-Lingual RAG",
        "Multilingual and specialized",
        "cross-language retrieval",
    ),
    ("multimodal-rag", "Multimodal RAG", "Multilingual and specialized", "multiple modalities"),
    ("vision-rag", "Vision RAG", "Multilingual and specialized", "visual source parsing/retrieval"),
    ("table-rag", "Table RAG", "Multilingual and specialized", "structured table extraction/query"),
    ("code-rag", "Code RAG", "Multilingual and specialized", "code parsing/chunk/retrieval"),
    (
        "temporal-rag",
        "Temporal RAG",
        "Multilingual and specialized",
        "time-aware metadata/retrieval",
    ),
    (
        "domain-specific-rag",
        "Domain-Specific RAG",
        "Multilingual and specialized",
        "domain constraints/schema",
    ),
    (
        "long-context-rag",
        "Long-Context RAG",
        "Production and advanced patterns",
        "context sizing/selection",
    ),
    (
        "contextual-rag",
        "Contextual RAG",
        "Production and advanced patterns",
        "contextual enrichment",
    ),
    (
        "knowledge-graph-enhanced-rag",
        "Knowledge-Graph-Enhanced RAG",
        "Production and advanced patterns",
        "graph + document retrieval",
    ),
    (
        "sql-text-to-sql-rag",
        "SQL/Text-to-SQL RAG",
        "Production and advanced patterns",
        "structured query generation/execution",
    ),
    (
        "federated-rag",
        "Federated RAG",
        "Production and advanced patterns",
        "multi-source retrieval/security",
    ),
    (
        "streaming-rag",
        "Streaming RAG",
        "Production and advanced patterns",
        "incremental events/tokens",
    ),
    (
        "rag-with-verification",
        "RAG with Verification",
        "Production and advanced patterns",
        "answer/evidence verification",
    ),
]

PRIORITY_TEMPLATE_IDS = (
    "traditional-rag",
    "hybrid-rag",
    "reranking-rag",
    "multilingual-rag",
    "agentic-rag",
    "multimodal-rag",
)
_STRATEGIES = [
    ("recursive", "Recursive"),
    ("fixed", "Fixed Size"),
    ("sentence", "Sentence Aware"),
    ("structure_aware", "Structure Aware"),
    ("auto", "Auto-Detect"),
    ("token", "Token Aware"),
    ("paragraph", "Paragraph Aware"),
    ("markdown", "Markdown Aware"),
    ("semantic", "Semantic"),
    ("sliding_window", "Sliding Window"),
    ("parent_child", "Parent-Child"),
    ("legal_clause", "Legal Clause Hierarchy"),
    ("table_aware", "Table Aware"),
    ("code_aware", "Code Aware"),
    ("contextual", "Contextual"),
    ("adaptive", "Adaptive"),
    ("hierarchical", "Hierarchical"),
]


def _entry(item: tuple[str, str, str, str]) -> ArchitectureCatalogEntry:
    identifier, name, category, facets = item
    return ArchitectureCatalogEntry(
        id=identifier,
        name=name,
        category=category,
        description=f"{name} capability family: {facets}.",
        implementation_status=ImplementationStatus.PLANNED,
        composition_facets=[part.strip() for part in facets.split(",")],
        provider_status="unverified",
        known_limitations=["Catalog metadata only; no executable template or runtime is registered."],
        template_id=identifier if identifier in PRIORITY_TEMPLATE_IDS else None,
    )


def _port(
    name: str,
    data_type: PortDataType,
    required: bool = True,
    cardinality: PortCardinality = PortCardinality.ONE,
) -> InputPortDefinition:
    return InputPortDefinition(
        id=name,
        display_name=name.replace("_", " ").title(),
        data_type=data_type,
        required=required,
        cardinality=cardinality,
    )


def _out(name: str, data_type: PortDataType) -> OutputPortDefinition:
    return OutputPortDefinition(
        id=name, display_name=name.replace("_", " ").title(), data_type=data_type
    )


def _node(
    type_id: str,
    name: str,
    category: str,
    inputs: list[InputPortDefinition],
    outputs: list[OutputPortDefinition],
    schema: NodeConfigSchema | None = None,
    limitations: list[str] | None = None,
    runtime_id: str | None = None,
    provider_requirements: list[str] | None = None,
) -> NodeDefinition:
    return NodeDefinition(
        node_type_id=type_id,
        display_name=name,
        category=category,
        description=f"Reusable {name.lower()} component contract.",
        inputs=inputs,
        outputs=outputs,
        config_schema=schema or NodeConfigSchema(),
        implementation_status=ImplementationStatus.IMPLEMENTED
        if runtime_id
        else ImplementationStatus.PLANNED,
        runtime_implementation_id=runtime_id,
        provider_requirements=provider_requirements or [],
        known_limitations=limitations or ["Schema only; no runtime implementation is registered."],
    )



def build_registry() -> RegistrySnapshot:
    implemented_architectures = {
        "traditional-rag",
        "hybrid-rag",
        "reranking-rag",
        "multilingual-rag",
        "agentic-rag",
    }
    catalog = [
        _entry(item).model_copy(
            update={
                "implementation_status": ImplementationStatus.IMPLEMENTED
                if item[0] in implemented_architectures
                else ImplementationStatus.PLANNED,
                "known_limitations": []
                if item[0] in implemented_architectures
                else ["Catalog metadata only; no executable template or runtime is registered."],
            }
        )
        for item in _CATALOG
    ]
    catalog.append(
        ArchitectureCatalogEntry(
            id="start-from-blank-canvas",
            name="Start from Blank Canvas",
            category="Foundational",
            kind=CatalogEntryKind.BLANK_CANVAS,
            description="Start an editable workflow with no preselected architecture or nodes.",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            provider_status="not_applicable",
            known_limitations=[],
        )
    )

    templates = [
        ArchitectureTemplateDefinition(
            id=identifier,
            architecture_id=identifier,
            name=next(row[1] for row in _CATALOG if row[0] == identifier),
            implementation_status=ImplementationStatus.PLANNED,
            graph=None,
            known_limitations=["Planned template; no runnable graph is defined in Story 2."],
        )
        for identifier in PRIORITY_TEMPLATE_IDS
    ]

    chunk_strategies_enum = [
        "recursive",
        "fixed",
        "sentence",
        "structure_aware",
        "hierarchical",
        "auto",
        "token",
        "paragraph",
        "markdown",
        "semantic",
        "sliding_window",
        "parent_child",
        "legal_clause",
        "table_aware",
        "code_aware",
        "contextual",
        "adaptive",
    ]
    chunk_schema = NodeConfigSchema(
        properties={
            "strategy": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Chunking strategy used by indexing.",
                required=True,
                enum=chunk_strategies_enum,
            ),
            "chunk_size": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="Target chunk size in characters.",
                minimum=64,
                maximum=8000,
                default=1000,
            ),
            "overlap": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="Chunk overlap.",
                minimum=0,
                maximum=7999,
                default=120,
            ),
        }
    )
    embedding_schema = NodeConfigSchema(
        properties={
            "provider": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Embedding adapter identifier.",
                enum=["sentence_transformers", "fast_local", "ollama", "gemini", "openai", "huggingface"],
                default="sentence_transformers",
            ),
            "strategy": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Embedding strategy.",
                enum=["dense", "hybrid_sparse_dense", "legal_domain_boost", "context_enriched"],
                default="dense",
            ),
            "model": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Free local or hosted embedding model name.",
                enum=[
                    # Multilingual Models (384 Dimensions)
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "intfloat/multilingual-e5-small",
                    # Multilingual Models (768 Dimensions)
                    "intfloat/multilingual-e5-base",
                    # Multilingual Models (1024 Dimensions)
                    "BAAI/bge-m3",
                    "intfloat/multilingual-e5-large",
                    # English / General Models (384 Dimensions)
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "BAAI/bge-small-en-v1.5",
                    # English / General Models (768 Dimensions)
                    "sentence-transformers/all-mpnet-base-v2",
                    "BAAI/bge-base-en-v1.5",
                    "nomic-ai/nomic-embed-text-v1.5",
                    # English / General Models (1024 Dimensions)
                    "mixedbread-ai/mxbai-embed-large",
                    # Local Ollama Models
                    "nomic-embed-text",
                    "all-minilm",
                    "bge-m3",
                    # Cloud Free-tier / Hosted Models
                    "models/text-embedding-004",
                    "text-embedding-3-small",
                    "text-embedding-3-large",
                ],
                default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            "dimension": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="Vector dimension (automatically calibrated to model).",
                minimum=64,
                maximum=4096,
                default=384,
            ),
            "supported_languages": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Supported language coverage of the embedding model.",
                enum=["multilingual", "english_only", "bilingual"],
                default="multilingual",
            ),
        }
    )
    vector_index_schema = NodeConfigSchema(
        properties={
            "backend": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Vector database backend.",
                enum=["chroma", "memory", "pinecone", "mongodb", "qdrant"],
                default="chroma",
            ),
            "collection_name": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Collection name for vector storage.",
                default="visual_rag_studio_documents",
            ),
            "url": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Host URL (hardcoded default for local Chroma or remote cluster).",
                default="http://localhost:8000",
            ),
            "api_key": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Not accepted in graphs. Set the backend environment variable (e.g. PINECONE_API_KEY) or save the key via /api/v1/providers/credentials/{backend}.",
                default="",
            ),
        }
    )
    retrieve_schema = NodeConfigSchema(
        properties={
            "strategy": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Retrieval strategy: semantic (dense vector), bm25 (sparse lexical), or hybrid (dense + sparse fusion).",
                enum=["semantic", "hybrid", "bm25", "rrf"],
                default="semantic",
            ),
            "top_k": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="Final number of retrieved chunks to return.",
                minimum=1,
                maximum=100,
                default=5,
            ),
            "fusion_method": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Method used to fuse dense and sparse rankings in hybrid mode.",
                enum=["rrf", "weighted"],
                default="rrf",
            ),
            "rrf_k": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="Reciprocal Rank Fusion smoothing constant (k). Standard default is 60.",
                minimum=1,
                maximum=200,
                default=60,
            ),
            "alpha": ConfigFieldDefinition(
                value_type=ConfigValueType.NUMBER,
                description="Weight balancing factor for hybrid retrieval (1.0 = pure dense, 0.0 = pure sparse BM25).",
                minimum=0.0,
                maximum=1.0,
                default=0.65,
            ),
            "dense_top_k": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="Number of candidate chunks to retrieve from dense semantic search before fusion.",
                minimum=1,
                maximum=100,
                default=10,
            ),
            "sparse_top_k": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="Number of candidate chunks to retrieve from sparse BM25 search before fusion.",
                minimum=1,
                maximum=100,
                default=10,
            ),
            "k1": ConfigFieldDefinition(
                value_type=ConfigValueType.NUMBER,
                description="BM25 term frequency saturation parameter k1.",
                minimum=0.1,
                maximum=5.0,
                default=1.5,
            ),
            "b": ConfigFieldDefinition(
                value_type=ConfigValueType.NUMBER,
                description="BM25 document length normalization parameter b.",
                minimum=0.0,
                maximum=1.0,
                default=0.75,
            ),
            "query_language_handling": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Language handling mode for query processing.",
                enum=["auto", "direct", "detect"],
                default="auto",
            ),
            "cross_lingual": ConfigFieldDefinition(
                value_type=ConfigValueType.BOOLEAN,
                description="Enable cross-lingual vector search across language boundaries.",
                default=True,
            ),
        }
    )
    sparse_retrieve_schema = NodeConfigSchema(
        properties={
            "top_k": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="Number of lexical chunks to retrieve.",
                minimum=1,
                maximum=100,
                default=10,
            ),
            "k1": ConfigFieldDefinition(
                value_type=ConfigValueType.NUMBER,
                description="BM25 term frequency saturation parameter k1.",
                minimum=0.1,
                maximum=5.0,
                default=1.5,
            ),
            "b": ConfigFieldDefinition(
                value_type=ConfigValueType.NUMBER,
                description="BM25 document length normalization parameter b.",
                minimum=0.0,
                maximum=1.0,
                default=0.75,
            ),
            "query_language_handling": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Language handling mode for query processing.",
                enum=["auto", "direct", "detect"],
                default="auto",
            ),
            "cross_lingual": ConfigFieldDefinition(
                value_type=ConfigValueType.BOOLEAN,
                description="Enable cross-lingual vector search across language boundaries.",
                default=True,
            ),
        }
    )
    fusion_schema = NodeConfigSchema(
        properties={
            "fusion_method": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Algorithm for fusing multiple candidate lists.",
                enum=["rrf", "weighted"],
                default="rrf",
            ),
            "rrf_k": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="RRF smoothing constant k.",
                minimum=1,
                maximum=200,
                default=60,
            ),
            "dense_weight": ConfigFieldDefinition(
                value_type=ConfigValueType.NUMBER,
                description="Weight for dense semantic ranking.",
                minimum=0.0,
                maximum=1.0,
                default=0.65,
            ),
            "sparse_weight": ConfigFieldDefinition(
                value_type=ConfigValueType.NUMBER,
                description="Weight for sparse lexical ranking.",
                minimum=0.0,
                maximum=1.0,
                default=0.35,
            ),
            "top_k": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="Number of fused chunks to pass to context assembly.",
                minimum=1,
                maximum=50,
                default=5,
            ),
        }
    )
    rerank_schema = NodeConfigSchema(
        properties={
            "model": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Reranker model.",
                enum=["flash", "cross_encoder"],
                default="flash",
            ),
            "top_k": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="Number of top candidates to return.",
                minimum=1,
                maximum=50,
                default=5,
            ),
        }
    )
    context_schema = NodeConfigSchema(
        properties={
            "technique": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Prompt engineering technique.",
                enum=["standard", "natural_law", "cot", "few_shot", "citation_strict", "legal_irac", "strict_irac", "concise"],
                default="standard",
            ),
            "max_chars": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="Maximum context characters.",
                minimum=500,
                maximum=50000,
                default=12000,
            ),
            "preserve_source_language": ConfigFieldDefinition(
                value_type=ConfigValueType.BOOLEAN,
                description="Preserve the original language and script of retrieved source passages in context.",
                default=True,
            ),
            "response_language": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Target language for grounded generation instructions.",
                enum=["query", "source", "en", "hi", "hinglish", "es", "fr", "de", "auto"],
                default="query",
            ),
        }
    )
    generation_schema = NodeConfigSchema(
        properties={
            "provider": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="LLM provider.",
                enum=["groq", "ollama", "huggingface", "extractive", "gemini", "openai", "anthropic"],
                default="groq",
            ),
            "model": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Model ID or name.",
                default="openai/gpt-oss-20b",
            ),
            "response_language": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Target response language for grounded answer generation.",
                enum=["query", "source", "en", "hi", "hinglish", "es", "fr", "de", "auto"],
                default="query",
            ),
            "temperature": ConfigFieldDefinition(
                value_type=ConfigValueType.NUMBER,
                description="Generation temperature.",
                minimum=0.0,
                maximum=2.0,
                default=0.2,
            ),
            "max_tokens": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="Maximum generation tokens.",
                minimum=1,
                maximum=8192,
                default=512,
            ),
        }
    )
    agent_schema = NodeConfigSchema(
        properties={
            "goal": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="High-level objective the agent must achieve.",
                default="Answer the query factually based on retrieved evidence.",
            ),
            "instructions": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Specific behavioural directives, constraints, or fallback instructions.",
                default="",
            ),
            "mode": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="Agent execution control flow mode.",
                enum=["fixed", "conditional", "bounded_loop"],
                default="bounded_loop",
            ),
            "max_steps": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="Maximum number of reasoning and tool execution steps.",
                minimum=1,
                maximum=20,
                default=5,
            ),
            "max_retrieval_iterations": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="Maximum retrieval attempts before forcing answer or abstention.",
                minimum=1,
                maximum=10,
                default=3,
            ),
            "token_budget": ConfigFieldDefinition(
                value_type=ConfigValueType.INTEGER,
                description="Cumulative token budget ceiling for LLM reasoning and generation.",
                minimum=512,
                maximum=32768,
                default=4096,
            ),
            "timeout_seconds": ConfigFieldDefinition(
                value_type=ConfigValueType.NUMBER,
                description="Maximum execution timeout in seconds.",
                minimum=1.0,
                maximum=300.0,
                default=60.0,
            ),
            "approval_policy": ConfigFieldDefinition(
                value_type=ConfigValueType.STRING,
                description="User authorization policy for sensitive or uncertain actions.",
                enum=["always_ask", "on_uncertainty", "fully_autonomous"],
                default="on_uncertainty",
            ),
        }
    )
    nodes = [
        _node(
            "document.source",
            "Document Source",
            "Ingestion",
            [],
            [_out("documents", PortDataType.DOCUMENT_SET)],
            runtime_id="rag.document_source.v1",
        ),
        _node(
            "document.parse",
            "Document Parser",
            "Ingestion",
            [_port("documents", PortDataType.DOCUMENT_SET)],
            [_out("parsed_documents", PortDataType.PARSED_DOCUMENTS)],
            runtime_id="rag.parser.v1",
        ),
        _node(
            "chunk.split",
            "Chunk Documents",
            "Ingestion",
            [_port("parsed_documents", PortDataType.PARSED_DOCUMENTS)],
            [_out("chunks", PortDataType.CHUNKS)],
            chunk_schema,
            runtime_id="rag.chunking.v1",
        ),
        _node(
            "embedding.create",
            "Create Embeddings",
            "Indexing",
            [_port("chunks", PortDataType.CHUNKS)],
            [_out("embeddings", PortDataType.EMBEDDINGS)],
            embedding_schema,
            runtime_id="rag.embedding_provider.v1",
            provider_requirements=["sentence-transformers.embedding", "ollama.embedding", "ragspark.fast_local.embedding"],
        ),
        _node(
            "vector.index",
            "Vector Index",
            "Indexing",
            [_port("embeddings", PortDataType.EMBEDDINGS)],
            [_out("index", PortDataType.VECTOR_INDEX)],
            vector_index_schema,
            runtime_id="rag.chroma_index.v1",
            provider_requirements=["chroma.persistent", "ragspark.memory.vector_store"],
        ),
        _node(
            "query.input",
            "Query Input",
            "Query",
            [],
            [_out("query", PortDataType.QUERY)],
            runtime_id="rag.query_input.v1",
        ),
        _node(
            "retrieval.dense",
            "Dense Retrieval",
            "Retrieval",
            [_port("index", PortDataType.VECTOR_INDEX), _port("query", PortDataType.QUERY)],
            [_out("retrieved_chunks", PortDataType.RETRIEVED_CHUNKS)],
            retrieve_schema,
            runtime_id="rag.chroma_dense_retrieval.v1",
        ),
        _node(
            "retrieval.sparse",
            "Sparse BM25 Retrieval",
            "Retrieval",
            [_port("index", PortDataType.VECTOR_INDEX), _port("query", PortDataType.QUERY)],
            [_out("retrieved_chunks", PortDataType.RETRIEVED_CHUNKS)],
            sparse_retrieve_schema,
            runtime_id="rag.bm25_sparse_retrieval.v1",
        ),
        _node(
            "merge.fuse",
            "Fusion & Ranking (RRF)",
            "Retrieval",
            [_port("retrieved_chunks", PortDataType.RETRIEVED_CHUNKS, cardinality=PortCardinality.MANY)],
            [_out("retrieved_chunks", PortDataType.RETRIEVED_CHUNKS)],
            fusion_schema,
            runtime_id="rag.reciprocal_rank_fusion.v1",
        ),
        _node(
            "context.assemble",
            "Assemble Context",
            "Generation",
            [_port("retrieved_chunks", PortDataType.RETRIEVED_CHUNKS)],
            [_out("context", PortDataType.CONTEXT)],
            context_schema,
            runtime_id="rag.context_assembly.v1",
        ),
        _node(
            "generation.answer",
            "Generate Answer",
            "Generation",
            [_port("query", PortDataType.QUERY), _port("context", PortDataType.CONTEXT)],
            [_out("answer", PortDataType.ANSWER)],
            generation_schema,
            limitations=[
                "Uses the selected generation provider; citation IDs are validated against retrieved chunks."
            ],
            runtime_id="rag.grounded_generation.v1",
            provider_requirements=[
                "groq.chat",
                "ollama.chat",
                "huggingface.chat",
                "ragspark.extractive.generation",
                "openai.chat",
                "gemini.chat",
                "anthropic.chat",
            ],
        ),
        _node(
            "retrieval.rerank",
            "Rerank Results",
            "Retrieval",
            [
                _port("query", PortDataType.QUERY),
                _port("retrieved_chunks", PortDataType.RETRIEVED_CHUNKS),
            ],
            [_out("retrieved_chunks", PortDataType.RETRIEVED_CHUNKS)],
            rerank_schema,
            provider_requirements=["ragspark.flash.reranker"],
        ),
        _node(
            "query.route",
            "Query Router",
            "Query",
            [_port("query", PortDataType.QUERY)],
            [_out("routed_query", PortDataType.QUERY)],
            limitations=["Planned routing node; no executable runtime is registered."],
        ),
        _node(
            "generation.verify",
            "Verify Answer",
            "Generation",
            [_port("answer", PortDataType.ANSWER), _port("context", PortDataType.CONTEXT)],
            [_out("verified_answer", PortDataType.ANSWER)],
            limitations=["Planned verification node; no executable runtime is registered."],
        ),
        _node(
            "retrieval.graph",
            "Graph Retrieval",
            "Retrieval",
            [_port("query", PortDataType.QUERY)],
            [_out("retrieved_chunks", PortDataType.RETRIEVED_CHUNKS)],
            limitations=["Planned graph retrieval; no executable runtime is registered."],
        ),
        _node(
            "agent.orchestrate",
            "Agent Orchestrator",
            "Agent",
            [
                _port("query", PortDataType.QUERY),
                _port("retrieved_chunks", PortDataType.RETRIEVED_CHUNKS, required=False),
                _port("context", PortDataType.CONTEXT, required=False),
            ],
            [
                _out("answer", PortDataType.ANSWER),
                _out("retrieved_chunks", PortDataType.RETRIEVED_CHUNKS),
                _out("state", PortDataType.TEXT),
            ],
            agent_schema,
            limitations=[
                "Bounded execution loop; allowlisted tools only; strict timeout and step limits."
            ],
            runtime_id="rag.agent.orchestrator.v1",
            provider_requirements=["ragspark.agent.orchestrator"],
        ),
    ]
    strategies = [
        ChunkingStrategyDefinition(
            id=identifier,
            display_name=label,
            implementation_status="implemented"
            if identifier in {"recursive", "fixed", "sentence", "structure_aware", "hierarchical"}
            else "planned",
            runtime_implementation_id=f"rag.chunking.{identifier}.v1"
            if identifier in {"recursive", "fixed", "sentence", "structure_aware", "hierarchical"}
            else None,
            config_schema=chunk_schema,
            known_limitations=[
                "Character-based chunk sizes; language-specific tokenization is not performed."
            ]
            if identifier in {"recursive", "fixed", "sentence", "structure_aware"}
            else [],
        )
        for identifier, label in _STRATEGIES
    ]

    template_graph = WorkflowGraph(
        id="traditional-rag-draft",
        name="Traditional RAG",
        architecture_id="traditional-rag",
        template_id="traditional-rag",
        nodes=[
            NodeInstance(
                id=node_id, node_type_id=type_id, config=config, provider_instance_id=provider_id
            )
            for node_id, type_id, config, provider_id in [
                ("source", "document.source", {}, None),
                ("parser", "document.parse", {}, None),
                (
                    "chunker",
                    "chunk.split",
                    {"strategy": "recursive", "chunk_size": 1000, "overlap": 120},
                    None,
                ),
                (
                    "embedder",
                    "embedding.create",
                    {
                        "provider": "sentence_transformers",
                        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        "dimension": 384,
                    },
                    "embedding-provider",
                ),
                (
                    "index",
                    "vector.index",
                    {
                        "backend": "chroma",
                        "collection_name": "visual_rag_studio_documents",
                        "url": "http://localhost:8000",
                    },
                    "chroma-local",
                ),
                ("query", "query.input", {}, None),
                ("retriever", "retrieval.dense", {"top_k": 5}, None),
                ("context", "context.assemble", {}, None),
                ("answer", "generation.answer", {}, "generation-provider"),
            ]
        ],
        connections=[
            GraphConnection(
                source_node_id="source",
                source_port_id="documents",
                target_node_id="parser",
                target_port_id="documents",
            ),
            GraphConnection(
                source_node_id="parser",
                source_port_id="parsed_documents",
                target_node_id="chunker",
                target_port_id="parsed_documents",
            ),
            GraphConnection(
                source_node_id="chunker",
                source_port_id="chunks",
                target_node_id="embedder",
                target_port_id="chunks",
            ),
            GraphConnection(
                source_node_id="embedder",
                source_port_id="embeddings",
                target_node_id="index",
                target_port_id="embeddings",
            ),
            GraphConnection(
                source_node_id="index",
                source_port_id="index",
                target_node_id="retriever",
                target_port_id="index",
            ),
            GraphConnection(
                source_node_id="query",
                source_port_id="query",
                target_node_id="retriever",
                target_port_id="query",
            ),
            GraphConnection(
                source_node_id="retriever",
                source_port_id="retrieved_chunks",
                target_node_id="context",
                target_port_id="retrieved_chunks",
            ),
            GraphConnection(
                source_node_id="query",
                source_port_id="query",
                target_node_id="answer",
                target_port_id="query",
            ),
            GraphConnection(
                source_node_id="context",
                source_port_id="context",
                target_node_id="answer",
                target_port_id="context",
            ),
        ],
    )
    rerank_template_graph = WorkflowGraph(
        id="reranking-rag-draft",
        name="Reranking RAG",
        architecture_id="reranking-rag",
        template_id="reranking-rag",
        nodes=[
            *[
                node.model_copy(
                    update={
                        "config": {
                            **node.config,
                            "dense_top_k": 15,
                        }
                    }
                )
                if node.id == "retriever"
                else node
                for node in template_graph.nodes
            ],
            NodeInstance(
                id="reranker",
                node_type_id="retrieval.rerank",
                config={"model": "flash", "top_k": 5},
                provider_instance_id="flash-reranker",
            ),
        ],
        connections=[
            *[
                c
                for c in template_graph.connections
                if not (c.source_node_id == "retriever" and c.target_node_id == "context")
            ],
            GraphConnection(
                source_node_id="retriever",
                source_port_id="retrieved_chunks",
                target_node_id="reranker",
                target_port_id="retrieved_chunks",
            ),
            GraphConnection(
                source_node_id="query",
                source_port_id="query",
                target_node_id="reranker",
                target_port_id="query",
            ),
            GraphConnection(
                source_node_id="reranker",
                source_port_id="retrieved_chunks",
                target_node_id="context",
                target_port_id="retrieved_chunks",
            ),
        ],
    )
    hybrid_template_graph = WorkflowGraph(
        id="hybrid-rag-draft",
        name="Hybrid RAG",
        architecture_id="hybrid-rag",
        template_id="hybrid-rag",
        nodes=[
            NodeInstance(id="source", node_type_id="document.source", config={}, provider_instance_id=None),
            NodeInstance(id="parser", node_type_id="document.parse", config={}, provider_instance_id=None),
            NodeInstance(id="chunker", node_type_id="chunk.split", config={"strategy": "recursive", "chunk_size": 1000, "overlap": 120}, provider_instance_id=None),
            NodeInstance(
                id="embedder",
                node_type_id="embedding.create",
                config={
                    "provider": "sentence_transformers",
                    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "dimension": 384,
                },
                provider_instance_id="embedding-provider",
            ),
            NodeInstance(
                id="index",
                node_type_id="vector.index",
                config={
                    "backend": "chroma",
                    "collection_name": "visual_rag_studio_documents",
                    "url": "http://localhost:8000",
                },
                provider_instance_id="chroma-local",
            ),
            NodeInstance(id="query", node_type_id="query.input", config={}, provider_instance_id=None),
            NodeInstance(id="dense_retriever", node_type_id="retrieval.dense", config={"strategy": "semantic", "top_k": 10}, provider_instance_id=None),
            NodeInstance(id="sparse_retriever", node_type_id="retrieval.sparse", config={"top_k": 10, "k1": 1.5, "b": 0.75}, provider_instance_id=None),
            NodeInstance(id="fuse", node_type_id="merge.fuse", config={"fusion_method": "rrf", "rrf_k": 60, "dense_weight": 0.65, "sparse_weight": 0.35}, provider_instance_id=None),
            NodeInstance(id="context", node_type_id="context.assemble", config={}, provider_instance_id=None),
            NodeInstance(id="answer", node_type_id="generation.answer", config={}, provider_instance_id="generation-provider"),
        ],
        connections=[
            GraphConnection(source_node_id="source", source_port_id="documents", target_node_id="parser", target_port_id="documents"),
            GraphConnection(source_node_id="parser", source_port_id="parsed_documents", target_node_id="chunker", target_port_id="parsed_documents"),
            GraphConnection(source_node_id="chunker", source_port_id="chunks", target_node_id="embedder", target_port_id="chunks"),
            GraphConnection(source_node_id="embedder", source_port_id="embeddings", target_node_id="index", target_port_id="embeddings"),
            GraphConnection(source_node_id="query", source_port_id="query", target_node_id="dense_retriever", target_port_id="query"),
            GraphConnection(source_node_id="index", source_port_id="index", target_node_id="dense_retriever", target_port_id="index"),
            GraphConnection(source_node_id="query", source_port_id="query", target_node_id="sparse_retriever", target_port_id="query"),
            GraphConnection(source_node_id="index", source_port_id="index", target_node_id="sparse_retriever", target_port_id="index"),
            GraphConnection(source_node_id="dense_retriever", source_port_id="retrieved_chunks", target_node_id="fuse", target_port_id="retrieved_chunks"),
            GraphConnection(source_node_id="sparse_retriever", source_port_id="retrieved_chunks", target_node_id="fuse", target_port_id="retrieved_chunks"),
            GraphConnection(source_node_id="fuse", source_port_id="retrieved_chunks", target_node_id="context", target_port_id="retrieved_chunks"),
            GraphConnection(source_node_id="query", source_port_id="query", target_node_id="answer", target_port_id="query"),
            GraphConnection(source_node_id="context", source_port_id="context", target_node_id="answer", target_port_id="context"),
        ],
    )
    multilingual_template_graph = WorkflowGraph(
        id="multilingual-rag-draft",
        name="Multilingual RAG",
        architecture_id="multilingual-rag",
        template_id="multilingual-rag",
        nodes=[
            NodeInstance(id="source", node_type_id="document.source", config={}, provider_instance_id=None),
            NodeInstance(id="parser", node_type_id="document.parse", config={}, provider_instance_id=None),
            NodeInstance(
                id="chunker",
                node_type_id="chunk.split",
                config={"strategy": "recursive", "chunk_size": 1000, "overlap": 120},
                provider_instance_id=None,
            ),
            NodeInstance(
                id="embedder",
                node_type_id="embedding.create",
                config={
                    "provider": "sentence_transformers",
                    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "dimension": 384,
                    "supported_languages": "multilingual",
                },
                provider_instance_id="embedding-provider",
            ),
            NodeInstance(
                id="index",
                node_type_id="vector.index",
                config={
                    "backend": "chroma",
                    "collection_name": "visual_rag_studio_documents",
                    "url": "http://localhost:8000",
                },
                provider_instance_id="chroma-local",
            ),
            NodeInstance(id="query", node_type_id="query.input", config={}, provider_instance_id=None),
            NodeInstance(
                id="retriever",
                node_type_id="retrieval.dense",
                config={
                    "strategy": "semantic",
                    "top_k": 5,
                    "query_language_handling": "auto",
                    "cross_lingual": True,
                },
                provider_instance_id=None,
            ),
            NodeInstance(
                id="context",
                node_type_id="context.assemble",
                config={
                    "preserve_source_language": True,
                    "response_language": "query",
                    "technique": "standard",
                    "max_chars": 12000,
                },
                provider_instance_id=None,
            ),
            NodeInstance(
                id="answer",
                node_type_id="generation.answer",
                config={
                    "provider": "groq",
                    "model": "openai/gpt-oss-20b",
                    "response_language": "query",
                    "temperature": 0.2,
                    "max_tokens": 512,
                },
                provider_instance_id="generation-provider",
            ),
        ],
        connections=[
            GraphConnection(source_node_id="source", source_port_id="documents", target_node_id="parser", target_port_id="documents"),
            GraphConnection(source_node_id="parser", source_port_id="parsed_documents", target_node_id="chunker", target_port_id="parsed_documents"),
            GraphConnection(source_node_id="chunker", source_port_id="chunks", target_node_id="embedder", target_port_id="chunks"),
            GraphConnection(source_node_id="embedder", source_port_id="embeddings", target_node_id="index", target_port_id="embeddings"),
            GraphConnection(source_node_id="query", source_port_id="query", target_node_id="retriever", target_port_id="query"),
            GraphConnection(source_node_id="index", source_port_id="index", target_node_id="retriever", target_port_id="index"),
            GraphConnection(source_node_id="retriever", source_port_id="retrieved_chunks", target_node_id="context", target_port_id="retrieved_chunks"),
            GraphConnection(source_node_id="query", source_port_id="query", target_node_id="answer", target_port_id="query"),
            GraphConnection(source_node_id="context", source_port_id="context", target_node_id="answer", target_port_id="context"),
        ],
    )
    agentic_template_graph = WorkflowGraph(
        id="agentic-rag-draft",
        name="Agentic RAG",
        architecture_id="agentic-rag",
        template_id="agentic-rag",
        execution=ExecutionContract(
            mode=ExecutionMode.AGENTIC,
            max_steps=10,
            max_loop_iterations=5,
            timeout_seconds=120,
        ),
        nodes=[
            NodeInstance(id="source", node_type_id="document.source", config={}, provider_instance_id=None),
            NodeInstance(id="parser", node_type_id="document.parse", config={}, provider_instance_id=None),
            NodeInstance(
                id="chunker",
                node_type_id="chunk.split",
                config={"strategy": "recursive", "chunk_size": 1000, "overlap": 120},
                provider_instance_id=None,
            ),
            NodeInstance(
                id="embedder",
                node_type_id="embedding.create",
                config={
                    "provider": "sentence_transformers",
                    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "dimension": 384,
                    "supported_languages": "multilingual",
                },
                provider_instance_id="embedding-provider",
            ),
            NodeInstance(
                id="index",
                node_type_id="vector.index",
                config={
                    "backend": "chroma",
                    "collection_name": "visual_rag_studio_documents",
                    "url": "http://localhost:8000",
                },
                provider_instance_id="chroma-local",
            ),
            NodeInstance(id="query", node_type_id="query.input", config={}, provider_instance_id=None),
            NodeInstance(
                id="retriever",
                node_type_id="retrieval.dense",
                config={"strategy": "semantic", "top_k": 10},
                provider_instance_id=None,
            ),
            NodeInstance(
                id="agent",
                node_type_id="agent.orchestrate",
                config={
                    "goal": "Answer the query factually based on retrieved evidence.",
                    "instructions": "Evaluate evidence relevance. If insufficient, iterate or abstain.",
                    "mode": "bounded_loop",
                    "max_steps": 5,
                    "max_retrieval_iterations": 3,
                    "token_budget": 4096,
                    "timeout_seconds": 60.0,
                    "approval_policy": "on_uncertainty",
                },
                provider_instance_id="agent-orchestrator",
            ),
            NodeInstance(id="context", node_type_id="context.assemble", config={}, provider_instance_id=None),
            NodeInstance(
                id="answer",
                node_type_id="generation.answer",
                config={
                    "provider": "groq",
                    "model": "openai/gpt-oss-20b",
                    "response_language": "query",
                    "temperature": 0.2,
                    "max_tokens": 512,
                },
                provider_instance_id="generation-provider",
            ),
        ],
        connections=[
            GraphConnection(source_node_id="source", source_port_id="documents", target_node_id="parser", target_port_id="documents"),
            GraphConnection(source_node_id="parser", source_port_id="parsed_documents", target_node_id="chunker", target_port_id="parsed_documents"),
            GraphConnection(source_node_id="chunker", source_port_id="chunks", target_node_id="embedder", target_port_id="chunks"),
            GraphConnection(source_node_id="embedder", source_port_id="embeddings", target_node_id="index", target_port_id="embeddings"),
            GraphConnection(source_node_id="index", source_port_id="index", target_node_id="retriever", target_port_id="index"),
            GraphConnection(source_node_id="query", source_port_id="query", target_node_id="retriever", target_port_id="query"),
            GraphConnection(source_node_id="query", source_port_id="query", target_node_id="agent", target_port_id="query"),
            GraphConnection(source_node_id="retriever", source_port_id="retrieved_chunks", target_node_id="agent", target_port_id="retrieved_chunks"),
            GraphConnection(source_node_id="agent", source_port_id="retrieved_chunks", target_node_id="context", target_port_id="retrieved_chunks"),
            GraphConnection(source_node_id="query", source_port_id="query", target_node_id="answer", target_port_id="query"),
            GraphConnection(source_node_id="context", source_port_id="context", target_node_id="answer", target_port_id="context"),
        ],
    )
    templates = [
        template.model_copy(
            update={"graph": hybrid_template_graph, "implementation_status": ImplementationStatus.IMPLEMENTED}
        )
        if template.id == "hybrid-rag"
        else (
            template.model_copy(
                update={"graph": rerank_template_graph, "implementation_status": ImplementationStatus.IMPLEMENTED}
            )
            if template.id == "reranking-rag"
            else (
                template.model_copy(
                    update={"graph": multilingual_template_graph, "implementation_status": ImplementationStatus.IMPLEMENTED}
                )
                if template.id == "multilingual-rag"
                else (
                    template.model_copy(
                        update={"graph": agentic_template_graph, "implementation_status": ImplementationStatus.IMPLEMENTED}
                    )
                    if template.id == "agentic-rag"
                    else (
                        template.model_copy(
                            update={"graph": template_graph, "implementation_status": ImplementationStatus.IMPLEMENTED}
                        )
                        if template.id == "traditional-rag"
                        else template
                    )
                )
            )
        )
        for template in templates
    ]
    providers = [
        ProviderAdapterDefinition(
            id="sentence-transformers.embedding",
            display_name="Sentence Transformers (Multilingual MiniLM)",
            provider_kind="embedding",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["text-embedding", "multilingual-embedding"],
            dependencies=["sentence-transformers"],
            runtime_implementation_id="rag.providers.SentenceTransformersProvider.embed",
            known_limitations=["Runs on CPU; first use downloads model weights unless cached."],
        ),
        ProviderAdapterDefinition(
            id="groq.chat",
            display_name="Groq Chat Completions",
            provider_kind="generation",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["chat-generation", "grounded-generation"],
            dependencies=["httpx"],
            runtime_implementation_id="rag.providers.GroqProvider.generate",
            verification_status="verified",
            evidence=[
                CapabilityEvidence(
                    verified_at="2026-11-21",
                    test_references=["backend/tests/test_live_providers.py::test_live_provider_minimal_prompt_and_grounded_query[groq]"],
                    notes="Live minimal prompt and one grounded RAG query with openai/gpt-oss-20b.",
                )
            ],
            known_limitations=["Requires a Groq API key and a model enabled for the account."],
        ),
        ProviderAdapterDefinition(
            id="ollama.embedding",
            display_name="Ollama Embeddings",
            provider_kind="embedding",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["text-embedding"],
            dependencies=["httpx"],
            runtime_implementation_id="rag.providers.OllamaProvider.embed",
        ),
        ProviderAdapterDefinition(
            id="ollama.chat",
            display_name="Ollama Chat",
            provider_kind="generation",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["chat-generation"],
            dependencies=["httpx"],
            runtime_implementation_id="rag.providers.OllamaProvider.generate",
            verification_status="verified",
            evidence=[
                CapabilityEvidence(
                    verified_at="2026-11-21",
                    test_references=["backend/tests/test_live_providers.py::test_live_provider_minimal_prompt_and_grounded_query[ollama]"],
                    notes="Live minimal prompt and one grounded RAG query with gemma3:4b. Local Ollama server.",
                )
            ],
        ),
        ProviderAdapterDefinition(
            id="huggingface.chat",
            display_name="Hugging Face Inference Providers",
            provider_kind="generation",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["chat-generation", "grounded-generation"],
            dependencies=["httpx"],
            runtime_implementation_id="rag.providers.HuggingFaceProvider.generate",
            verification_status="unverified",
            evidence=[
                CapabilityEvidence(
                    verified_at="2026-11-21",
                    test_references=["backend/tests/test_live_providers.py::test_live_provider_minimal_prompt_and_grounded_query[huggingface]"],
                    notes="Live check failed: Hugging Face reported no available account credit. Not verified; rerun after billing is resolved.",
                )
            ],
            known_limitations=[
                "Requires a user-supplied HF_TOKEN and a model/provider combination enabled for Inference Providers."
            ],
        ),
        ProviderAdapterDefinition(
            id="chroma.persistent",
            display_name="Persistent Chroma Vector Store",
            provider_kind="vector_store",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["dense-vector-store"],
            dependencies=["chromadb"],
            runtime_implementation_id="rag.vector_store.ChromaVectorStore",
        ),
        ProviderAdapterDefinition(
            id="ragspark.fast_local.embedding",
            display_name="Fast Local Feature Embeddings (Zero-Dependency)",
            provider_kind="embedding",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["text-embedding", "fast-local", "zero-dependency"],
            dependencies=["ragspark"],
            runtime_implementation_id="rag.providers.FastLocalEmbeddingProvider.embed",
        ),
        ProviderAdapterDefinition(
            id="ragspark.sentence_transformers.embedding",
            display_name="Dense Sentence Transformers Embeddings",
            provider_kind="embedding",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["text-embedding", "dense-embedding"],
            dependencies=["sentence-transformers"],
            runtime_implementation_id="rag.providers.SentenceTransformersProvider.embed",
        ),
        ProviderAdapterDefinition(
            id="ragspark.memory.vector_store",
            display_name="In-Memory Vector Store",
            provider_kind="vector_store",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["dense-vector-store", "in-memory", "multi-strategy-retrieval"],
            dependencies=[],
            runtime_implementation_id="rag.vector_store.MemoryVectorStore",
        ),
        ProviderAdapterDefinition(
            id="ragspark.chroma.vector_store",
            display_name="Persistent Chroma Vector Store",
            provider_kind="vector_store",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["dense-vector-store", "chroma-persistent", "multi-strategy-retrieval"],
            dependencies=["chromadb"],
            runtime_implementation_id="rag.vector_store.ChromaVectorStore",
        ),
        ProviderAdapterDefinition(
            id="ragspark.flash.reranker",
            display_name="Fast Cross-Scoring Reranker (<2ms Zero-Dependency)",
            provider_kind="reranker",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["fast-reranking", "cross-scoring", "zero-dependency"],
            dependencies=["ragspark"],
            runtime_implementation_id="ragspark.reranker.Reranker.flash",
        ),
        ProviderAdapterDefinition(
            id="ragspark.cross_encoder.reranker",
            display_name="Neural Cross-Encoder Reranker",
            provider_kind="reranker",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["neural-reranking", "cross-encoder"],
            dependencies=["sentence-transformers"],
            runtime_implementation_id="ragspark.reranker.Reranker.cross_encoder",
        ),
        ProviderAdapterDefinition(
            id="ragspark.extractive.generation",
            display_name="Extractive Grounded Generator (Local Zero-Dependency)",
            provider_kind="generation",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["extractive-generation", "grounded-generation", "zero-dependency"],
            dependencies=["ragspark"],
            runtime_implementation_id="rag.providers.ExtractiveGeneratorProvider.generate",
        ),
        ProviderAdapterDefinition(
            id="openai.chat",
            display_name="OpenAI Chat Completions",
            provider_kind="generation",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["chat-generation", "grounded-generation"],
            dependencies=["httpx"],
            runtime_implementation_id="rag.providers.OpenAIProvider.generate",
        ),
        ProviderAdapterDefinition(
            id="gemini.chat",
            display_name="Google Gemini Content Generation",
            provider_kind="generation",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["chat-generation", "grounded-generation"],
            dependencies=["httpx"],
            runtime_implementation_id="rag.providers.GeminiProvider.generate",
        ),
        ProviderAdapterDefinition(
            id="anthropic.chat",
            display_name="Anthropic Claude Messages",
            provider_kind="generation",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["chat-generation", "grounded-generation"],
            dependencies=["httpx"],
            runtime_implementation_id="rag.providers.AnthropicProvider.generate",
        ),
        ProviderAdapterDefinition(
            id="ragspark.agent.orchestrator",
            display_name="RagSpark Bounded Agent Orchestrator",
            provider_kind="agent",
            implementation_status=ImplementationStatus.IMPLEMENTED,
            capabilities=["agent-orchestration", "evidence-evaluation", "bounded-loop-execution"],
            dependencies=["ragspark"],
            runtime_implementation_id="app.rag.agent.AgentOrchestrator.run",
            known_limitations=["Bounded to allowlisted tools and configured step budgets."],
        ),
    ]
    return RegistrySnapshot(
        catalog=catalog,
        templates=templates,
        node_types=nodes,
        provider_adapters=providers,
        chunking_strategies=strategies,
    )


REGISTRY = build_registry()
CATALOG_BY_ID = {entry.id: entry for entry in REGISTRY.catalog}
NODE_TYPES_BY_ID = {definition.node_type_id: definition for definition in REGISTRY.node_types}
TEMPLATES_BY_ID = {template.id: template for template in REGISTRY.templates}
CHUNKING_STRATEGIES = {strategy.id: strategy for strategy in REGISTRY.chunking_strategies}
