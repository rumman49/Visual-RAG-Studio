"""Guardrail for the one deterministic execution graph supported in Story 4."""

import importlib.util

from app.config import Settings
from app.contracts import (
    ConfiguredProviderInstance,
    ConfiguredProviderStatus,
    GraphValidationIssue,
    GraphValidationReport,
    ImplementationStatus,
    IssueSeverity,
    WorkflowGraph,
)
from app.rag.chunking import ChunkingError, validate_settings
from app.rag.credentials import runtime_credentials
from app.registry import NODE_TYPES_BY_ID
from app.validation import validate_graph


def configured_provider_instances(
    settings: Settings,
    generation_provider: str | None = None,
    generation_model: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    vector_backend: str | None = None,
) -> list[ConfiguredProviderInstance]:
    selected_gen = generation_provider or settings.generation_provider
    gen_model = generation_model
    if not gen_model:
        if selected_gen == "groq":
            gen_model = settings.groq_generation_model
        elif selected_gen == "ollama":
            gen_model = settings.ollama_generation_model
        elif selected_gen == "huggingface":
            gen_model = settings.hf_generation_model
        elif selected_gen == "extractive":
            gen_model = "extractive"
        else:
            gen_model = "default"

    groq_key = runtime_credentials.get_key("groq") or (
        settings.groq_api_key.get_secret_value().strip() if settings.groq_api_key else None
    )
    openai_key = runtime_credentials.get_key("openai") or (
        settings.openai_api_key.get_secret_value().strip() if getattr(settings, "openai_api_key", None) else None
    )
    gemini_key = runtime_credentials.get_key("gemini") or (
        settings.gemini_api_key.get_secret_value().strip() if getattr(settings, "gemini_api_key", None) else None
    )
    anthropic_key = runtime_credentials.get_key("anthropic") or (
        settings.anthropic_api_key.get_secret_value().strip() if getattr(settings, "anthropic_api_key", None) else None
    )
    token_present = bool(settings.hf_token and settings.hf_token.get_secret_value().strip()) or bool(runtime_credentials.get_key("huggingface"))

    if selected_gen == "extractive":
        gen_configured = True
        gen_adapter = "ragspark.extractive.generation"
    elif selected_gen == "groq":
        gen_configured = bool(gen_model) and bool(groq_key)
        gen_adapter = "groq.chat"
    elif selected_gen == "openai":
        gen_configured = bool(gen_model) and bool(openai_key)
        gen_adapter = "openai.chat"
    elif selected_gen == "gemini":
        gen_configured = bool(gen_model) and bool(gemini_key)
        gen_adapter = "gemini.chat"
    elif selected_gen == "anthropic":
        gen_configured = bool(gen_model) and bool(anthropic_key)
        gen_adapter = "anthropic.chat"
    elif selected_gen == "huggingface":
        gen_configured = bool(gen_model) and token_present
        gen_adapter = "huggingface.chat"
    elif selected_gen == "ollama":
        gen_configured = bool(gen_model)
        gen_adapter = "ollama.chat"
    else:
        gen_configured = False
        gen_adapter = f"{selected_gen}.chat" if selected_gen else "unknown.chat"

    selected_emb = embedding_provider or settings.embedding_provider
    emb_model = embedding_model
    if not emb_model:
        if selected_emb == "sentence_transformers":
            emb_model = settings.sentence_transformers_model
        elif selected_emb == "fast_local":
            emb_model = "fast_local"
        else:
            emb_model = settings.ollama_embedding_model or "default"

    if selected_emb == "fast_local":
        emb_available = True
        emb_adapter = "ragspark.fast_local.embedding"
    elif selected_emb == "sentence_transformers":
        emb_available = bool(emb_model) and (
            importlib.util.find_spec("sentence_transformers") is not None
        )
        emb_adapter = "sentence-transformers.embedding"
    elif selected_emb == "ollama":
        emb_available = bool(emb_model)
        emb_adapter = "ollama.embedding"
    else:
        emb_available = False
        emb_adapter = f"{selected_emb}.embedding" if selected_emb else "unknown.embedding"

    v_backend = (vector_backend or "chroma").lower()
    if v_backend == "memory":
        store_adapter = "ragspark.memory.vector_store"
        store_available = True
    elif v_backend in {"chroma", "chromadb"}:
        store_adapter = "chroma.persistent"
        chroma_path = getattr(settings, "chroma_path", None)
        store_available = bool(chroma_path)
    else:
        store_adapter = f"{v_backend}.vector_store"
        store_available = False

    return [
        ConfiguredProviderInstance(
            id="embedding-provider",
            adapter_id=emb_adapter,
            status=ConfiguredProviderStatus.CONFIGURED
            if emb_available
            else ConfiguredProviderStatus.UNAVAILABLE,
            config={"model": emb_model} if emb_model else {},
        ),
        ConfiguredProviderInstance(
            id="generation-provider",
            adapter_id=gen_adapter,
            status=ConfiguredProviderStatus.CONFIGURED
            if gen_configured
            else ConfiguredProviderStatus.UNAVAILABLE,
            config={"model": gen_model} if gen_model else {},
        ),
        ConfiguredProviderInstance(
            id="chroma-local",
            adapter_id=store_adapter,
            status=ConfiguredProviderStatus.CONFIGURED
            if store_available
            else ConfiguredProviderStatus.UNAVAILABLE,
        ),
        ConfiguredProviderInstance(id="flash-reranker", adapter_id="ragspark.flash.reranker"),
        ConfiguredProviderInstance(id="agent-orchestrator", adapter_id="ragspark.agent.orchestrator"),
    ]


def validate_traditional_graph(
    graph: WorkflowGraph,
    settings: Settings,
    generation_provider: str | None = None,
    generation_model: str | None = None,
    require_executable: bool = True,
) -> GraphValidationReport:
    supported_archs = {
        "traditional-rag",
        "reranking-rag",
        "hybrid-rag",
        "multilingual-rag",
        "agentic-rag",
        "custom-rag",
        "start-from-blank-canvas",
        "draft-workflow",
        "",
        None,
    }
    if graph.architecture_id not in supported_archs or (
        graph.template_id and graph.template_id not in supported_archs
    ):
        issue = GraphValidationIssue(
            code="unsupported_execution_template",
            severity=IssueSeverity.ERROR,
            message=f"Architecture '{graph.architecture_id}' is not executable. Supported architectures: {sorted(s for s in supported_archs if s)}.",
        )
        return GraphValidationReport(valid=False, executable=False, workflow_state="draft", issues=[issue])

    has_agent = any(n.node_type_id == "agent.orchestrate" for n in graph.nodes)
    has_reranker = any(n.node_type_id == "retrieval.rerank" for n in graph.nodes)
    has_fuse = any(n.node_type_id == "merge.fuse" for n in graph.nodes)

    baseline_nodes = {
        "document.source",
        "document.parse",
        "chunk.split",
        "embedding.create",
        "vector.index",
        "query.input",
    }
    if not has_agent:
        baseline_nodes.update({"context.assemble", "generation.answer"})

    provided_types = {node.node_type_id for node in graph.nodes}
    allowed_types = baseline_nodes | {
        "retrieval.dense",
        "retrieval.sparse",
        "merge.fuse",
        "retrieval.rerank",
        "context.assemble",
        "generation.answer",
        "agent.orchestrate",
    }

    has_retriever = bool(provided_types & {"retrieval.dense", "retrieval.sparse", "agent.orchestrate"})

    if not baseline_nodes.issubset(provided_types) or not has_retriever or not provided_types.issubset(allowed_types):
        issue = GraphValidationIssue(
            code="unsupported_graph_components",
            severity=IssueSeverity.ERROR,
            message="Workflow requires baseline RAG nodes (source, parser, chunker, embedder, index, query, retriever/agent); additional components beyond reranker, sparse retrieval, fusion, and agent orchestrator are not supported.",
        )
        return GraphValidationReport(valid=False, executable=False, workflow_state="draft", issues=[issue])

    has_reranker = any(n.node_type_id == "retrieval.rerank" for n in graph.nodes)
    has_fuse = any(n.node_type_id == "merge.fuse" for n in graph.nodes)

    # Validate topology by typed dataflow rather than literal string IDs
    node_id_to_type = {n.id: n.node_type_id for n in graph.nodes}
    actual_type_connections = {
        (node_id_to_type.get(c.source_node_id), node_id_to_type.get(c.target_node_id))
        for c in graph.connections
        if node_id_to_type.get(c.source_node_id) and node_id_to_type.get(c.target_node_id)
    }

    missing_flows: list[str] = []
    # Ingestion flows
    for src, tgt, label in [
        ("document.source", "document.parse", "Document Source must connect to Document Parser"),
        ("document.parse", "chunk.split", "Document Parser must connect to Chunk Splitter"),
        ("chunk.split", "embedding.create", "Chunk Splitter must connect to Embedding Creator"),
        ("embedding.create", "vector.index", "Embedding Creator must connect to Vector Store Index"),
    ]:
        if (src, tgt) not in actual_type_connections:
            missing_flows.append(label)

    # Query flow to retrieval or agent
    if has_agent:
        has_query_to_target = any(
            ("query.input", tgt) in actual_type_connections
            for tgt in ("agent.orchestrate", "retrieval.dense", "retrieval.sparse")
            if tgt in provided_types
        )
        if not has_query_to_target:
            missing_flows.append("Query Input must connect to Retrieval or Agent Orchestrator")
        if "generation.answer" in provided_types and ("context.assemble", "generation.answer") not in actual_type_connections and ("agent.orchestrate", "generation.answer") not in actual_type_connections:
            missing_flows.append("Agent Orchestrator or Context Assembler must connect to Answer Generation")
    else:
        has_query_to_retriever = any(
            ("query.input", ret) in actual_type_connections
            for ret in ("retrieval.dense", "retrieval.sparse")
            if ret in provided_types
        )
        if not has_query_to_retriever:
            missing_flows.append("Query Input must connect to Retrieval")

        # Retrieval to merge.fuse / reranker / context.assemble
        if has_fuse:
            has_retriever_to_fuse = any(
                (ret, "merge.fuse") in actual_type_connections
                for ret in ("retrieval.dense", "retrieval.sparse")
                if ret in provided_types
            )
            if not has_retriever_to_fuse:
                missing_flows.append("Retrieval must connect to Fusion & Ranking (RRF)")
            if has_reranker:
                if ("merge.fuse", "retrieval.rerank") not in actual_type_connections:
                    missing_flows.append("Fusion & Ranking (RRF) must connect to Reranker")
            else:
                if ("merge.fuse", "context.assemble") not in actual_type_connections:
                    missing_flows.append("Fusion & Ranking (RRF) must connect to Context Assembler")
        else:
            target_stage = "retrieval.rerank" if has_reranker else "context.assemble"
            has_retriever_to_next = any(
                (ret, target_stage) in actual_type_connections
                for ret in ("retrieval.dense", "retrieval.sparse")
                if ret in provided_types
            )
            if not has_retriever_to_next:
                missing_flows.append(f"Retrieval must connect to {'Reranker' if has_reranker else 'Context Assembler'}")

        if has_reranker and ("retrieval.rerank", "context.assemble") not in actual_type_connections:
            missing_flows.append("Reranker must connect to Context Assembler")

        if ("context.assemble", "generation.answer") not in actual_type_connections:
            missing_flows.append("Context Assembler must connect to Answer Generation")

    if missing_flows:
        issue = GraphValidationIssue(
            code="unsupported_graph_topology",
            severity=IssueSeverity.ERROR,
            message=f"Graph connections do not match the required Traditional RAG execution topology: {'; '.join(missing_flows)}",
        )
        return GraphValidationReport(valid=False, executable=False, workflow_state="draft", issues=[issue])

    # Extract configs for provider validation
    embed_node = next((n for n in graph.nodes if n.node_type_id == "embedding.create"), None)
    embed_prov = str(embed_node.config.get("provider", "")) if embed_node else None
    embed_mod = str(embed_node.config.get("model", "")) if embed_node else None

    v_node = next((n for n in graph.nodes if n.node_type_id == "vector.index"), None)
    v_backend = str(v_node.config.get("backend", "chroma")) if v_node else "chroma"

    gen_node = next((n for n in graph.nodes if n.node_type_id == "generation.answer"), None)
    gen_prov = generation_provider or (str(gen_node.config.get("provider", "")) if gen_node else None)
    gen_mod = generation_model or (str(gen_node.config.get("model", "")) if gen_node else None)

    providers = configured_provider_instances(
        settings,
        generation_provider=gen_prov,
        generation_model=gen_mod,
        embedding_provider=embed_prov,
        embedding_model=embed_mod,
        vector_backend=v_backend,
    )
    node_types = dict(NODE_TYPES_BY_ID)
    if has_reranker and "retrieval.rerank" in node_types:
        node_types["retrieval.rerank"] = node_types["retrieval.rerank"].model_copy(
            update={
                "implementation_status": ImplementationStatus.IMPLEMENTED,
                "runtime_implementation_id": "ragspark.reranker.Reranker.flash",
            }
        )
    if has_agent and "agent.orchestrate" in node_types:
        node_types["agent.orchestrate"] = node_types["agent.orchestrate"].model_copy(
            update={
                "implementation_status": ImplementationStatus.IMPLEMENTED,
                "runtime_implementation_id": "rag.agent.orchestrator.v1",
            }
        )
    DEFAULT_PROVIDER_BY_TYPE = {
        "embedding.create": "embedding-provider",
        "vector.index": "chroma-local",
        "generation.answer": "generation-provider",
        "retrieval.rerank": "flash-reranker",
        "agent.orchestrate": "agent-orchestrator",
    }
    normalized_nodes = [
        node.model_copy(update={"provider_instance_id": DEFAULT_PROVIDER_BY_TYPE[node.node_type_id]})
        if (not node.provider_instance_id and node.node_type_id in DEFAULT_PROVIDER_BY_TYPE)
        else node
        for node in graph.nodes
    ]
    graph = graph.model_copy(update={"nodes": normalized_nodes})

    report = validate_graph(
        graph, node_types, configured_providers=providers, require_executable=require_executable
    )
    if has_agent:
        allowed_modes = {"deterministic_dag", "agentic", "stateful_conditional"}
        if graph.execution.mode.value not in allowed_modes:
            issue = GraphValidationIssue(
                code="unsupported_execution_mode",
                severity=IssueSeverity.ERROR,
                message=f"Agent workflow mode must be one of {sorted(allowed_modes)}.",
            )
            return report.model_copy(
                update={"valid": False, "executable": False, "workflow_state": "draft", "issues": [*report.issues, issue]}
            )
    else:
        if graph.execution.mode.value != "deterministic_dag":
            issue = GraphValidationIssue(
                code="unsupported_execution_mode",
                severity=IssueSeverity.ERROR,
                message="RAG runs only as a deterministic DAG.",
            )
            return report.model_copy(
                update={"valid": False, "executable": False, "workflow_state": "draft", "issues": [*report.issues, issue]}
            )
        if graph.execution.max_steps is not None or graph.execution.max_loop_iterations is not None:
            issue = GraphValidationIssue(
                code="unsupported_execution_bounds",
                severity=IssueSeverity.ERROR,
                message="Control-flow bounds are not used by the deterministic RAG runner.",
            )
            return report.model_copy(
                update={"valid": False, "executable": False, "workflow_state": "draft", "issues": [*report.issues, issue]}
            )
    chunk_node = next(node for node in graph.nodes if node.node_type_id == "chunk.split")
    try:
        validate_settings(
            str(chunk_node.config.get("strategy", "")),
            {
                "chunk_size": chunk_node.config.get("chunk_size"),
                "overlap": chunk_node.config.get("overlap"),
            },
        )
    except ChunkingError as exc:
        issue = GraphValidationIssue(
            code="invalid_chunking_config",
            severity=IssueSeverity.ERROR,
            message=str(exc),
            node_id=chunk_node.id,
        )
        return report.model_copy(
            update={"valid": False, "executable": False, "issues": [*report.issues, issue]}
        )
    return report
