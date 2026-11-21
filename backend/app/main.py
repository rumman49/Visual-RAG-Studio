"""FastAPI app with distinct liveness and readiness checks."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.auth import install_api_key_auth
from app.config import get_settings
from app.contracts import RegistrySnapshot, WorkflowValidationRequest
from app.rag.graph import configured_provider_instances, validate_traditional_graph
from app.rag.pipeline import PipelineError
from app.rag.routes import router as rag_router
from app.registry import CATALOG_BY_ID, NODE_TYPES_BY_ID, REGISTRY
from app.validation import validate_graph

settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")

# Registered before CORS so that 401 responses still receive CORS headers.
install_api_key_auth(app, settings)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router)


@app.exception_handler(PipelineError)
def pipeline_error_handler(_, exc: PipelineError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code, content={"detail": {"code": exc.code, "message": str(exc)}}
    )


@app.get("/api/health/live", tags=["health"])
def liveness() -> dict[str, str]:
    """Process is responsive; this does not imply dependencies are ready."""
    return {"status": "alive"}


@app.get("/api/health/ready", tags=["health"])
def readiness() -> dict[str, object]:
    """Report configured prerequisites without pretending providers are connected."""
    return {
        "status": "ready",
        "mode": settings.environment,
        "auth": "required" if settings.api_key is not None else "none",
        "checks": {
            "configuration": "valid",
            "database": "not_checked",
            "vector_store": "not_checked",
            "ollama": "not_checked",
        },
        "message": "Foundation readiness only; data and model providers are not initialized.",
    }


@app.get("/api/v1/catalog", response_model=list)
def list_catalog(
    category: str | None = None,
    status: str | None = None,
    q: str | None = Query(default=None, max_length=200),
) -> list[dict]:
    """Return product-owned catalog metadata; entries do not imply runnable templates."""
    rows = REGISTRY.catalog
    if category:
        rows = [entry for entry in rows if entry.category.casefold() == category.casefold()]
    if status:
        rows = [entry for entry in rows if entry.implementation_status.value == status.casefold()]
    if q:
        needle = q.casefold()
        rows = [
            entry
            for entry in rows
            if needle in entry.name.casefold()
            or needle in entry.description.casefold()
            or any(needle in facet.casefold() for facet in entry.composition_facets)
        ]
    return [entry.model_dump(mode="json") for entry in rows]


@app.get("/api/v1/catalog/{entry_id}")
def get_catalog_entry(entry_id: str) -> dict:
    entry = CATALOG_BY_ID.get(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Catalog entry not found")
    return entry.model_dump(mode="json")


@app.get("/api/v1/node-types")
def list_node_types() -> list[dict]:
    return [node.model_dump(mode="json") for node in REGISTRY.node_types]


@app.get("/api/v1/registry", response_model=RegistrySnapshot)
def get_registry() -> RegistrySnapshot:
    return REGISTRY


@app.post("/api/v1/workflows/validate")
def validate_workflow(request: WorkflowValidationRequest) -> dict:
    supported_archs = {
        "traditional-rag",
        "hybrid-rag",
        "reranking-rag",
        "multilingual-rag",
        "agentic-rag",
        "custom-rag",
        "start-from-blank-canvas",
        "draft-workflow",
        "",
        None,
    }
    arch_id = request.graph.architecture_id
    temp_id = request.graph.template_id

    provided_node_types = {n.node_type_id for n in request.graph.nodes}
    has_agent = "agent.orchestrate" in provided_node_types
    baseline_rag_nodes = {
        "document.source",
        "document.parse",
        "chunk.split",
        "embedding.create",
        "vector.index",
        "query.input",
    }
    if not has_agent:
        baseline_rag_nodes.update({"context.assemble", "generation.answer"})

    has_rag_baseline = baseline_rag_nodes.issubset(provided_node_types)
    is_supported_arch = (arch_id in supported_archs) or (temp_id in supported_archs) or (arch_id is None)

    if is_supported_arch and has_rag_baseline:
        report = validate_traditional_graph(
            request.graph,
            get_settings(),
            request.generation_provider,
            request.generation_model,
            require_executable=request.require_executable,
        )
        return report.model_dump(mode="json")

    embed_node = next((n for n in request.graph.nodes if n.node_type_id == "embedding.create"), None)
    embed_prov = str(embed_node.config.get("provider", "")) if embed_node and embed_node.config else None
    embed_mod = str(embed_node.config.get("model", "")) if embed_node and embed_node.config else None

    v_node = next((n for n in request.graph.nodes if n.node_type_id == "vector.index"), None)
    v_backend = str(v_node.config.get("backend", "chroma")) if v_node and v_node.config else "chroma"

    gen_node = next((n for n in request.graph.nodes if n.node_type_id == "generation.answer"), None)
    gen_prov = request.generation_provider or (str(gen_node.config.get("provider", "")) if gen_node and gen_node.config else None)
    gen_mod = request.generation_model or (str(gen_node.config.get("model", "")) if gen_node and gen_node.config else None)

    providers = request.configured_providers or configured_provider_instances(
        get_settings(),
        generation_provider=gen_prov,
        generation_model=gen_mod,
        embedding_provider=embed_prov,
        embedding_model=embed_mod,
        vector_backend=v_backend,
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
        for node in request.graph.nodes
    ]
    graph_to_validate = request.graph.model_copy(update={"nodes": normalized_nodes})

    report = validate_graph(
        graph_to_validate,
        NODE_TYPES_BY_ID,
        configured_providers=providers,
        available_dependencies=set(request.available_dependencies)
        if request.available_dependencies is not None
        else None,
        require_executable=request.require_executable,
    )
    return report.model_dump(mode="json")
