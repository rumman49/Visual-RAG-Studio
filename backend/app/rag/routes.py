from __future__ import annotations

from datetime import UTC, datetime
from pathlib import PurePath
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from starlette.requests import Request

from app.config import Settings, get_settings
from app.contracts import ChunkingChoiceMode, DocumentProfile
from app.rag.agent import AgentApprovalPolicy, AgentExecutionMode, AgentOrchestrator
from app.rag.chunking import (
    SUPPORTED_STRATEGIES,
    ChunkingError,
    MultiStrategyChunkPlan,
    chunk_document_multi,
    validate_settings,
)
from app.rag.contracts import (
    AgentRunRequest,
    AgentRunResponse,
    ChunkingSelection,
    ChunkingSelectionRequest,
    ChunkPage,
    ChunkRecord,
    DeleteBulkResponse,
    DeleteChunkResponse,
    DeleteChunksResponse,
    DeleteDocumentResponse,
    DocumentChunkList,
    DocumentStatusResponse,
    ExecutionProgress,
    GenerationProviderConfig,
    GroqCredentialStatus,
    IngestionRequest,
    IngestionResponse,
    ProviderStatus,
    ProviderTestRequest,
    ProviderTestResponse,
    QueryRequest,
    QueryResponse,
    RAGModel,
    StorageCleanupResponse,
    StorageCollection,
    StorageSummary,
    WorkflowResponse,
    WorkflowSaveRequest,
)
from app.rag.credentials import runtime_credentials
from app.rag.dependencies import (
    get_document_store,
    get_groq,
    get_huggingface,
    get_ollama,
    get_sentence_transformers,
    get_vector_store,
)
from app.rag.execution import execution_tracker
from app.rag.graph import validate_traditional_graph
from app.rag.parser import DocumentParseError, parse_document
from app.rag.pipeline import PipelineError, _parsed, answer_query, index_document
from app.rag.planner import IngestionPlanner, PlanningMode
from app.rag.providers import (
    AnthropicProvider,
    ExtractiveGeneratorProvider,
    FastLocalEmbeddingProvider,
    GeminiProvider,
    GroqProvider,
    HuggingFaceProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    SentenceTransformersProvider,
)
from app.rag.ragspark_adapters import RagSparkReranker
from app.rag.recommendations import recommend_chunking
from app.rag.storage import DocumentStore
from app.rag.vector_store import (
    ChromaVectorStore,
    VectorStoreError,
    get_vector_store_adapter,
)
from app.validation import secret_config_issues

router = APIRouter(prefix="/api/v1")
SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}
INDEX_NODE_IDS = ["source", "parser", "chunker", "embedder", "index"]
QUERY_NODE_IDS = ["query", "retriever", "context", "answer"]
_last_storage_success: datetime | None = None


def _build_step_tracker(execution_id: str, graph_nodes: list[Any]):
    type_to_id = {getattr(n, "node_type_id", None): getattr(n, "id", None) for n in graph_nodes}
    stage_to_type = {
        "source": "document.source",
        "parser": "document.parse",
        "chunker": "chunk.split",
        "embedder": "embedding.create",
        "index": "vector.index",
        "query": "query.input",
        "retriever": "retrieval.dense",
        "sparse": "retrieval.sparse",
        "fuse": "merge.fuse",
        "rerank": "retrieval.rerank",
        "context": "context.assemble",
        "answer": "generation.answer",
        "agent": "agent.orchestrate",
    }

    def on_step(stage: str, status: str):
        execution_tracker.node(execution_id, stage, status)
        node_type = stage_to_type.get(stage)
        if node_type and node_type in type_to_id:
            actual_id = type_to_id[node_type]
            if actual_id and actual_id != stage:
                execution_tracker.node(execution_id, actual_id, status)

    return on_step


def _reject_graph_secrets(graph) -> None:
    """Refuse graphs that embed credentials so they are never processed or stored (AD-9)."""
    issues = secret_config_issues(graph)
    if issues:
        raise HTTPException(
            status_code=422,
            detail={"code": "secret_in_graph", "message": issues[0].message},
        )


def _is_local_origin(origin: str) -> bool:
    parsed_origin = urlsplit(origin)
    return parsed_origin.scheme == "http" and parsed_origin.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def _document_response(row: dict) -> DocumentStatusResponse:
    profile = DocumentProfile.model_validate(row["profile"]) if row.get("profile") else None
    return DocumentStatusResponse(
        document_id=row["document_id"],
        filename=row["filename"],
        file_type=row["file_type"],
        status=row["status"],
        created_at=row["created_at"],
        profile=profile,
        recommendation=row.get("recommendation"),
        plan=row.get("plan"),
        selection=row.get("selection"),
        chunk_count=len(row.get("chunks", [])) if row["status"] == "ready" else 0,
        error=row.get("error"),
    )


def _find_document(store: DocumentStore, document_id: str) -> dict:
    row = store.get(document_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "document_not_found", "message": "Document was not found."},
        )
    return row


def _resolve_document_vector_store(
    row: dict, default_store: ChromaVectorStore, settings: Settings
):
    backend = row.get("vector_backend") or "chroma"
    collection = row.get("collection_name") or "visual_rag_studio_documents"
    if backend == "memory":
        from app.rag.vector_store import get_shared_memory_vector_store

        return get_shared_memory_vector_store(collection)
    if backend in {"pinecone", "mongodb", "qdrant"}:
        try:
            return get_vector_store_adapter(backend=backend, collection_name=collection)
        except Exception:
            return default_store
    if collection == getattr(default_store, "collection_name", "visual_rag_studio_documents"):
        return default_store
    try:
        return ChromaVectorStore(settings.chroma_path, collection_name=collection)
    except Exception:
        return default_store


def _key_for_provider(provider_id: str, settings: Settings) -> str | None:
    session_key = runtime_credentials.get_key(provider_id)
    if session_key:
        return session_key
    if provider_id == "groq" and settings.groq_api_key:
        return settings.groq_api_key.get_secret_value().strip()
    if provider_id == "openai" and getattr(settings, "openai_api_key", None):
        return settings.openai_api_key.get_secret_value().strip()
    if provider_id == "gemini" and getattr(settings, "gemini_api_key", None):
        return settings.gemini_api_key.get_secret_value().strip()
    if provider_id == "anthropic" and getattr(settings, "anthropic_api_key", None):
        return settings.anthropic_api_key.get_secret_value().strip()
    if provider_id == "huggingface" and settings.hf_token:
        return settings.hf_token.get_secret_value().strip()
    return None


def _groq_key(settings: Settings) -> str | None:
    return _key_for_provider("groq", settings)


def _require_generation_config(provider_id: str, model: str | None, settings: Settings) -> None:
    if provider_id == "extractive":
        return
    if not model:
        setting = {
            "huggingface": "VRS_HF_GENERATION_MODEL",
            "groq": "VRS_GROQ_GENERATION_MODEL",
            "ollama": "VRS_OLLAMA_GENERATION_MODEL",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }.get(provider_id, "generation model")
        raise PipelineError("generation_model_missing", f"Enter a model ID or set {setting}.", 503)
    if provider_id in {"groq", "openai", "gemini", "anthropic"} and not _key_for_provider(provider_id, settings):
        raise PipelineError(
            f"{provider_id}_api_key_missing", f"Add a {provider_id.title()} API key in the generation settings.", 503
        )
    if provider_id == "huggingface" and not _key_for_provider("huggingface", settings):
        raise PipelineError(
            "huggingface_token_missing", "HF_TOKEN is missing from the backend environment.", 503
        )


def _model_for(provider_id: str, request_model: str | None, settings: Settings) -> str | None:
    if provider_id == "extractive":
        return request_model or "extractive"
    configured = {
        "groq": settings.groq_generation_model,
        "ollama": settings.ollama_generation_model or "qwen3:1.7b",
        "huggingface": settings.hf_generation_model or "meta-llama/Llama-3.1-8B-Instruct",
        "openai": getattr(settings, "openai_generation_model", "gpt-4o-mini"),
        "gemini": getattr(settings, "gemini_generation_model", "gemini-2.0-flash"),
        "anthropic": getattr(settings, "anthropic_generation_model", "claude-3-5-sonnet-20241022"),
    }
    return request_model or configured.get(provider_id) or "default"


@router.get("/providers/health", response_model=ProviderStatus)
def provider_health(
    settings: Settings = Depends(get_settings), provider: OllamaProvider = Depends(get_ollama)
):
    generation_model = _model_for(settings.generation_provider, None, settings)
    if settings.embedding_provider == "sentence_transformers":
        status = SentenceTransformersProvider.health(
            settings.sentence_transformers_model, settings.sentence_transformers_local_files_only
        )
        status["generation_model"] = generation_model
        return status
    status = provider.health(settings.ollama_embedding_model, generation_model)
    status["generation_model"] = generation_model
    return status


@router.get("/executions/{execution_id}", response_model=ExecutionProgress)
def execution_progress(execution_id: str):
    progress = execution_tracker.get(execution_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Execution was not found or has expired.")
    return progress


@router.get("/providers/config", response_model=GenerationProviderConfig)
def generation_provider_config(settings: Settings = Depends(get_settings)):
    selected = settings.generation_provider
    model = _model_for(selected, None, settings)
    groq_key_present = bool(_key_for_provider("groq", settings))
    hf_token_present = bool(_key_for_provider("huggingface", settings))
    openai_key_present = bool(_key_for_provider("openai", settings))
    gemini_key_present = bool(_key_for_provider("gemini", settings))
    anthropic_key_present = bool(_key_for_provider("anthropic", settings))
    credential_present = (
        selected in {"ollama", "extractive"}
        or (selected == "huggingface" and hf_token_present)
        or (selected == "groq" and groq_key_present)
        or (selected == "openai" and openai_key_present)
        or (selected == "gemini" and gemini_key_present)
        or (selected == "anthropic" and anthropic_key_present)
    )
    model = model or None
    return GenerationProviderConfig(
        provider=selected,
        model=model,
        configured=bool(model) and credential_present,
        credential_present=credential_present,
        huggingface_token_present=hf_token_present,
        groq_api_key_present=groq_key_present,
        openai_api_key_present=openai_key_present,
        gemini_api_key_present=gemini_key_present,
        anthropic_api_key_present=anthropic_key_present,
        execution_progress_supported=True,
        provider_policy=settings.hf_provider_policy if selected == "huggingface" else None,
    )


@router.get("/providers/ollama/models")
def list_ollama_models(settings: Settings = Depends(get_settings)):
    try:
        with httpx.Client(timeout=3.0) as client:
            res = client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            if res.is_error:
                return []
            data = res.json()
            return [m.get("name") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


@router.get("/providers/groq/models")
def list_groq_models(settings: Settings = Depends(get_settings)):
    fallback = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "openai/gpt-oss-20b",
    ]
    key = _key_for_provider("groq", settings)
    if not key:
        return fallback
    try:
        with httpx.Client(timeout=5.0) as client:
            res = client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            if res.is_success:
                data = res.json()
                models = [
                    m["id"]
                    for m in data.get("data", [])
                    if isinstance(m, dict) and "id" in m and not m["id"].startswith("whisper")
                ]
                if models:
                    return sorted(models)
    except Exception:
        pass
    return fallback


@router.get("/providers/openai/models")
def list_openai_models(settings: Settings = Depends(get_settings)):
    fallback = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "o1-mini", "o3-mini"]
    key = _key_for_provider("openai", settings)
    if not key:
        return fallback
    try:
        with httpx.Client(timeout=5.0) as client:
            res = client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            if res.is_success:
                data = res.json()
                models = [
                    m["id"]
                    for m in data.get("data", [])
                    if isinstance(m, dict) and "id" in m and ("gpt" in m["id"] or "o1" in m["id"] or "o3" in m["id"])
                ]
                if models:
                    return sorted(models)
    except Exception:
        pass
    return fallback


@router.post("/providers/credentials/groq", response_model=GroqCredentialStatus)
async def set_groq_credential(request: Request):
    """Accept a key without allowing request validation errors to echo its value."""
    origin = request.headers.get("origin")
    if origin and not _is_local_origin(origin):
        raise HTTPException(status_code=403, detail="Groq keys can only be set from a local origin.")
    body = await request.body()
    if len(body) > 1024:
        raise HTTPException(status_code=413, detail="Credential request is too large.")
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Credential request must be valid JSON."
        ) from exc
    api_key = payload.get("api_key") if isinstance(payload, dict) else None
    if not isinstance(api_key, str) or not api_key.strip() or len(api_key) > 512:
        raise HTTPException(status_code=422, detail="Enter a valid Groq API key.")
    runtime_credentials.set_groq_api_key(api_key.strip())
    return GroqCredentialStatus(configured=True, source="session")


@router.delete("/providers/credentials/groq", response_model=GroqCredentialStatus)
def clear_groq_credential(settings: Settings = Depends(get_settings)):
    runtime_credentials.clear_groq_api_key()
    environment_key = (
        settings.groq_api_key.get_secret_value().strip() if settings.groq_api_key else None
    )
    return GroqCredentialStatus(
        configured=bool(environment_key), source="environment" if environment_key else "none"
    )


@router.post("/providers/credentials/{provider}")
async def set_provider_credential(provider: str, request: Request):
    origin = request.headers.get("origin")
    if origin and not _is_local_origin(origin):
        raise HTTPException(status_code=403, detail="Keys can only be set from a local origin.")
    body = await request.body()
    if len(body) > 1024:
        raise HTTPException(status_code=413, detail="Credential request is too large.")
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Credential request must be valid JSON.") from exc
    api_key = payload.get("api_key") if isinstance(payload, dict) else None
    if not isinstance(api_key, str) or not api_key.strip() or len(api_key) > 512:
        raise HTTPException(status_code=422, detail=f"Enter a valid {provider.title()} API key.")
    runtime_credentials.set_key(provider, api_key.strip())
    return {"provider": provider, "configured": True, "source": "session"}


@router.delete("/providers/credentials/{provider}")
def clear_provider_credential(provider: str, settings: Settings = Depends(get_settings)):
    runtime_credentials.clear_key(provider)
    has_env = bool(_key_for_provider(provider, settings))
    return {"provider": provider, "configured": has_env, "source": "environment" if has_env else "none"}


@router.get("/providers/credentials/{provider}")
def get_provider_credential_status(provider: str, settings: Settings = Depends(get_settings)):
    has_key = bool(_key_for_provider(provider, settings))
    source = "session" if runtime_credentials.get_key(provider) else ("environment" if has_key else "none")
    return {"provider": provider, "configured": has_key, "source": source}


@router.post("/providers/test", response_model=ProviderTestResponse)
def test_generation_provider(
    request: ProviderTestRequest,
    settings: Settings = Depends(get_settings),
    ollama: OllamaProvider = Depends(get_ollama),
    huggingface: HuggingFaceProvider = Depends(get_huggingface),
    groq: GroqProvider = Depends(get_groq),
):
    try:
        prov = request.provider
        if prov == "extractive":
            return ProviderTestResponse(ready=True, provider=prov, model=request.model, message="Local extractive generator is ready.")
        elif prov == "huggingface":
            if _key_for_provider("huggingface", settings) and not getattr(huggingface, "token", None):
                huggingface.token = _key_for_provider("huggingface", settings)
            huggingface.generate(
                request.model,
                "Reply with exactly OK.",
                "OK",
                max_tokens=8,
                temperature=0,
                policy=settings.hf_provider_policy,
            )
        elif prov == "groq":
            if _key_for_provider("groq", settings) and not getattr(groq, "api_key", None):
                groq.api_key = _key_for_provider("groq", settings)
            groq.generate(
                request.model, "Reply with exactly OK.", "OK", max_tokens=64, temperature=0
            )
        elif prov == "openai":
            key = _key_for_provider("openai", settings)
            OpenAIProvider(api_key=key).generate(
                request.model, "Reply with exactly OK.", "OK", max_tokens=16, temperature=0
            )
        elif prov == "gemini":
            key = _key_for_provider("gemini", settings)
            GeminiProvider(api_key=key).generate(
                request.model, "Reply with exactly OK.", "OK", max_tokens=16, temperature=0
            )
        elif prov == "anthropic":
            key = _key_for_provider("anthropic", settings)
            AnthropicProvider(api_key=key).generate(
                request.model, "Reply with exactly OK.", "OK", max_tokens=16, temperature=0
            )
        else:
            ollama.generate(
                request.model,
                "Reply with exactly OK.",
                "OK",
                max_tokens=64,
                temperature=0,
            )
    except ProviderError as exc:
        raise PipelineError("provider_test_failed", str(exc), 502) from exc
    return ProviderTestResponse(
        provider=request.provider,
        model=request.model,
        ready=True,
        message="A real provider completion succeeded. This verifies access now, not future availability or quota.",
    )


@router.post(
    "/documents", response_model=DocumentStatusResponse, status_code=status.HTTP_201_CREATED
)
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    store: DocumentStore = Depends(get_document_store),
):
    filename = (file.filename or "").replace("\\", "/").split("/")[-1].strip()
    if not filename or len(filename) > 240:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_filename",
                "message": "A filename of at most 240 characters is required.",
            },
        )
    extension = PurePath(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail={
                "code": "unsupported_file_type",
                "message": "Upload .txt, .md, .markdown, or text-based .pdf files.",
            },
        )
    content = bytearray()
    while True:
        piece = await file.read(1024 * 1024)
        if not piece:
            break
        content.extend(piece)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "file_too_large",
                    "message": f"File exceeds the {settings.max_upload_bytes}-byte upload limit.",
                },
            )
    if not content:
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_upload", "message": "The uploaded file is empty."},
        )
    ext_type = {".txt": "txt", ".md": "md", ".markdown": "markdown", ".pdf": "pdf"}[extension]
    document_id = store.create(filename, ext_type, file.content_type, bytes(content))
    store.update(document_id, status="parsing", error=None)
    try:
        parsed = parse_document(document_id, filename, bytes(content))
        recommendation = recommend_chunking(parsed.profile)
        store.update(
            document_id,
            status="profiled",
            profile=parsed.profile.model_dump(mode="json"),
            recommendation=recommendation.model_dump(mode="json"),
            error=None,
        )
    except DocumentParseError as exc:
        store.update(document_id, status="failed", error=str(exc))
    return _document_response(_find_document(store, document_id))


@router.post(
    "/documents/batch",
    response_model=list[DocumentStatusResponse],
    status_code=status.HTTP_201_CREATED,
)
async def upload_documents_batch(
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
    store: DocumentStore = Depends(get_document_store),
):
    results: list[DocumentStatusResponse] = []
    for file in files:
        filename = (file.filename or "").replace("\\", "/").split("/")[-1].strip()
        if not filename or len(filename) > 240:
            continue
        extension = PurePath(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue
        content = await file.read()
        if not content or len(content) > settings.max_upload_bytes:
            continue
        ext_type = {".txt": "txt", ".md": "md", ".markdown": "markdown", ".pdf": "pdf"}[extension]
        document_id = store.create(filename, ext_type, file.content_type, bytes(content))
        store.update(document_id, status="parsing", error=None)
        try:
            parsed = parse_document(document_id, filename, bytes(content))
            recommendation = recommend_chunking(parsed.profile)
            store.update(
                document_id,
                status="profiled",
                profile=parsed.profile.model_dump(mode="json"),
                recommendation=recommendation.model_dump(mode="json"),
                error=None,
            )
        except DocumentParseError as exc:
            store.update(document_id, status="failed", error=str(exc))
        results.append(_document_response(_find_document(store, document_id)))
    return results


@router.get("/documents", response_model=list[DocumentStatusResponse])
def list_documents(store: DocumentStore = Depends(get_document_store)):
    return [_document_response(row) for row in store.list()]


@router.get("/documents/{document_id}", response_model=DocumentStatusResponse)
def get_document(document_id: str, store: DocumentStore = Depends(get_document_store)):
    return _document_response(_find_document(store, document_id))


@router.get("/documents/{document_id}/profile", response_model=DocumentProfile)
def get_document_profile(document_id: str, store: DocumentStore = Depends(get_document_store)):
    row = _find_document(store, document_id)
    if not row.get("profile"):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "profile_unavailable",
                "message": row.get("error") or "Document profile is not available.",
            },
        )
    return DocumentProfile.model_validate(row["profile"])


@router.post("/documents/{document_id}/recommendation")
def refresh_recommendation(document_id: str, store: DocumentStore = Depends(get_document_store)):
    row = _find_document(store, document_id)
    if not row.get("profile"):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "profile_unavailable",
                "message": "Parse the document before requesting a chunking recommendation.",
            },
        )
    recommendation = recommend_chunking(DocumentProfile.model_validate(row["profile"]))
    store.update(document_id, recommendation=recommendation.model_dump(mode="json"))
    return recommendation


@router.post("/documents/{document_id}/chunking-selection", response_model=DocumentStatusResponse)
def select_chunking(
    document_id: str,
    request: ChunkingSelectionRequest,
    store: DocumentStore = Depends(get_document_store),
):
    row = _find_document(store, document_id)
    selection = request.selection
    if selection.document_id != document_id:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "document_id_mismatch",
                "message": "Selection document_id must match the URL document.",
            },
        )
    recommendation = row.get("recommendation")
    if not recommendation:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "recommendation_unavailable",
                "message": "A document profile and chunking recommendation are required first.",
            },
        )
    if selection.choice_mode == ChunkingChoiceMode.ACCEPT_RECOMMENDATION and (
        selection.source_recommendation_id != recommendation["recommendation_id"]
        or selection.strategy_id != recommendation["recommended_strategy_id"]
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "recommendation_changed",
                "message": "The accepted strategy must match the current recommendation ID and strategy.",
            },
        )
    if selection.strategy_id not in SUPPORTED_STRATEGIES:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_chunking_strategy",
                "message": f"Strategy '{selection.strategy_id}' has no runtime; choose one of {sorted(SUPPORTED_STRATEGIES)}.",
            },
        )
    try:
        validate_settings(selection.strategy_id, selection.settings)
    except ChunkingError as exc:
        raise HTTPException(
            status_code=422, detail={"code": "invalid_chunking_config", "message": str(exc)}
        ) from exc
    stale_chunks = list(
        {
            item["chunk_id"]: item
            for item in [*row.get("stale_chunks", []), *row.get("chunks", [])]
        }.values()
    )
    updated = store.update(
        document_id,
        selection=selection.model_dump(mode="json"),
        status="ready" if row["status"] == "ready" else "profiled",
        stale_chunks=stale_chunks,
        error=None,
    )
    return _document_response(updated)


class IngestionPlanRequest(RAGModel):
    mode: PlanningMode = PlanningMode.USER_APPROVED_AUTO
    approval_override: bool = False


@router.post("/documents/{document_id}/plan-ingestion")
def plan_ingestion(
    document_id: str,
    request: IngestionPlanRequest | None = None,
    store: DocumentStore = Depends(get_document_store),
    settings: Settings = Depends(get_settings),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    row = _find_document(store, document_id)
    if not row.get("profile"):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "profile_unavailable",
                "message": "Parse the document before requesting an ingestion plan.",
            },
        )
    parsed = _parsed(store, row)
    doc_profile = DocumentProfile.model_validate(row["profile"])
    req = request or IngestionPlanRequest()
    plan = IngestionPlanner.plan(
        document=parsed,
        profile=doc_profile,
        mode=req.mode,
        settings=settings,
        existing_vector_store=vector_store,
        approval_override=req.approval_override,
    )
    plan_dict = plan.model_dump(mode="json")
    store.update(document_id, plan=plan_dict)
    return plan_dict


@router.post("/documents/{document_id}/approve-plan", response_model=DocumentStatusResponse)
def approve_plan(
    document_id: str,
    store: DocumentStore = Depends(get_document_store),
):
    row = _find_document(store, document_id)
    plan = row.get("plan")
    if not plan:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "plan_unavailable",
                "message": "No ingestion plan exists for this document to approve.",
            },
        )
    plan["approved"] = True
    plan["requires_approval"] = False

    selected_strategies = plan.get("selected_strategies") or []
    primary = selected_strategies[0] if selected_strategies else None
    strat_id = primary["strategy_id"] if primary else "recursive"
    settings_dict = primary["settings"] if primary else {"chunk_size": 1000, "overlap": 120}

    selection = {
        "document_id": document_id,
        "strategy_id": strat_id,
        "choice_mode": ChunkingChoiceMode.ACCEPT_RECOMMENDATION,
        "source_recommendation_id": plan.get("plan_id"),
        "settings": settings_dict,
    }

    updated = store.update(
        document_id,
        plan=plan,
        selection=selection,
        status="ready" if row["status"] == "ready" else "profiled",
    )
    return _document_response(updated)


@router.post("/documents/{document_id}/chunk-multi")
def chunk_multi(
    document_id: str,
    plan_req: MultiStrategyChunkPlan,
    store: DocumentStore = Depends(get_document_store),
):
    row = _find_document(store, document_id)
    parsed = _parsed(store, row)
    result = chunk_document_multi(parsed, plan_req)
    return result.model_dump(mode="json")


@router.post("/documents/{document_id}/ingest", response_model=IngestionResponse)
def ingest_document(
    document_id: str,
    request: IngestionRequest,
    settings: Settings = Depends(get_settings),
    store: DocumentStore = Depends(get_document_store),
    provider: OllamaProvider = Depends(get_ollama),
    sentence_provider: SentenceTransformersProvider = Depends(get_sentence_transformers),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    _find_document(store, document_id)
    selected_provider = request.generation_provider or settings.generation_provider
    selected_model = _model_for(selected_provider, request.generation_model, settings)
    _require_generation_config(selected_provider, selected_model, settings)
    report = validate_traditional_graph(request.graph, settings, selected_provider, selected_model)
    if not report.valid or not report.executable:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "workflow_not_executable",
                "message": "Traditional RAG graph validation failed.",
                "issues": [issue.model_dump(mode="json") for issue in report.issues],
            },
        )
    embed_node = next(
        (node for node in request.graph.nodes if node.node_type_id == "embedding.create"), None
    )
    embed_provider_name = (
        embed_node.config.get("provider") if embed_node and embed_node.config else None
    )
    embed_model_name = (
        embed_node.config.get("model") if embed_node and embed_node.config else None
    )
    if embed_provider_name == "fast_local":
        active_embedding_provider = FastLocalEmbeddingProvider()
    elif settings.embedding_provider == "ollama":
        active_embedding_provider = provider
        if not embed_model_name or "sentence-transformers" in embed_model_name:
            embed_model_name = settings.ollama_embedding_model or embed_model_name
    elif settings.embedding_provider == "sentence_transformers" or embed_provider_name == "sentence_transformers":
        active_embedding_provider = sentence_provider
        if not embed_model_name:
            embed_model_name = settings.sentence_transformers_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    else:
        active_embedding_provider = provider
        if not embed_model_name:
            embed_model_name = settings.sentence_transformers_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


    _reject_graph_secrets(request.graph)
    vector_node = next(
        (node for node in request.graph.nodes if node.node_type_id == "vector.index"), None
    )
    vector_backend = "chroma"
    collection_name = "visual_rag_studio_documents"
    api_key = None
    url = None
    if vector_node and vector_node.config:
        vector_backend = str(vector_node.config.get("backend") or "chroma")
        collection_name = str(vector_node.config.get("collection_name") or "visual_rag_studio_documents")
        api_key = runtime_credentials.get_key(vector_backend)
        url = vector_node.config.get("url")

    from app.rag.vector_store import get_shared_memory_vector_store
    if vector_backend == "memory":
        active_vector_store = get_shared_memory_vector_store(collection_name)
    elif vector_backend in {"pinecone", "mongodb", "qdrant"}:
        active_vector_store = get_vector_store_adapter(
            backend=vector_backend,
            collection_name=collection_name,
            api_key=api_key,
            url=url,
        )
    else:
        if collection_name == getattr(vector_store, "collection_name", "visual_rag_studio_documents"):
            active_vector_store = vector_store
        else:
            active_vector_store = ChromaVectorStore(settings.chroma_path, collection_name=collection_name)
    chunk_node = next(
        node for node in request.graph.nodes if node.node_type_id == "chunk.split"
    )
    try:
        graph_chunk_settings = validate_settings(
            str(chunk_node.config.get("strategy", "")),
            {
                "chunk_size": chunk_node.config.get("chunk_size"),
                "overlap": chunk_node.config.get("overlap"),
            },
        )
    except ChunkingError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_chunking_config", "message": str(exc)},
        ) from exc
    current_selection = _find_document(store, document_id).get("selection")
    if not current_selection or (
        current_selection.get("strategy_id") != chunk_node.config.get("strategy")
        or current_selection.get("settings") != graph_chunk_settings
    ):
        store.update(
            document_id,
            selection=ChunkingSelection(
                document_id=document_id,
                strategy_id=str(chunk_node.config["strategy"]),
                choice_mode=ChunkingChoiceMode.EXPLICIT_OVERRIDE,
                settings=graph_chunk_settings,
            ).model_dump(mode="json"),
        )
    execution_id = request.execution_id or uuid4().hex
    source_node_id = next((n.id for n in request.graph.nodes if n.node_type_id == "document.source"), "source")
    all_index_node_ids = list({*INDEX_NODE_IDS, *(n.id for n in request.graph.nodes)})
    execution_tracker.start(execution_id, all_index_node_ids, list({"source", source_node_id}))
    try:
        result = index_document(
            document_id,
            store,
            active_vector_store,
            active_embedding_provider,
            settings,
            on_step=_build_step_tracker(execution_id, request.graph.nodes),
            embedding_model=embed_model_name,
        )
    except Exception as exc:
        message = str(exc) if isinstance(exc, PipelineError) else "Workflow execution failed."
        execution_tracker.fail(execution_id, message)
        raise
    execution_tracker.finish(execution_id)
    return IngestionResponse(**result)


@router.post("/documents/ingest-batch")
def ingest_documents_batch(
    request: IngestionRequest,
    settings: Settings = Depends(get_settings),
    store: DocumentStore = Depends(get_document_store),
    provider: OllamaProvider = Depends(get_ollama),
    sentence_provider: SentenceTransformersProvider = Depends(get_sentence_transformers),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    selected_provider = request.generation_provider or settings.generation_provider
    selected_model = _model_for(selected_provider, request.generation_model, settings)
    _require_generation_config(selected_provider, selected_model, settings)
    report = validate_traditional_graph(request.graph, settings, selected_provider, selected_model)
    if not report.valid or not report.executable:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "workflow_not_executable",
                "message": "Traditional RAG graph validation failed.",
                "issues": [issue.model_dump(mode="json") for issue in report.issues],
            },
        )
    embed_node = next(
        (node for node in request.graph.nodes if node.node_type_id == "embedding.create"), None
    )
    embed_provider_name = (
        embed_node.config.get("provider") if embed_node and embed_node.config else None
    )
    embed_model_name = (
        embed_node.config.get("model") if embed_node and embed_node.config else None
    )
    if embed_provider_name == "fast_local":
        active_embedding_provider = FastLocalEmbeddingProvider()
    elif settings.embedding_provider == "ollama":
        active_embedding_provider = provider
        if not embed_model_name or "sentence-transformers" in embed_model_name:
            embed_model_name = settings.ollama_embedding_model or embed_model_name
    elif settings.embedding_provider == "sentence_transformers" or embed_provider_name == "sentence_transformers":
        active_embedding_provider = sentence_provider
        if not embed_model_name:
            embed_model_name = settings.sentence_transformers_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    else:
        active_embedding_provider = provider
        if not embed_model_name:
            embed_model_name = settings.sentence_transformers_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


    _reject_graph_secrets(request.graph)
    vector_node = next(
        (node for node in request.graph.nodes if node.node_type_id == "vector.index"), None
    )
    vector_backend = "chroma"
    collection_name = "visual_rag_studio_documents"
    api_key = None
    url = None
    if vector_node and vector_node.config:
        vector_backend = str(vector_node.config.get("backend") or "chroma")
        collection_name = str(vector_node.config.get("collection_name") or "visual_rag_studio_documents")
        api_key = runtime_credentials.get_key(vector_backend)
        url = vector_node.config.get("url")

    from app.rag.vector_store import get_shared_memory_vector_store
    if vector_backend == "memory":
        active_vector_store = get_shared_memory_vector_store(collection_name)
    elif vector_backend in {"pinecone", "mongodb", "qdrant"}:
        active_vector_store = get_vector_store_adapter(
            backend=vector_backend,
            collection_name=collection_name,
            api_key=api_key,
            url=url,
        )
    else:
        if collection_name == getattr(vector_store, "collection_name", "visual_rag_studio_documents"):
            active_vector_store = vector_store
        else:
            active_vector_store = ChromaVectorStore(settings.chroma_path, collection_name=collection_name)

    chunk_node = next(
        node for node in request.graph.nodes if node.node_type_id == "chunk.split"
    )
    graph_chunk_settings = validate_settings(
        str(chunk_node.config.get("strategy", "")),
        {
            "chunk_size": chunk_node.config.get("chunk_size"),
            "overlap": chunk_node.config.get("overlap"),
        },
    )

    all_docs = store.list()
    target_ids = request.document_ids or [
        d["document_id"] for d in all_docs if d.get("status") in {"profiled", "ready", "uploaded"}
    ]
    if not target_ids and all_docs:
        target_ids = [d["document_id"] for d in all_docs]

    execution_id = request.execution_id or uuid4().hex
    source_node_id = next((n.id for n in request.graph.nodes if n.node_type_id == "document.source"), "source")
    all_index_node_ids = list({*INDEX_NODE_IDS, *(n.id for n in request.graph.nodes)})
    execution_tracker.start(execution_id, all_index_node_ids, list({"source", source_node_id}))
    batch_step_tracker = _build_step_tracker(execution_id, request.graph.nodes)
    results = []
    try:
        for doc_id in target_ids:
            doc_row = store.get(doc_id)
            if not doc_row:
                continue
            cur_selection = doc_row.get("selection")
            if not cur_selection or (
                cur_selection.get("strategy_id") != chunk_node.config.get("strategy")
                or cur_selection.get("settings") != graph_chunk_settings
            ):
                store.update(
                    doc_id,
                    selection=ChunkingSelection(
                        document_id=doc_id,
                        strategy_id=str(chunk_node.config["strategy"]),
                        choice_mode=ChunkingChoiceMode.EXPLICIT_OVERRIDE,
                        settings=graph_chunk_settings,
                    ).model_dump(mode="json"),
                )
            res = index_document(
                doc_id,
                store,
                active_vector_store,
                active_embedding_provider,
                settings,
                on_step=batch_step_tracker,
                embedding_model=embed_model_name,
            )
            results.append(res)
    except Exception as exc:
        message = str(exc) if isinstance(exc, PipelineError) else "Workflow execution failed."
        execution_tracker.fail(execution_id, message)
        raise
    execution_tracker.finish(execution_id)
    return {"indexed_count": len(results), "documents": results}


@router.post("/query", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    settings: Settings = Depends(get_settings),
    store: DocumentStore = Depends(get_document_store),
    provider: OllamaProvider = Depends(get_ollama),
    sentence_provider: SentenceTransformersProvider = Depends(get_sentence_transformers),
    huggingface: HuggingFaceProvider = Depends(get_huggingface),
    groq: GroqProvider = Depends(get_groq),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    selected_provider = request.generation_provider or settings.generation_provider
    selected_model = _model_for(selected_provider, request.generation_model, settings)
    _require_generation_config(selected_provider, selected_model, settings)
    report = validate_traditional_graph(request.graph, settings, selected_provider, selected_model)
    if not report.valid or not report.executable:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "workflow_not_executable",
                "message": "Traditional RAG graph validation failed.",
                "issues": [issue.model_dump(mode="json") for issue in report.issues],
            },
        )
    has_agent = any(n.node_type_id == "agent.orchestrate" for n in request.graph.nodes)
    if has_agent:
        agent_node = next((n for n in request.graph.nodes if n.node_type_id == "agent.orchestrate"), None)
        cfg = agent_node.config if agent_node else {}
        agent_req = AgentRunRequest(
            query=request.question,
            graph=request.graph,
            goal=str(cfg["goal"]) if cfg.get("goal") else None,
            instructions=str(cfg["instructions"]) if cfg.get("instructions") else None,
            mode=cfg.get("mode") if cfg.get("mode") in ("fixed", "conditional", "bounded_loop") else None,
            max_steps=int(cfg["max_steps"]) if cfg.get("max_steps") is not None else None,
            max_retrieval_iterations=int(cfg["max_retrieval_iterations"]) if cfg.get("max_retrieval_iterations") is not None else None,
            token_budget=int(cfg["token_budget"]) if cfg.get("token_budget") is not None else None,
            timeout_seconds=float(cfg["timeout_seconds"]) if cfg.get("timeout_seconds") is not None else None,
            approval_policy=cfg.get("approval_policy") if cfg.get("approval_policy") in ("always_ask", "on_uncertainty", "fully_autonomous") else None,
            allowed_tools=list(cfg["allowed_tools"]) if cfg.get("allowed_tools") else None,
            document_ids=request.document_ids,
            generation_provider=request.generation_provider,
            generation_model=request.generation_model,
            initial_strategy="hybrid",
        )
        agent_res = run_agent(
            agent_req,
            settings=settings,
            store=store,
            provider=provider,
            sentence_provider=sentence_provider,
            huggingface=huggingface,
            groq=groq,
            vector_store=vector_store,
        )
        status_val: Literal["completed", "abstained", "authorization_required", "generation_failed"] = (
            "abstained" if agent_res.abstained
            else "authorization_required" if agent_res.status == "authorization_required"
            else "generation_failed" if agent_res.status in ("generation_failed", "provider_error")
            else "completed"
        )
        return QueryResponse(
            question=request.question,
            answer=agent_res.answer,
            abstained=agent_res.abstained,
            citations=agent_res.citations,
            retrieved_chunks=agent_res.citations,
            embedding_model=agent_res.embedding_model or "fast_local",
            generation_model=selected_model,
            execution_id=agent_res.execution_id,
            status=status_val,
            generation_provider=selected_provider,
            generation_error=agent_res.error,
            execution_started_at=datetime.now(UTC),
            execution_completed_at=datetime.now(UTC),
        )
    if selected_provider == "groq":
        generation_provider = groq
        if _key_for_provider("groq", settings) and not getattr(groq, "api_key", None):
            groq.api_key = _key_for_provider("groq", settings)
    elif selected_provider == "huggingface":
        generation_provider = huggingface
        if _key_for_provider("huggingface", settings) and not getattr(huggingface, "token", None):
            huggingface.token = _key_for_provider("huggingface", settings)
    elif selected_provider == "openai":
        generation_provider = OpenAIProvider(api_key=_key_for_provider("openai", settings))
    elif selected_provider == "gemini":
        generation_provider = GeminiProvider(api_key=_key_for_provider("gemini", settings))
    elif selected_provider == "anthropic":
        generation_provider = AnthropicProvider(api_key=_key_for_provider("anthropic", settings))
    elif selected_provider == "extractive":
        generation_provider = ExtractiveGeneratorProvider()
    elif selected_provider == "ollama":
        generation_provider = provider
    else:
        generation_provider = provider

    embed_node = next(
        (node for node in request.graph.nodes if node.node_type_id == "embedding.create"), None
    )
    embed_provider_name = (
        embed_node.config.get("provider") if embed_node and embed_node.config else None
    )
    embed_model_name = (
        embed_node.config.get("model") if embed_node and embed_node.config else None
    )
    if embed_provider_name == "fast_local":
        active_embedding_provider = FastLocalEmbeddingProvider()
    elif settings.embedding_provider == "ollama":
        active_embedding_provider = provider
        if not embed_model_name or "sentence-transformers" in embed_model_name:
            embed_model_name = settings.ollama_embedding_model or embed_model_name
    elif settings.embedding_provider == "sentence_transformers" or embed_provider_name == "sentence_transformers":
        active_embedding_provider = sentence_provider
        if not embed_model_name:
            embed_model_name = settings.sentence_transformers_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    else:
        active_embedding_provider = provider
        if not embed_model_name:
            embed_model_name = settings.sentence_transformers_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


    _reject_graph_secrets(request.graph)
    vector_node = next(
        (node for node in request.graph.nodes if node.node_type_id == "vector.index"), None
    )
    vector_backend = "chroma"
    collection_name = "visual_rag_studio_documents"
    api_key = None
    url = None
    if vector_node and vector_node.config:
        vector_backend = str(vector_node.config.get("backend") or "chroma")
        collection_name = str(vector_node.config.get("collection_name") or "visual_rag_studio_documents")
        api_key = runtime_credentials.get_key(vector_backend)
        url = vector_node.config.get("url")

    from app.rag.vector_store import get_shared_memory_vector_store
    if vector_backend == "memory":
        active_vector_store = get_shared_memory_vector_store(collection_name)
    elif vector_backend in {"pinecone", "mongodb", "qdrant"}:
        active_vector_store = get_vector_store_adapter(
            backend=vector_backend,
            collection_name=collection_name,
            api_key=api_key,
            url=url,
        )
    else:
        if collection_name == getattr(vector_store, "collection_name", "visual_rag_studio_documents"):
            active_vector_store = vector_store
        else:
            active_vector_store = ChromaVectorStore(settings.chroma_path, collection_name=collection_name)

    rerank_node = next(
        (node for node in request.graph.nodes if node.node_type_id == "retrieval.rerank"), None
    )
    reranker_instance = None
    reranker_top_k = request.top_k
    if rerank_node:
        model_name = rerank_node.config.get("model", "flash") if rerank_node.config else "flash"
        reranker_top_k = int(rerank_node.config.get("top_k", request.top_k)) if rerank_node.config else request.top_k
        reranker_instance = RagSparkReranker(model=str(model_name))

    retrieval_dense_node = next(
        (node for node in request.graph.nodes if node.node_type_id == "retrieval.dense"), None
    )
    retrieval_sparse_node = next(
        (node for node in request.graph.nodes if node.node_type_id == "retrieval.sparse"), None
    )
    fuse_node = next(
        (node for node in request.graph.nodes if node.node_type_id == "merge.fuse"), None
    )

    retrieval_strat = "semantic"
    if fuse_node or (retrieval_dense_node and str(retrieval_dense_node.config.get("strategy")) in ("hybrid", "rrf")):
        retrieval_strat = "hybrid"
    elif retrieval_sparse_node and not retrieval_dense_node or retrieval_dense_node and str(retrieval_dense_node.config.get("strategy")) in ("bm25", "keyword"):
        retrieval_strat = "bm25"
    elif retrieval_dense_node:
        retrieval_strat = str(retrieval_dense_node.config.get("strategy", "semantic"))

    fusion_method = "rrf"
    if fuse_node and fuse_node.config.get("fusion_method"):
        fusion_method = str(fuse_node.config.get("fusion_method"))
    elif retrieval_dense_node and retrieval_dense_node.config.get("fusion_method"):
        fusion_method = str(retrieval_dense_node.config.get("fusion_method"))

    rrf_k = 60
    if fuse_node and fuse_node.config.get("rrf_k"):
        rrf_k = int(fuse_node.config.get("rrf_k"))
    elif retrieval_dense_node and retrieval_dense_node.config.get("rrf_k"):
        rrf_k = int(retrieval_dense_node.config.get("rrf_k"))

    alpha = 0.65
    if retrieval_dense_node and retrieval_dense_node.config.get("alpha") is not None:
        alpha = float(retrieval_dense_node.config.get("alpha"))
    elif fuse_node and fuse_node.config.get("dense_weight") is not None:
        alpha = float(fuse_node.config.get("dense_weight"))

    dense_top_k = int(retrieval_dense_node.config.get("dense_top_k", retrieval_dense_node.config.get("top_k", 10))) if retrieval_dense_node else 10
    sparse_top_k = int(retrieval_sparse_node.config.get("top_k", 10)) if retrieval_sparse_node else (
        int(retrieval_dense_node.config.get("sparse_top_k", 10)) if retrieval_dense_node else 10
    )
    k1 = float(retrieval_sparse_node.config.get("k1", 1.5)) if retrieval_sparse_node else (
        float(retrieval_dense_node.config.get("k1", 1.5)) if retrieval_dense_node else 1.5
    )
    b = float(retrieval_sparse_node.config.get("b", 0.75)) if retrieval_sparse_node else (
        float(retrieval_dense_node.config.get("b", 0.75)) if retrieval_dense_node else 0.75
    )

    context_node = next(
        (node for node in request.graph.nodes if node.node_type_id == "context.assemble"), None
    )
    generation_node = next(
        (node for node in request.graph.nodes if node.node_type_id == "generation.answer"), None
    )
    preserve_source_language = True
    response_language = "query"
    if context_node and context_node.config:
        preserve_source_language = bool(context_node.config.get("preserve_source_language", True))
        if context_node.config.get("response_language"):
            response_language = str(context_node.config.get("response_language"))
    if response_language == "query" and generation_node and generation_node.config and generation_node.config.get("response_language"):
        response_language = str(generation_node.config.get("response_language"))

    query_language_handling = "auto"
    cross_lingual = True
    if retrieval_dense_node and retrieval_dense_node.config:
        if retrieval_dense_node.config.get("query_language_handling"):
            query_language_handling = str(retrieval_dense_node.config.get("query_language_handling"))
        if "cross_lingual" in retrieval_dense_node.config:
            cross_lingual = bool(retrieval_dense_node.config.get("cross_lingual"))

    execution_id = request.execution_id or uuid4().hex
    started_at = datetime.now(UTC)
    query_node_id = next((n.id for n in request.graph.nodes if n.node_type_id == "query.input"), "query")
    all_query_node_ids = list({*QUERY_NODE_IDS, *(n.id for n in request.graph.nodes)})
    execution_tracker.start(execution_id, all_query_node_ids, list({"query", query_node_id}))
    try:
        result = answer_query(
            request.question,
            request.top_k,
            request.document_ids,
            active_vector_store,
            active_embedding_provider,
            settings,
            store,
            generation_provider=generation_provider,
            generation_model=selected_model,
            allow_external_processing=request.allow_external_processing,
            generation_provider_id=selected_provider,
            on_step=_build_step_tracker(execution_id, request.graph.nodes),
            reranker=reranker_instance,
            reranker_top_k=reranker_top_k,
            embedding_model=embed_model_name,
            retrieval_strategy=retrieval_strat,
            fusion_method=fusion_method,
            rrf_k=rrf_k,
            alpha=alpha,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            k1=k1,
            b=b,
            response_language=response_language,
            query_language_handling=query_language_handling,
            preserve_source_language=preserve_source_language,
            cross_lingual=cross_lingual,
        )
    except (VectorStoreError, PipelineError) as exc:
        execution_tracker.fail(execution_id, str(exc))
        raise
    if result.get("status") == "generation_failed":
        execution_tracker.fail(execution_id, result.get("generation_error", "Generation failed."))
    else:
        execution_tracker.finish(execution_id)
    return QueryResponse(
        **result,
        question=request.question,
        execution_id=execution_id,
        generation_provider=selected_provider,
        execution_started_at=started_at,
        execution_completed_at=datetime.now(UTC),
    )


@router.post("/agent/run", response_model=AgentRunResponse)
def run_agent(
    request: AgentRunRequest,
    settings: Settings = Depends(get_settings),
    store: DocumentStore = Depends(get_document_store),
    provider: OllamaProvider = Depends(get_ollama),
    sentence_provider: SentenceTransformersProvider = Depends(get_sentence_transformers),
    huggingface: HuggingFaceProvider = Depends(get_huggingface),
    groq: GroqProvider = Depends(get_groq),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    selected_provider = request.generation_provider or settings.generation_provider
    selected_model = _model_for(selected_provider, request.generation_model, settings)
    _require_generation_config(selected_provider, selected_model, settings)
    report = validate_traditional_graph(request.graph, settings, selected_provider, selected_model)
    if not report.valid or not report.executable:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "workflow_not_executable",
                "message": "Agent graph validation failed.",
                "issues": [issue.model_dump(mode="json") for issue in report.issues],
            },
        )

    if selected_provider == "groq":
        generation_provider = groq
        if _key_for_provider("groq", settings) and not getattr(groq, "api_key", None):
            groq.api_key = _key_for_provider("groq", settings)
    elif selected_provider == "huggingface":
        generation_provider = huggingface
        if _key_for_provider("huggingface", settings) and not getattr(huggingface, "token", None):
            huggingface.token = _key_for_provider("huggingface", settings)
    elif selected_provider == "openai":
        generation_provider = OpenAIProvider(api_key=_key_for_provider("openai", settings))
    elif selected_provider == "gemini":
        generation_provider = GeminiProvider(api_key=_key_for_provider("gemini", settings))
    elif selected_provider == "anthropic":
        generation_provider = AnthropicProvider(api_key=_key_for_provider("anthropic", settings))
    elif selected_provider == "extractive":
        generation_provider = ExtractiveGeneratorProvider()
    elif selected_provider == "ollama":
        generation_provider = provider
    else:
        generation_provider = provider

    embed_node = next(
        (node for node in request.graph.nodes if node.node_type_id == "embedding.create"), None
    )
    embed_provider_name = (
        embed_node.config.get("provider") if embed_node and embed_node.config else None
    )
    embed_model_name = (
        embed_node.config.get("model") if embed_node and embed_node.config else None
    )
    if embed_provider_name == "fast_local":
        active_embedding_provider = FastLocalEmbeddingProvider()
    elif settings.embedding_provider == "ollama":
        active_embedding_provider = provider
        if not embed_model_name or "sentence-transformers" in embed_model_name:
            embed_model_name = settings.ollama_embedding_model or embed_model_name
    elif settings.embedding_provider == "sentence_transformers" or embed_provider_name == "sentence_transformers":
        active_embedding_provider = sentence_provider
        if not embed_model_name:
            embed_model_name = settings.sentence_transformers_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    else:
        active_embedding_provider = provider
        if not embed_model_name:
            embed_model_name = settings.sentence_transformers_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    _reject_graph_secrets(request.graph)
    vector_node = next(
        (node for node in request.graph.nodes if node.node_type_id == "vector.index"), None
    )
    vector_backend = "chroma"
    collection_name = "visual_rag_studio_documents"
    api_key = None
    url = None
    if vector_node and vector_node.config:
        vector_backend = str(vector_node.config.get("backend") or "chroma")
        collection_name = str(vector_node.config.get("collection_name") or "visual_rag_studio_documents")
        api_key = runtime_credentials.get_key(vector_backend)
        url = vector_node.config.get("url")

    from app.rag.vector_store import get_shared_memory_vector_store
    if vector_backend == "memory":
        active_vector_store = get_shared_memory_vector_store(collection_name)
    elif vector_backend in {"pinecone", "mongodb", "qdrant"}:
        active_vector_store = get_vector_store_adapter(
            backend=vector_backend,
            collection_name=collection_name,
            api_key=api_key,
            url=url,
        )
    else:
        if collection_name == getattr(vector_store, "collection_name", "visual_rag_studio_documents"):
            active_vector_store = vector_store
        else:
            active_vector_store = ChromaVectorStore(settings.chroma_path, collection_name=collection_name)

    agent_node = next((n for n in request.graph.nodes if n.node_type_id == "agent.orchestrate"), None)
    agent_config = agent_node.config if agent_node else {}

    goal = request.goal or agent_config.get("goal") or "Answer the query factually based on retrieved evidence."
    instructions = request.instructions or agent_config.get("instructions") or ""
    mode_str = request.mode or agent_config.get("mode") or "bounded_loop"
    raw_max_steps = request.max_steps or agent_config.get("max_steps") or 5
    max_steps = int(raw_max_steps)
    max_retrieval_iter = request.max_retrieval_iterations or agent_config.get("max_retrieval_iterations") or 3
    token_budget = request.token_budget or agent_config.get("token_budget") or 4096
    timeout_sec = request.timeout_seconds or agent_config.get("timeout_seconds") or 60.0
    approval_pol = request.approval_policy or agent_config.get("approval_policy") or "on_uncertainty"
    allowed_tools = request.allowed_tools or agent_config.get("allowed_tools")

    execution_id = request.execution_id or uuid4().hex
    query_node_id = next((n.id for n in request.graph.nodes if n.node_type_id == "query.input"), "query")
    agent_node_id = next((n.id for n in request.graph.nodes if n.node_type_id == "agent.orchestrate"), "agent")
    all_query_node_ids = list({*QUERY_NODE_IDS, *(n.id for n in request.graph.nodes), "agent", agent_node_id})
    execution_tracker.start(execution_id, all_query_node_ids, list({"query", query_node_id}))
    tracker_step = _build_step_tracker(execution_id, request.graph.nodes)
    tracker_step("agent", "running")

    orchestrator = AgentOrchestrator(
        goal=goal,
        instructions=instructions,
        mode=AgentExecutionMode(mode_str),
        max_steps=max_steps,
        max_retrieval_iterations=int(max_retrieval_iter),
        token_budget=int(token_budget),
        timeout_seconds=float(timeout_sec),
        approval_policy=AgentApprovalPolicy(approval_pol),
        allowed_tools=allowed_tools,
    )

    try:
        result = orchestrator.run(
            query=request.query,
            vector_store=active_vector_store,
            provider=active_embedding_provider,
            settings=settings,
            store=store,
            generation_provider=generation_provider,
            generation_model=selected_model,
            document_ids=request.document_ids,
            embedding_model=embed_model_name,
            initial_strategy=request.initial_strategy,
            allow_external_processing=request.allow_external_processing,
            on_step=tracker_step,
        )
    except Exception as exc:
        tracker_step("agent", "failed")
        execution_tracker.fail(execution_id, str(exc))
        raise

    tracker_step("agent", "completed")
    execution_tracker.finish(execution_id)
    result["execution_id"] = execution_id

    return AgentRunResponse(**result)


@router.get("/documents/{document_id}/chunks", response_model=DocumentChunkList)
def get_indexed_chunks(
    document_id: str,
    store: DocumentStore = Depends(get_document_store),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    row = _find_document(store, document_id)
    if row["status"] != "ready":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "document_not_ready",
                "message": "Only indexed documents have retrievable chunks.",
            },
        )
    results = vector_store.get_document(document_id)
    from app.rag.pipeline import _citation

    return DocumentChunkList(
        document_id=document_id, chunks=[_citation({**item, "distance": None}) for item in results]
    )


@router.get("/storage/summary", response_model=StorageSummary)
def storage_summary(
    store: DocumentStore = Depends(get_document_store),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    global _last_storage_success
    checked_at = datetime.now(UTC)
    metadata = store.storage_summary()
    try:
        vectors = vector_store.storage_summary()
    except VectorStoreError as exc:
        return StorageSummary(
            metadata_backend="SQLite + SQLAlchemy",
            metadata_location="Configured application database (absolute path hidden)",
            metadata_document_count=metadata["document_count"],
            source_files_present=metadata["source_files_present"],
            vector_backend="Chroma PersistentClient",
            vector_persistence="Configured local persistent directory (absolute path hidden)",
            vector_store_healthy=False,
            checked_at=checked_at,
            last_successful_check=_last_storage_success,
            error=str(exc),
        )
    _last_storage_success = checked_at
    return StorageSummary(
        metadata_backend="SQLite + SQLAlchemy",
        metadata_location="Configured application database (absolute path hidden)",
        metadata_document_count=metadata["document_count"],
        source_files_present=metadata["source_files_present"],
        vector_backend="Chroma PersistentClient",
        vector_persistence="Configured local persistent directory (absolute path hidden)",
        vector_store_healthy=vectors["healthy"],
        vector_count=vectors["count"],
        collections=[StorageCollection(**item) for item in vectors["collections"]],
        checked_at=checked_at,
        last_successful_check=_last_storage_success,
    )


@router.get("/chunks", response_model=ChunkPage)
def list_stored_chunks(
    limit: int = 50,
    offset: int = 0,
    document_id: str | None = None,
    store: DocumentStore = Depends(get_document_store),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    if not 1 <= limit <= 100 or offset < 0:
        raise HTTPException(status_code=422, detail="Use limit 1–100 and a non-negative offset.")
    if document_id:
        row = _find_document(store, document_id)
        if row["status"] != "ready":
            raise HTTPException(status_code=409, detail="Only indexed documents have stored chunks.")
    total, rows = vector_store.list_chunks(limit, offset, document_id)
    documents = {row["document_id"]: row for row in store.list()}
    records = []
    for item in rows:
        metadata = item["metadata"]
        document = documents.get(metadata.get("document_id"), {})
        section = metadata.get("section_path", "")
        records.append(ChunkRecord(
            chunk_id=item["id"],
            document_id=metadata.get("document_id", ""),
            document_name=document.get("filename") or metadata.get("document_name", "Unknown source"),
            ordinal=int(metadata.get("ordinal", 0)),
            strategy_id=metadata.get("strategy_id"),
            page_number=metadata.get("page_number"),
            section_path=[part.strip() for part in section.split("/") if part.strip()],
            source_start=metadata.get("source_start"),
            source_end=metadata.get("source_end"),
            text=item["text"],
            embedding_available=item["embedding_available"],
            embedding_dimension=item["embedding_dimension"],
        ))
    return ChunkPage(total=total, limit=limit, offset=offset, chunks=records)


@router.get("/documents/{document_id}/preview")
def preview_document(document_id: str, store: DocumentStore = Depends(get_document_store)):
    row = _find_document(store, document_id)
    try:
        parsed = parse_document(document_id, row["filename"], store.source_bytes(document_id))
    except (DocumentParseError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail="Stored document preview is unavailable.") from exc
    text = "\n\n".join(segment.text for segment in parsed.segments)
    return {
        "document_id": document_id,
        "filename": row["filename"],
        "text": text[:20000],
        "truncated": len(text) > 20000,
        "page_count": parsed.profile.structure.page_count.value,
    }


@router.post("/workflows", response_model=WorkflowResponse)
def save_workflow(
    request: WorkflowSaveRequest,
    store: DocumentStore = Depends(get_document_store),
):
    _reject_graph_secrets(request.graph)
    workflow_id = request.id or f"wf_{uuid4().hex[:12]}"
    saved = store.save_workflow(
        workflow_id=workflow_id,
        name=request.name,
        graph=request.graph.model_dump(mode="json"),
    )
    return WorkflowResponse(**saved)


@router.get("/workflows", response_model=list[WorkflowResponse])
def list_workflows(
    store: DocumentStore = Depends(get_document_store),
):
    return [WorkflowResponse(**item) for item in store.list_workflows()]


@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
def get_workflow(
    workflow_id: str,
    store: DocumentStore = Depends(get_document_store),
):
    row = store.get_workflow(workflow_id)
    if not row:
        raise HTTPException(
            status_code=404,
            detail={"code": "workflow_not_found", "message": "Saved workflow was not found."},
        )
    return WorkflowResponse(**row)


@router.delete("/workflows/{workflow_id}")
def delete_workflow(
    workflow_id: str,
    store: DocumentStore = Depends(get_document_store),
):
    deleted = store.delete_workflow(workflow_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={"code": "workflow_not_found", "message": "Saved workflow was not found."},
        )
    return {"deleted": True, "workflow_id": workflow_id}


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document_endpoint(
    document_id: str,
    store: DocumentStore = Depends(get_document_store),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
):
    row = store.get(document_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "document_not_found", "message": "Document was not found."},
        )
    target_vs = _resolve_document_vector_store(row, vector_store, settings)
    chunks_removed = 0
    try:
        chunks_removed = target_vs.delete_document(document_id)
    except Exception:
        pass
    if target_vs != vector_store:
        try:
            vector_store.delete_document(document_id)
        except Exception:
            pass
    deleted = store.delete(document_id)
    return DeleteDocumentResponse(
        deleted=deleted,
        document_id=document_id,
        filename=row.get("filename"),
        chunks_removed=chunks_removed,
    )


@router.delete("/documents/{document_id}/chunks", response_model=DeleteChunksResponse)
def delete_document_chunks_endpoint(
    document_id: str,
    store: DocumentStore = Depends(get_document_store),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
):
    row = store.get(document_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "document_not_found", "message": "Document was not found."},
        )
    target_vs = _resolve_document_vector_store(row, vector_store, settings)
    chunks_removed = 0
    try:
        chunks_removed = target_vs.delete_document(document_id)
    except Exception:
        pass
    if target_vs != vector_store:
        try:
            vector_store.delete_document(document_id)
        except Exception:
            pass
    updated = store.clear_chunks(document_id)
    return DeleteChunksResponse(
        cleared=True,
        document_id=document_id,
        chunks_removed=chunks_removed,
        message=f"Cleared {chunks_removed} chunks for document {row.get('filename')}.",
        document=_document_response(updated) if updated else None,
    )


@router.delete("/documents", response_model=DeleteBulkResponse)
def delete_documents_endpoint(
    document_ids: list[str] | None = Query(default=None),
    store: DocumentStore = Depends(get_document_store),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
):
    if document_ids:
        deleted_count = 0
        chunks_removed = 0
        for doc_id in document_ids:
            row = store.get(doc_id)
            if row:
                target_vs = _resolve_document_vector_store(row, vector_store, settings)
                try:
                    chunks_removed += target_vs.delete_document(doc_id)
                except Exception:
                    pass
                if target_vs != vector_store:
                    try:
                        vector_store.delete_document(doc_id)
                    except Exception:
                        pass
                if store.delete(doc_id):
                    deleted_count += 1
        return DeleteBulkResponse(
            deleted_count=deleted_count,
            chunks_removed=chunks_removed,
            message=f"Successfully deleted {deleted_count} document(s) and {chunks_removed} vector chunk(s).",
        )
    # Clear all documents
    chunks_removed = 0
    try:
        chunks_removed = vector_store.delete_all()
    except Exception:
        pass
    deleted_count = store.delete_all()
    return DeleteBulkResponse(
        deleted_count=deleted_count,
        chunks_removed=chunks_removed,
        message="All uploaded documents, source files, and index chunks have been deleted.",
    )


@router.delete("/chunks/{chunk_id}", response_model=DeleteChunkResponse)
def delete_single_chunk_endpoint(
    chunk_id: str,
    store: DocumentStore = Depends(get_document_store),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    try:
        res = vector_store.collection.get(ids=[chunk_id], include=["metadatas"])
        metadatas = res.get("metadatas", [])
        if metadatas and metadatas[0]:
            doc_id = metadatas[0].get("document_id")
            if doc_id:
                store.remove_chunk_from_document(doc_id, chunk_id)
    except Exception:
        pass
    deleted = vector_store.delete_chunks([chunk_id])
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={"code": "chunk_not_found", "message": "Chunk was not found or already deleted."},
        )
    return DeleteChunkResponse(deleted=True, chunk_id=chunk_id)


@router.delete("/chunks", response_model=DeleteChunksResponse)
def clear_all_chunks_endpoint(
    document_id: str | None = Query(default=None),
    store: DocumentStore = Depends(get_document_store),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
):
    if document_id:
        row = store.get(document_id)
        if not row:
            raise HTTPException(
                status_code=404,
                detail={"code": "document_not_found", "message": "Document was not found."},
            )
        target_vs = _resolve_document_vector_store(row, vector_store, settings)
        chunks_removed = 0
        try:
            chunks_removed = target_vs.delete_document(document_id)
        except Exception:
            pass
        if target_vs != vector_store:
            try:
                vector_store.delete_document(document_id)
            except Exception:
                pass
        updated = store.clear_chunks(document_id)
        return DeleteChunksResponse(
            cleared=True,
            document_id=document_id,
            chunks_removed=chunks_removed,
            message=f"Cleared {chunks_removed} chunks for document {row.get('filename')}.",
            document=_document_response(updated) if updated else None,
        )
    # Clear all chunks across all documents
    chunks_removed = 0
    try:
        chunks_removed = vector_store.delete_all()
    except Exception:
        pass
    for row in store.list():
        store.clear_chunks(row["document_id"])
    return DeleteChunksResponse(
        cleared=True,
        chunks_removed=chunks_removed,
        message=f"Cleared all {chunks_removed} vector chunks and reset document chunk statuses.",
    )


@router.post("/storage/cleanup", response_model=StorageCleanupResponse)
def cleanup_storage_endpoint(
    store: DocumentStore = Depends(get_document_store),
):
    result = store.cleanup_orphan_files()
    return StorageCleanupResponse(
        cleaned_count=result["cleaned_count"],
        cleaned_files=result["cleaned_files"],
    )

