from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.contracts import (
    ChunkingSelection,
    ChunkingStrategyRecommendation,
    DocumentProfile,
    WorkflowGraph,
)


class RAGModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentStatusResponse(RAGModel):
    document_id: str
    filename: str
    file_type: str
    status: str
    created_at: datetime
    profile: DocumentProfile | None = None
    recommendation: ChunkingStrategyRecommendation | None = None
    plan: dict | None = None
    selection: ChunkingSelection | None = None
    chunk_count: int = 0
    error: str | None = None


class ChunkingSelectionRequest(RAGModel):
    selection: ChunkingSelection


class IngestionRequest(RAGModel):
    graph: WorkflowGraph
    document_ids: list[str] | None = None
    execution_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    generation_provider: Literal["groq", "ollama", "huggingface", "extractive", "openai", "gemini", "anthropic"] | None = None
    generation_model: str | None = Field(
        default=None, min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_./:-]+$"
    )


class IngestionResponse(RAGModel):
    document_id: str
    status: str
    chunk_count: int
    strategy_id: str
    embedding_model: str
    vector_store: str = "chroma-persistent"


class SourceCitation(RAGModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None = None
    section_path: list[str] = Field(default_factory=list)
    excerpt: str
    text: str = ""
    distance: float | None = None
    distance_metric: Literal["cosine"] | None = None
    rank: int | None = None
    retrieval_strategy: str | None = None
    fallback_reason: str | None = None
    requested_strategy: str | None = None
    score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    weighted_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None


class QueryRequest(RAGModel):
    question: str = Field(min_length=1, max_length=10000)
    graph: WorkflowGraph
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[str] | None = None
    execution_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    generation_provider: Literal["groq", "ollama", "huggingface", "extractive", "openai", "gemini", "anthropic"] | None = None
    generation_model: str | None = Field(
        default=None, min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_./:-]+$"
    )
    allow_external_processing: bool = False


class QueryResponse(RAGModel):
    question: str
    answer: str
    abstained: bool
    citations: list[SourceCitation] = Field(default_factory=list)
    retrieved_chunks: list[SourceCitation] = Field(default_factory=list)
    embedding_model: str
    generation_model: str
    execution_id: str | None = None
    status: Literal["completed", "abstained", "authorization_required", "generation_failed"] = "completed"
    generation_provider: Literal["groq", "ollama", "huggingface", "extractive", "openai", "gemini", "anthropic"] | None = None
    provider_output: str | None = None
    generation_error: str | None = None
    context_preview: str | None = None
    context_truncated: bool = False
    detected_language: str | None = None
    response_language: str | None = None
    collection: str = "visual_rag_studio_documents"
    execution_started_at: datetime | None = None
    execution_completed_at: datetime | None = None


class StorageCollection(RAGModel):
    name: str
    count: int


class StorageSummary(RAGModel):
    metadata_backend: str
    metadata_location: str
    metadata_document_count: int
    source_files_present: int
    vector_backend: str
    vector_persistence: str
    vector_store_healthy: bool
    vector_count: int | None = None
    collections: list[StorageCollection] = Field(default_factory=list)
    checked_at: datetime
    last_successful_check: datetime | None = None
    error: str | None = None


class ChunkRecord(RAGModel):
    chunk_id: str
    document_id: str
    document_name: str
    ordinal: int
    strategy_id: str | None = None
    page_number: int | None = None
    section_path: list[str] = Field(default_factory=list)
    source_start: int | None = None
    source_end: int | None = None
    text: str
    embedding_available: bool
    embedding_dimension: int | None = None


class ChunkPage(RAGModel):
    total: int
    limit: int
    offset: int
    chunks: list[ChunkRecord]


class ProviderStatus(RAGModel):
    provider: str
    configured: bool
    online: bool
    embedding_model: str | None = None
    generation_model: str | None = None
    embedding_model_available: bool | None = None
    generation_model_available: bool | None = None
    error: str | None = None
    weights_load_on_first_use: bool = False


class GenerationProviderConfig(RAGModel):
    provider: Literal["groq", "ollama", "huggingface", "openai", "gemini", "anthropic", "extractive"]
    model: str | None = None
    configured: bool
    credential_present: bool
    huggingface_token_present: bool = False
    groq_api_key_present: bool = False
    openai_api_key_present: bool = False
    gemini_api_key_present: bool = False
    anthropic_api_key_present: bool = False
    execution_progress_supported: bool = False
    readiness: Literal["not_tested"] = "not_tested"
    provider_policy: str | None = None


class ProviderTestRequest(RAGModel):
    provider: Literal["groq", "ollama", "huggingface", "openai", "gemini", "anthropic", "extractive"]
    model: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_./:-]+$")


class ProviderTestResponse(RAGModel):
    provider: Literal["groq", "ollama", "huggingface", "openai", "gemini", "anthropic", "extractive"]
    model: str
    ready: bool
    message: str


class GroqCredentialStatus(RAGModel):
    configured: bool
    source: Literal["session", "environment", "none"]


class ExecutionProgress(RAGModel):
    execution_id: str
    status: Literal["running", "succeeded", "failed"]
    node_statuses: dict[str, Literal["queued", "running", "completed", "failed", "skipped"]]
    error: str | None = None


class DocumentChunkList(RAGModel):
    document_id: str
    chunks: list[SourceCitation]


class ErrorDetail(RAGModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class WorkflowSaveRequest(RAGModel):
    id: str | None = None
    name: str
    graph: WorkflowGraph


class WorkflowResponse(RAGModel):
    id: str
    name: str
    graph: dict[str, Any]
    created_at: str
    updated_at: str
    revision: int = 1


class WorkflowRevisionSummary(RAGModel):
    revision: int
    name: str
    created_at: str


class DeleteDocumentResponse(RAGModel):
    deleted: bool
    document_id: str
    filename: str | None = None
    chunks_removed: int = 0


class DeleteChunksResponse(RAGModel):
    cleared: bool = True
    document_id: str | None = None
    chunks_removed: int = 0
    message: str | None = None
    document: DocumentStatusResponse | None = None


class DeleteBulkResponse(RAGModel):
    deleted_count: int
    chunks_removed: int = 0
    message: str


class DeleteChunkResponse(RAGModel):
    deleted: bool
    chunk_id: str


class StorageCleanupResponse(RAGModel):
    cleaned_count: int
    cleaned_files: list[str] = Field(default_factory=list)


class AgentRunRequest(RAGModel):
    query: str = Field(min_length=1, max_length=10000)
    graph: WorkflowGraph
    execution_id: str | None = None
    goal: str | None = None
    instructions: str | None = None
    mode: Literal["fixed", "conditional", "bounded_loop"] | None = None
    max_steps: int | None = Field(default=None, ge=1, le=20)
    max_retrieval_iterations: int | None = Field(default=None, ge=1, le=10)
    token_budget: int | None = Field(default=None, ge=100, le=32000)
    timeout_seconds: float | None = Field(default=None, ge=5.0, le=300.0)
    approval_policy: Literal["always_ask", "on_uncertainty", "fully_autonomous"] | None = None
    allowed_tools: list[str] | None = None
    document_ids: list[str] | None = None
    initial_strategy: str = "hybrid"
    generation_provider: Literal["groq", "ollama", "huggingface", "extractive", "openai", "gemini", "anthropic"] | None = None
    generation_model: str | None = Field(
        default=None, min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_./:-]+$"
    )
    allow_external_processing: bool = False


class AgentRunResponse(RAGModel):
    execution_id: str
    goal: str
    status: str
    answer: str
    abstained: bool
    citations: list[SourceCitation] = Field(default_factory=list)
    step_count: int
    tokens_used: int
    retrieval_iterations: int
    trace: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    collection: str = "visual_rag_studio_documents"
    generation_provider: str | None = None
    generation_model: str | None = None
    embedding_model: str | None = None
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)


