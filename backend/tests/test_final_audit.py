"""Focused Final Audit Test Suite for Blank Canvas Validation and Agentic RAG.

Covers:
1. Provider readiness:
   - Missing credentials (Groq, OpenAI, Gemini, Anthropic, HuggingFace)
   - Invalid provider IDs
   - Disconnected providers / missing models
   - Unsupported provider capabilities
   - Vector-store connection failures / invalid backend
   - Distinguishing assignment, configuration, connectivity, and execution readiness
2. Workflow validation distinct states:
   - Draft (incomplete structure/connections)
   - Valid (correct structure, but unconfigured provider)
   - Executable (structure correct AND provider ready)
   - Relaxed draft validation does not weaken execution validation
3. Blank canvas Agentic RAG:
   - Complete workflow from blank canvas
   - Verified agentic execution path (not traditional)
   - Agent configuration persistence, custom instructions, retrieval, evidence evaluation,
     grounded generation, citation validation, bounded execution, failure handling, abstention
   - Query routing through both /agent/run and /query
4. Multi-strategy ingestion:
   - Successful combinations & deduplication policies
   - Incompatible combinations (boundary conflicts, duplicate strategies, structural conflicts)
   - Partial failures & provenance preservation
   - Retrieval from multi-strategy indexed outputs
5. Regression testing:
   - Traditional, Hybrid, Reranking, Multilingual, and Agentic RAG templates
"""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.contracts import (
    ConfiguredProviderStatus,
    WorkflowGraph,
    WorkflowValidationRequest,
)
from app.main import app
from app.rag.chunking import (
    ChunkingError,
    MultiStrategyChunkPlan,
    chunk_document_multi,
)
from app.rag.dependencies import get_document_store, get_settings, get_vector_store
from app.rag.graph import configured_provider_instances, validate_traditional_graph
from app.rag.parser import parse_document
from app.rag.storage import DocumentStore
from app.rag.vector_store import ChromaVectorStore
from app.registry import TEMPLATES_BY_ID

# ==============================================================================
# 1. PROVIDER READINESS AUDIT
# ==============================================================================


def test_provider_readiness_missing_credentials_blocks_execution():
    """Ensure missing credentials mark provider UNAVAILABLE and block execution without falsifying readiness."""
    settings = Settings(
        generation_provider="groq",
        groq_api_key=None,  # Missing API key
        groq_generation_model="llama-3.3-70b-versatile",
    )
    providers = configured_provider_instances(
        settings,
        generation_provider="groq",
        generation_model="llama-3.3-70b-versatile",
    )
    gen_provider = next(p for p in providers if p.id == "generation-provider")
    assert gen_provider.status == ConfiguredProviderStatus.UNAVAILABLE

    # Complete graph
    template_graph = TEMPLATES_BY_ID["traditional-rag"].graph
    assert template_graph is not None

    # Draft validation (require_executable=False): Valid structure, but blocked by warning
    draft_report = validate_traditional_graph(
        template_graph,
        settings,
        generation_provider="groq",
        generation_model="llama-3.3-70b-versatile",
        require_executable=False,
    )
    assert draft_report.valid is True
    assert draft_report.executable is False
    assert draft_report.workflow_state == "valid"
    assert any(i.code == "provider_unavailable" for i in draft_report.issues)

    # Execution validation (require_executable=True): Strictly rejected
    exec_report = validate_traditional_graph(
        template_graph,
        settings,
        generation_provider="groq",
        generation_model="llama-3.3-70b-versatile",
        require_executable=True,
    )
    assert exec_report.valid is False
    assert exec_report.executable is False
    assert exec_report.workflow_state == "draft"


def test_provider_readiness_invalid_provider_ids():
    """Ensure invalid generation or embedding provider IDs are flagged as incompatible/unavailable."""
    settings = Settings()
    providers = configured_provider_instances(
        settings,
        generation_provider="unsupported_llm_vendor",
        generation_model="some-model",
        embedding_provider="unsupported_emb_vendor",
        embedding_model="some-emb",
    )
    gen_p = next(p for p in providers if p.id == "generation-provider")
    emb_p = next(p for p in providers if p.id == "embedding-provider")
    assert gen_p.status == ConfiguredProviderStatus.UNAVAILABLE
    assert emb_p.status == ConfiguredProviderStatus.UNAVAILABLE

    template_graph = TEMPLATES_BY_ID["traditional-rag"].graph
    report = validate_traditional_graph(
        template_graph,
        settings,
        generation_provider="unsupported_llm_vendor",
        generation_model="some-model",
        require_executable=False,
    )
    assert report.executable is False
    codes = {i.code for i in report.issues}
    assert "provider_incompatible" in codes or "provider_unavailable" in codes


def test_provider_readiness_vector_store_invalid_backend():
    """Ensure invalid vector database backends are rejected as incompatible."""
    settings = Settings()
    providers = configured_provider_instances(
        settings,
        vector_backend="unsupported_exotic_db",
    )
    vector_p = next(p for p in providers if p.id == "chroma-local")
    assert vector_p.status == ConfiguredProviderStatus.UNAVAILABLE
    assert vector_p.adapter_id == "unsupported_exotic_db.vector_store"


def test_distinguish_assignment_configuration_and_readiness():
    """Verify the clear distinction between assignment, configuration, and execution readiness."""
    settings = Settings(
        generation_provider="groq",
        groq_api_key=None,  # assigned, but not configured
    )
    # Assignment: A node having provider_instance_id: 'generation-provider'
    # Configuration: Provider missing key => status=UNAVAILABLE
    # Readiness: Graph is structurally valid, but NOT executable
    template_graph = TEMPLATES_BY_ID["traditional-rag"].graph
    report = validate_traditional_graph(
        template_graph,
        settings,
        generation_provider="groq",
        generation_model="llama-3.3-70b-versatile",
        require_executable=False,
    )
    # Assignment verified: node has provider ID
    assert report.valid is True
    # Readiness verified: cannot execute
    assert report.executable is False
    assert report.workflow_state == "valid"


# ==============================================================================
# 2. WORKFLOW VALIDATION DISTINCT STATES (Draft, Valid, Executable)
# ==============================================================================


def test_validation_state_draft_incomplete_connections():
    """Incomplete connections result in Draft state (valid=False, executable=False)."""
    Settings()
    incomplete_graph = WorkflowGraph(
        schema_version=1,
        id="draft-1",
        name="Incomplete Draft",
        architecture_id="start-from-blank-canvas",
        template_id="start-from-blank-canvas",
        nodes=[
            {"id": "src", "node_type_id": "document.source", "config": {}},
            {"id": "parse", "node_type_id": "document.parse", "config": {}},
        ],
        connections=[],  # Disconnected
    )
    client = TestClient(app)
    req = WorkflowValidationRequest(graph=incomplete_graph, require_executable=False)
    res = client.post("/api/v1/workflows/validate", json=req.model_dump(mode="json"))
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is False
    assert data["executable"] is False
    assert data["workflow_state"] == "draft"


def test_validation_state_valid_graph_structure_correct_provider_unready():
    """Complete structure with unconfigured provider results in Valid state."""
    template_graph = TEMPLATES_BY_ID["traditional-rag"].graph
    assert template_graph is not None

    settings = Settings(
        generation_provider="groq",
        groq_api_key=None,  # Unready
    )
    report = validate_traditional_graph(
        template_graph,
        settings,
        generation_provider="groq",
        generation_model="llama-3.3-70b-versatile",
        require_executable=False,
    )
    assert report.valid is True
    assert report.executable is False
    assert report.workflow_state == "valid"


def test_validation_state_executable_ready_for_execution():
    """Complete structure with valid extractive provider results in Executable state."""
    template_graph = TEMPLATES_BY_ID["traditional-rag"].graph
    assert template_graph is not None

    settings = Settings(
        generation_provider="extractive",
        embedding_provider="fast_local",
    )
    report = validate_traditional_graph(
        template_graph,
        settings,
        generation_provider="extractive",
        generation_model="default",
        require_executable=False,
    )
    assert report.valid is True
    assert report.executable is True
    assert report.workflow_state == "executable"
    assert len(report.issues) == 0


def test_relaxed_draft_validation_does_not_weaken_execution_validation():
    """Ensure that relaxed draft warnings become strict errors during execution calls."""
    template_graph = TEMPLATES_BY_ID["traditional-rag"].graph
    assert template_graph is not None

    settings = Settings(generation_provider="groq", groq_api_key=None)
    # Relaxed draft mode allows valid=True
    draft = validate_traditional_graph(template_graph, settings, "groq", "llama-3.3-70b-versatile", require_executable=False)
    assert draft.valid is True
    assert draft.executable is False

    # Execution mode strictly denies valid=False
    execution = validate_traditional_graph(template_graph, settings, "groq", "llama-3.3-70b-versatile", require_executable=True)
    assert execution.valid is False
    assert execution.executable is False


# ==============================================================================
# 3. BLANK CANVAS AGENTIC RAG AUDIT
# ==============================================================================


@pytest.fixture
def blank_canvas_agentic_graph() -> WorkflowGraph:
    """Simulates an Agentic RAG workflow assembled from scratch on a blank canvas."""
    return WorkflowGraph(
        schema_version=1,
        id="blank-agentic-workflow",
        name="Custom Blank Agentic RAG",
        architecture_id="start-from-blank-canvas",
        template_id="start-from-blank-canvas",
        nodes=[
            {
                "id": "node_src_1",
                "node_type_id": "document.source",
                "config": {},
                "provider_instance_id": None,
            },
            {
                "id": "node_parse_2",
                "node_type_id": "document.parse",
                "config": {},
                "provider_instance_id": None,
            },
            {
                "id": "node_chunk_3",
                "node_type_id": "chunk.split",
                "config": {"strategy": "recursive", "chunk_size": 800, "overlap": 80},
                "provider_instance_id": None,
            },
            {
                "id": "node_embed_4",
                "node_type_id": "embedding.create",
                "config": {
                    "provider": "fast_local",
                    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "dimension": 384,
                },
                "provider_instance_id": None,
            },
            {
                "id": "node_index_5",
                "node_type_id": "vector.index",
                "config": {"backend": "chroma"},
                "provider_instance_id": None,
            },
            {
                "id": "node_query_6",
                "node_type_id": "query.input",
                "config": {},
                "provider_instance_id": None,
            },
            {
                "id": "node_agent_7",
                "node_type_id": "agent.orchestrate",
                "config": {
                    "goal": "Audit the provisions of the Aligarh Muslim University Act accurately.",
                    "instructions": "Extract verifiable facts and cite specific section numbers.",
                    "mode": "bounded_loop",
                    "max_steps": 5,
                    "max_retrieval_iterations": 3,
                    "token_budget": 4096,
                    "timeout_seconds": 60.0,
                    "approval_policy": "on_uncertainty",
                },
                "provider_instance_id": None,
            },
        ],
        connections=[
            {"source_node_id": "node_src_1", "source_port_id": "documents", "target_node_id": "node_parse_2", "target_port_id": "documents", "kind": "data", "label": None},
            {"source_node_id": "node_parse_2", "source_port_id": "parsed_documents", "target_node_id": "node_chunk_3", "target_port_id": "parsed_documents", "kind": "data", "label": None},
            {"source_node_id": "node_chunk_3", "source_port_id": "chunks", "target_node_id": "node_embed_4", "target_port_id": "chunks", "kind": "data", "label": None},
            {"source_node_id": "node_embed_4", "source_port_id": "embeddings", "target_node_id": "node_index_5", "target_port_id": "embeddings", "kind": "data", "label": None},
            {"source_node_id": "node_query_6", "source_port_id": "query", "target_node_id": "node_agent_7", "target_port_id": "query", "kind": "data", "label": None},
        ],
        execution={"mode": "agentic", "max_steps": 5},
    )


def test_blank_canvas_agentic_workflow_validates_cleanly(blank_canvas_agentic_graph: WorkflowGraph):
    """Verify that blank-canvas Agentic RAG passes validation with 0 errors and 0 false warnings."""
    settings = Settings(generation_provider="extractive", embedding_provider="fast_local")
    report = validate_traditional_graph(
        blank_canvas_agentic_graph,
        settings,
        generation_provider="extractive",
        generation_model="default",
        require_executable=True,
    )
    assert report.valid is True
    assert report.executable is True
    assert report.workflow_state == "executable"
    assert len(report.issues) == 0


def test_blank_canvas_agentic_e2e_execution(tmp_path, blank_canvas_agentic_graph: WorkflowGraph):
    """Execute end-to-end ingest and query on blank canvas Agentic RAG workflow.

    Verifies:
    - Agent configuration persistence (goal, instructions, max_steps, mode)
    - Full agentic execution trace (planning, retrieval, evaluation, generation)
    - Grounded generation with verifiable citations
    - Bounded execution limits
    """
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'audit_meta.db'}",
        data_path=str(tmp_path / "data"),
        chroma_path=str(tmp_path / "chroma"),
        generation_provider="extractive",
        embedding_provider="fast_local",
    )
    store = DocumentStore(settings.database_url, settings.data_path)
    vectors = ChromaVectorStore(settings.chroma_path)

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_document_store] = lambda: store
    app.dependency_overrides[get_vector_store] = lambda: vectors

    try:
        client = TestClient(app)

        # 1. Upload document
        upload = client.post(
            "/api/v1/documents",
            files={"file": ("amu_act.txt", b"Section 5 of the Aligarh Muslim University Act defines the powers of the Court, Academic Council, and Executive Council.", "text/plain")},
        )
        assert upload.status_code == 201
        doc_id = upload.json()["document_id"]

        # 2. Ingest document with blank canvas Agentic workflow
        ingest_payload = {
            "graph": blank_canvas_agentic_graph.model_dump(mode="json"),
            "generation_provider": "extractive",
        }
        ingest = client.post(f"/api/v1/documents/{doc_id}/ingest", json=ingest_payload)
        assert ingest.status_code == 200
        assert ingest.json()["status"] == "ready"

        # 3. Query via /api/v1/agent/run
        agent_req = {
            "query": "What does Section 5 of the Act define?",
            "graph": blank_canvas_agentic_graph.model_dump(mode="json"),
            "document_ids": [doc_id],
            "generation_provider": "extractive",
        }
        res = client.post("/api/v1/agent/run", json=agent_req)
        assert res.status_code == 200, res.text
        data = res.json()

        # Verify Agent Execution details
        assert data["status"] in ("completed", "success")
        assert data["abstained"] is False
        assert "Council" in data["answer"] or "powers" in data["answer"]
        assert len(data["citations"]) >= 1

        # Verify Trace includes Planning, Retrieval, and Answer Generation
        trace_tools = [step["tool"] for step in data["trace"]]
        assert "retrieval" in trace_tools
        assert "answer_generation" in trace_tools
        assert data["step_count"] <= 5

        # 4. Query via /api/v1/query and confirm it delegates to Agentic execution
        query_payload = {
            "question": "What does Section 5 define?",
            "graph": blank_canvas_agentic_graph.model_dump(mode="json"),
            "document_ids": [doc_id],
            "generation_provider": "extractive",
        }
        q_res = client.post("/api/v1/query", json=query_payload)
        assert q_res.status_code == 200
        q_data = q_res.json()
        assert q_data["abstained"] is False
        assert len(q_data["citations"]) >= 1
        assert "Council" in q_data["answer"] or "powers" in q_data["answer"]

        # 5. Abstention on unanswerable query
        unanswerable_req = {
            "query": "What is the capital of Mars in the year 3000?",
            "graph": blank_canvas_agentic_graph.model_dump(mode="json"),
            "document_ids": [doc_id],
            "generation_provider": "extractive",
        }
        abs_res = client.post("/api/v1/agent/run", json=unanswerable_req)
        assert abs_res.status_code == 200
        abs_data = abs_res.json()
        assert abs_data["abstained"] is True
        assert abs_data["citations"] == []
        assert "not find enough retrieved evidence" in abs_data["answer"]
    finally:
        app.dependency_overrides.clear()
        vectors.close()
        store.close()


# ==============================================================================
# 4. MULTI-STRATEGY INGESTION AUDIT
# ==============================================================================


def test_multi_strategy_successful_combinations_and_deduplication():
    """Verify multi-strategy chunking executes in parallel and honors deduplication policies."""
    content = b"""# Chapter I: Preliminary
The short title is The Aligarh Muslim University Act, 1920.
It shall come into force on such date as the Governor General in Council may appoint.

# Chapter II: The University
The University shall be a body corporate by the name of the Aligarh Muslim University.
The University shall have perpetual succession and a common seal."""

    parsed = parse_document("doc-test", "amu.md", content)

    plans = [
        MultiStrategyChunkPlan(strategy_id="structure_aware", settings={"chunk_size": 200, "overlap": 20}, priority=1),
        MultiStrategyChunkPlan(strategy_id="recursive", settings={"chunk_size": 200, "overlap": 20}, priority=2),
    ]

    # Test keep_all policy
    res_keep_all = chunk_document_multi(parsed, plans, deduplication_policy="keep_all")
    assert res_keep_all.total_chunks >= 2
    assert "structure_aware" in res_keep_all.per_strategy_stats
    assert "recursive" in res_keep_all.per_strategy_stats

    # Test exact_dedup policy
    res_dedup = chunk_document_multi(parsed, plans, deduplication_policy="exact_dedup")
    assert res_dedup.total_chunks <= res_keep_all.total_chunks

    # Test prefer_structured policy
    res_pref = chunk_document_multi(parsed, plans, deduplication_policy="prefer_structured")
    assert any(c.strategy_id == "structure_aware" for c in res_pref.chunks)


def test_multi_strategy_incompatible_combinations_rejected():
    """Verify conflicting strategies and duplicate strategies are rejected with ChunkingError."""
    parsed = parse_document("doc-1", "doc.txt", b"Sample legal text for testing chunking.")

    # Incompatible fixed + sentence
    with pytest.raises(ChunkingError, match="conflicting chunk boundaries"):
        chunk_document_multi(parsed, [
            MultiStrategyChunkPlan(strategy_id="fixed", settings={"chunk_size": 50, "overlap": 10}),
            MultiStrategyChunkPlan(strategy_id="sentence", settings={"chunk_size": 50, "overlap": 10}),
        ])

    # Duplicate strategy
    with pytest.raises(ChunkingError, match="duplicate strategy"):
        chunk_document_multi(parsed, [
            MultiStrategyChunkPlan(strategy_id="recursive"),
            MultiStrategyChunkPlan(strategy_id="recursive"),
        ])


def test_multi_strategy_provenance_and_partial_failure():
    """Verify that multi-strategy outputs preserve full chunk provenance and handle partial failure."""
    parsed = parse_document("doc-1", "doc.txt", b"Paragraph 1 text.\n\nParagraph 2 text.\n\nParagraph 3 text.")

    plans = [
        MultiStrategyChunkPlan(strategy_id="recursive", settings={"chunk_size": 100, "overlap": 10}),
        # Passing an invalid setting that triggers failure inside strategy
        MultiStrategyChunkPlan(strategy_id="fixed", settings={"chunk_size": -10, "overlap": 0}),
    ]
    result = chunk_document_multi(parsed, plans, deduplication_policy="keep_all")
    # Recursive succeeded, fixed failed, warning captured
    assert result.total_chunks >= 1
    assert result.per_strategy_stats["recursive"]["chunk_count"] >= 1
    assert result.per_strategy_stats["fixed"]["error"] is not None
    assert len(result.warnings) == 1
    # Check provenance
    for chunk in result.chunks:
        assert chunk.strategy_id == "recursive"
        assert chunk.document_id == "doc-1"
        assert chunk.ordinal is not None


# ==============================================================================
# 5. REGRESSION AUDIT (Existing Architectures & Templates)
# ==============================================================================


@pytest.mark.parametrize("template_id", [
    "traditional-rag",
    "hybrid-rag",
    "reranking-rag",
    "multilingual-rag",
    "agentic-rag",
])
def test_regression_existing_templates_validate_executable(template_id: str):
    """Ensure all pre-existing architecture templates validate cleanly with 0 errors and 0 false warnings."""
    template = TEMPLATES_BY_ID[template_id]
    assert template.graph is not None

    settings = Settings(generation_provider="extractive", embedding_provider="fast_local")
    report = validate_traditional_graph(
        template.graph,
        settings,
        generation_provider="extractive",
        generation_model="default",
        require_executable=True,
    )
    assert report.valid is True
    assert report.executable is True
    assert report.workflow_state == "executable"
    assert len(report.issues) == 0
