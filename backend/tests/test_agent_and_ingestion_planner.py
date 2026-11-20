import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.contracts import ExecutionMode, ImplementationStatus
from app.main import app
from app.rag.agent import (
    AgentApprovalPolicy,
    AgentExecutionMode,
    AgentOrchestrator,
)
from app.rag.chunking import (
    ChunkingError,
    MultiStrategyChunkPlan,
    chunk_document,
    chunk_document_multi,
)
from app.rag.parser import parse_document
from app.rag.pipeline import PipelineError, assemble_context, index_document
from app.rag.planner import IngestionPlanner, PlanningMode
from app.rag.providers import ExtractiveGeneratorProvider, FastLocalEmbeddingProvider, ProviderError
from app.rag.storage import DocumentStore
from app.rag.vector_store import MemoryVectorStore
from app.registry import REGISTRY, TEMPLATES_BY_ID
from app.rag.graph import validate_traditional_graph


@pytest.fixture
def test_client():
    return TestClient(app)


def test_ingestion_planner_modes():
    raw_doc = (
        "# Strategic Corporate Overview\n\n"
        "## Executive Summary\n"
        "Visual RAG Studio delivers enterprise-grade visual retrieval pipelines.\n\n"
        "| Strategy | Precision | Latency |\n"
        "|---|---|---|\n"
        "| Recursive | 0.88 | 12ms |\n"
        "| Semantic | 0.94 | 45ms |\n\n"
        "```python\ndef run_pipeline():\n    return True\n```\n"
    )
    doc = parse_document("doc-test-1", "strategy_doc.md", raw_doc.encode("utf-8"))

    # 1. Recommendation Only mode
    plan_rec = IngestionPlanner.plan(doc, mode=PlanningMode.RECOMMENDATION_ONLY)
    assert plan_rec.mode == PlanningMode.RECOMMENDATION_ONLY
    assert plan_rec.requires_approval is True
    assert plan_rec.approved is False
    assert len(plan_rec.selected_strategies) >= 1
    strat_ids = [s.strategy_id for s in plan_rec.selected_strategies]
    assert "ragspark.table_aware" in strat_ids or "recursive" in strat_ids

    # 2. User Approved Auto mode
    plan_user_auto = IngestionPlanner.plan(doc, mode=PlanningMode.USER_APPROVED_AUTO)
    assert plan_user_auto.mode == PlanningMode.USER_APPROVED_AUTO
    assert plan_user_auto.requires_approval is True
    assert plan_user_auto.approved is False

    # 3. Fully Autonomous mode
    plan_autonomous = IngestionPlanner.plan(doc, mode=PlanningMode.FULLY_AUTONOMOUS)
    assert plan_autonomous.mode == PlanningMode.FULLY_AUTONOMOUS
    assert plan_autonomous.requires_approval is False
    assert plan_autonomous.approved is True

    # 4. Manual mode
    plan_manual = IngestionPlanner.plan(doc, mode=PlanningMode.MANUAL)
    assert plan_manual.mode == PlanningMode.MANUAL
    assert plan_manual.requires_approval is True


def test_ingestion_planner_multilingual_routing():
    hindi_doc = (
        "Visual RAG Studio ek shaktishali AI enterprise search platform hai. "
        "Yeh Hindi aur English dono bhashaon ko asani se process karta hai."
    )
    doc = parse_document("doc-hi-1", "hindi_intro.txt", hindi_doc.encode("utf-8"))
    plan = IngestionPlanner.plan(doc, mode=PlanningMode.USER_APPROVED_AUTO)
    assert "hi" in plan.embedding_proposal.languages or "multilingual" in plan.embedding_proposal.languages
    assert "multilingual" in plan.embedding_proposal.model.lower()


def test_multi_strategy_chunking_and_deduplication():
    text = (
        "# Title 1\n\n"
        "Visual RAG Studio transforms complex documents into high dimensional semantic representations.\n\n"
        "Section Alpha outlines the foundation of modular pipelines.\n\n"
        "# Title 2\n\n"
        "Section Beta describes hybrid fusion scoring using BM25 and dense bi-encoders."
    )
    doc = parse_document("multi-chunk-doc", "doc.md", text.encode("utf-8"))

    plans = [
        MultiStrategyChunkPlan(strategy_id="recursive", settings={"chunk_size": 150, "overlap": 20}),
        MultiStrategyChunkPlan(strategy_id="hierarchical", settings={"chunk_size": 200, "overlap": 30}),
    ]

    result = chunk_document_multi(doc, plans, deduplication_policy="exact_dedup")
    assert len(result.chunks) > 0
    assert result.total_chunks == len(result.chunks)
    assert "recursive" in result.per_strategy_stats
    assert "hierarchical" in result.per_strategy_stats
    assert len(result.warnings) == 0

    # Ensure stable unique ordinals
    ordinals = [c.ordinal for c in result.chunks]
    assert ordinals == list(range(len(result.chunks)))

    # Ensure metadata preserved
    for chunk in result.chunks:
        assert chunk.document_id == "multi-chunk-doc"
        assert chunk.strategy_id in {"recursive", "hierarchical"}


def test_dimension_mismatch_detection_in_planner():
    raw = "Quick reference notes."
    doc = parse_document("doc-dim-test", "notes.txt", raw.encode("utf-8"))
    chunks = chunk_document(doc, "recursive", {"chunk_size": 100, "overlap": 10})

    # Vector store with 1024 dimension collection
    v_store = MemoryVectorStore()
    v_store.upsert_document(
        "doc-dim-test",
        chunks,
        [[0.1] * 1024 for _ in chunks],
        "model-1024",
        1024,
    )

    plan = IngestionPlanner.plan(doc, existing_vector_store=v_store)
    # Target model is 384 dim, but collection has 1024 dim
    assert plan.vector_store_proposal.compatible is False
    assert any("dimension mismatch" in w.lower() for w in plan.warnings)


def test_agent_orchestrator_bounded_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "docs.db"
        store = DocumentStore(f"sqlite:///{db_path.as_posix()}", Path(tmpdir) / "sources")
        try:
            settings = Settings()
            v_store = MemoryVectorStore()
            emb_provider = FastLocalEmbeddingProvider()
            gen_provider = ExtractiveGeneratorProvider()

            # Seed indexed document
            text = (
                "Visual RAG Studio orchestrator provides autonomous multi-step reasoning. "
                "The agent loop is strictly bounded by max_steps and token limits to ensure safety."
            )
            doc_id = store.create("ai_architecture.txt", "txt", "text/plain", text.encode("utf-8"))
            parsed = parse_document(doc_id, "ai_architecture.txt", text.encode("utf-8"))
            store.update(doc_id, profile=parsed.profile.model_dump(mode="json"), status="ready")

            chunks = chunk_document(parsed, "recursive", {"chunk_size": 200, "overlap": 20})
            vecs, dim = emb_provider.embed("fast_local", [c.text for c in chunks])
            v_store.upsert_document(
                doc_id,
                chunks,
                vecs,
                "fast_local",
                dim,
            )
            store.update(doc_id, chunks=[{"chunk_id": c.chunk_id, "text": c.text, "ordinal": c.ordinal} for c in chunks])

            # Test Agent with max_steps=4
            orchestrator = AgentOrchestrator(
                goal="Explain the safety bounds of the agent loop.",
                mode=AgentExecutionMode.BOUNDED_LOOP,
                max_steps=4,
                max_retrieval_iterations=2,
                approval_policy=AgentApprovalPolicy.FULLY_AUTONOMOUS,
            )

            res = orchestrator.run(
                query="How is the agent loop bounded?",
                vector_store=v_store,
                provider=emb_provider,
                settings=settings,
                store=store,
                generation_provider=gen_provider,
            )

            assert res["status"] == "completed"
            assert res["step_count"] <= 4
            assert len(res["trace"]) > 0
            assert "strictly bounded by max_steps" in res["answer"]
            assert len(res["citations"]) >= 1
            assert res["tokens_used"] > 0
        finally:
            store.close()


def test_agent_orchestrator_abstention_on_unanswerable():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "docs.db"
        store = DocumentStore(f"sqlite:///{db_path.as_posix()}", Path(tmpdir) / "sources")
        try:
            settings = Settings()
            v_store = MemoryVectorStore()
            emb_provider = FastLocalEmbeddingProvider()
            gen_provider = ExtractiveGeneratorProvider()

            text = "Jupiter is the largest planet in our solar system, with a Great Red Spot."
            doc_id = store.create("solar_system.txt", "txt", "text/plain", text.encode("utf-8"))
            parsed = parse_document(doc_id, "solar_system.txt", text.encode("utf-8"))
            store.update(doc_id, profile=parsed.profile.model_dump(mode="json"), status="ready")

            chunks = chunk_document(parsed, "recursive", {"chunk_size": 200, "overlap": 20})
            vecs, dim = emb_provider.embed("fast_local", [c.text for c in chunks])
            v_store.upsert_document(
                doc_id,
                chunks,
                vecs,
                "fast_local",
                dim,
            )
            store.update(doc_id, chunks=[{"chunk_id": c.chunk_id, "text": c.text, "ordinal": c.ordinal} for c in chunks])

            orchestrator = AgentOrchestrator(
                goal="Find information about quantum supremacy.",
                mode=AgentExecutionMode.BOUNDED_LOOP,
                max_steps=3,
                max_retrieval_iterations=2,
            )

            res = orchestrator.run(
                query="What quantum algorithm achieved quantum supremacy in 2019?",
                vector_store=v_store,
                provider=emb_provider,
                settings=settings,
                store=store,
                generation_provider=gen_provider,
            )

            assert "could not find enough retrieved evidence" in res["answer"]
            assert len(res["citations"]) == 0
            assert res["abstained"] is True
            assert any(t["tool"] == "abstain" for t in res["trace"])
        finally:
            store.close()


def test_api_plan_and_approve_ingestion(test_client):
    # Upload and parse document
    content = b"Visual RAG Studio autonomous planner test document content."
    res = test_client.post(
        "/api/v1/documents",
        files={"file": ("planner_test.txt", content, "text/plain")},
    )
    assert res.status_code == 201
    doc_id = res.json()["document_id"]

    # Plan ingestion
    plan_res = test_client.post(f"/api/v1/documents/{doc_id}/plan-ingestion")
    assert plan_res.status_code == 200
    plan_data = plan_res.json()
    assert plan_data["document_id"] == doc_id
    assert len(plan_data["selected_strategies"]) >= 1

    # Approve plan
    approve_res = test_client.post(f"/api/v1/documents/{doc_id}/approve-plan")
    assert approve_res.status_code == 200
    doc_status = approve_res.json()
    assert doc_status["plan"]["approved"] is True
    assert doc_status["selection"] is not None


def test_api_agent_run_endpoint(test_client):
    template = TEMPLATES_BY_ID["agentic-rag"]
    assert template.graph is not None

    req_payload = {
        "query": "What is the agent goal?",
        "graph": template.graph.model_dump(mode="json"),
        "goal": "Verify system goals.",
        "mode": "bounded_loop",
        "max_steps": 2,
        "generation_provider": "extractive",
    }

    res = test_client.post("/api/v1/agent/run", json=req_payload)
    assert res.status_code == 200
    data = res.json()
    assert "execution_id" in data
    assert "answer" in data
    assert "trace" in data
    assert data["step_count"] <= 2


def _create_test_env(doc_text: str = "Visual RAG Studio provides bounded agentic execution."):
    tmpdir = tempfile.TemporaryDirectory()
    db_path = Path(tmpdir.name) / "docs.db"
    store = DocumentStore(f"sqlite:///{db_path.as_posix()}", Path(tmpdir.name) / "sources")
    settings = Settings()
    v_store = MemoryVectorStore()
    emb_provider = FastLocalEmbeddingProvider()
    gen_provider = ExtractiveGeneratorProvider()

    doc_id = store.create("test_doc.txt", "txt", "text/plain", doc_text.encode("utf-8"))
    parsed = parse_document(doc_id, "test_doc.txt", doc_text.encode("utf-8"))
    store.update(doc_id, profile=parsed.profile.model_dump(mode="json"), status="ready")

    chunks = chunk_document(parsed, "recursive", {"chunk_size": 200, "overlap": 20})
    vecs, dim = emb_provider.embed("fast_local", [c.text for c in chunks])
    v_store.upsert_document(doc_id, chunks, vecs, "fast_local", dim)
    store.update(doc_id, chunks=[{"chunk_id": c.chunk_id, "text": c.text, "ordinal": c.ordinal} for c in chunks])

    return tmpdir, store, v_store, emb_provider, gen_provider, settings, doc_id


def test_scenario_1_multiline_goal_and_instructions_persistence(test_client):
    template = TEMPLATES_BY_ID["agentic-rag"]
    graph = template.graph.model_copy(deep=True)
    agent_node = next(n for n in graph.nodes if n.node_type_id == "agent.orchestrate")
    multiline_goal = "Line 1: Primary goal.\nLine 2: Target requirements.\nLine 3: Safety constraints."
    multiline_instructions = "1. Be precise.\n2. Do not hallucinate.\n3. Format citations as [1], [2].\n4. Maintain neutral tone."
    agent_node.config["goal"] = multiline_goal
    agent_node.config["instructions"] = multiline_instructions

    create_res = test_client.post("/api/v1/workflows", json={
        "name": "Multiline Test Workflow",
        "graph": graph.model_dump(mode="json"),
    })
    assert create_res.status_code == 200
    wf_id = create_res.json()["id"]

    get_res = test_client.get(f"/api/v1/workflows/{wf_id}")
    assert get_res.status_code == 200
    loaded_graph = get_res.json()["graph"]
    loaded_agent = next(n for n in loaded_graph["nodes"] if n["node_type_id"] == "agent.orchestrate")
    assert loaded_agent["config"]["goal"] == multiline_goal
    assert loaded_agent["config"]["instructions"] == multiline_instructions
    assert "\n" in loaded_agent["config"]["goal"]
    assert "\n" in loaded_agent["config"]["instructions"]


def test_scenario_2_custom_instructions_reach_agent_runtime():
    tmpdir, store, v_store, emb_provider, gen_provider, settings, doc_id = _create_test_env(
        "Tax rule: Provisional collection of taxes takes effect immediately upon declaration."
    )
    try:
        custom_goal = "Extract provisional tax rules precisely."
        custom_inst = "Prefix all answers with [OFFICIAL_NOTICE]."

        retrieved = v_store.search([0.1] * 384, top_k=2)
        prompt, supplied = assemble_context(
            "What is provisional tax?",
            retrieved,
            max_chars=4000,
            agent_goal=custom_goal,
            agent_instructions=custom_inst,
        )
        assert custom_goal in prompt
        assert custom_inst in prompt

        orchestrator = AgentOrchestrator(
            goal=custom_goal,
            instructions=custom_inst,
            mode=AgentExecutionMode.BOUNDED_LOOP,
            max_steps=4,
        )
        res = orchestrator.run(
            query="What is provisional tax?",
            vector_store=v_store,
            provider=emb_provider,
            settings=settings,
            store=store,
            generation_provider=gen_provider,
        )
        assert res["goal"] == custom_goal
        assert res["status"] == "completed"
    finally:
        store.close()
        tmpdir.cleanup()


def test_scenario_3_simple_query_no_unnecessary_reranking():
    tmpdir, store, v_store, emb_provider, gen_provider, settings, doc_id = _create_test_env(
        "Tax rate is set at fifteen percent per annum."
    )
    try:
        orchestrator = AgentOrchestrator(
            goal="Answer tax questions.",
            mode=AgentExecutionMode.BOUNDED_LOOP,
            max_steps=5,
        )
        res = orchestrator.run(
            query="What is tax?",
            vector_store=v_store,
            provider=emb_provider,
            settings=settings,
            store=store,
            generation_provider=gen_provider,
        )
        assert res["status"] == "completed"
        assert not any(t["tool"] == "rerank" for t in res["trace"])
        planning_step = next((t for t in res["trace"] if t["tool"] == "planning"), None)
        if planning_step:
            assert planning_step["action_input"]["complexity"] == "simple"
    finally:
        store.close()
        tmpdir.cleanup()


def test_scenario_4_complex_query_triggers_reranking():
    multi_paragraph_text = (
        "Income tax is levied on annual personal earnings across progressive tax brackets.\n\n"
        "Corporate tax applies to company net profits after allowable business expense deductions.\n\n"
        "Capital gains tax targets profits made from selling assets, investments, and shares.\n\n"
        "Compare and contrast both personal income tax and corporate tax structures and filing requirements."
    )
    tmpdir, store, v_store, emb_provider, gen_provider, settings, doc_id = _create_test_env(multi_paragraph_text)
    try:
        orchestrator = AgentOrchestrator(
            goal="Answer comparative queries.",
            mode=AgentExecutionMode.BOUNDED_LOOP,
            max_steps=6,
        )
        complex_q = "Compare and contrast income tax versus corporate tax and explain both"
        res = orchestrator.run(
            query=complex_q,
            vector_store=v_store,
            provider=emb_provider,
            settings=settings,
            store=store,
            generation_provider=gen_provider,
        )
        planning_decision = next((d for d in res["decisions"] if d["decision_type"] == "planning"), None)
        assert planning_decision is not None
        assert "complex" in planning_decision["action_taken"].lower() or "complex" in planning_decision["rationale"].lower()
        assert any(t["tool"] == "rerank" for t in res["trace"])
    finally:
        store.close()
        tmpdir.cleanup()


def test_scenario_5_insufficient_evidence_triggers_reformulation_or_abstention():
    tmpdir, store, v_store, emb_provider, gen_provider, settings, doc_id = _create_test_env(
        "Ancient Egyptian pyramids were constructed along the Nile river."
    )
    try:
        orchestrator = AgentOrchestrator(
            goal="Answer space science queries.",
            mode=AgentExecutionMode.BOUNDED_LOOP,
            max_steps=5,
            max_retrieval_iterations=2,
            allowed_tools=["planning", "retrieval", "evaluate_evidence", "reformulate_query", "abstain"],
        )
        res = orchestrator.run(
            query="What is the chemical composition of Saturn's outer rings?",
            vector_store=v_store,
            provider=emb_provider,
            settings=settings,
            store=store,
            generation_provider=gen_provider,
        )
        assert res["abstained"] is True
        assert res["status"] == "abstained"
        assert "could not find enough retrieved evidence" in res["answer"].lower()
        tools_used = {t["tool"] for t in res["trace"]}
        assert "abstain" in tools_used
    finally:
        store.close()
        tmpdir.cleanup()


def test_scenario_6_valid_retrieval_not_discarded_on_step_limit():
    tmpdir, store, v_store, emb_provider, gen_provider, settings, doc_id = _create_test_env(
        "Section 3 of the Act authorizes provisional collection of customs duties immediately upon declaration."
    )
    try:
        orchestrator = AgentOrchestrator(
            goal="Find authorization section.",
            mode=AgentExecutionMode.BOUNDED_LOOP,
            max_steps=2,
            allowed_tools=["retrieval", "answer_generation"],
        )
        res = orchestrator.run(
            query="Which section authorizes provisional collection?",
            vector_store=v_store,
            provider=emb_provider,
            settings=settings,
            store=store,
            generation_provider=gen_provider,
        )
        assert res["status"] == "completed"
        assert res["step_count"] <= 2
        assert "provisional collection" in res["answer"].lower() or "section" in res["answer"].lower()
        assert len(res["citations"]) > 0
    finally:
        store.close()
        tmpdir.cleanup()


def test_scenario_7_multi_strategy_chunking_real_implementations():
    raw_text = (
        "# Chapter 1: Introduction\n\n"
        "Visual RAG Studio provides automated strategy selection.\n\n"
        "## Subchapter 1.1: Core Methods\n"
        "Deterministic pipeline execution ensures reproducible evaluations across models.\n"
    )
    doc = parse_document("multi-doc-7", "doc.md", raw_text.encode("utf-8"))
    plans = [
        MultiStrategyChunkPlan(strategy_id="recursive", settings={"chunk_size": 120, "overlap": 15}),
        MultiStrategyChunkPlan(strategy_id="hierarchical", settings={"chunk_size": 250, "overlap": 30}),
    ]
    res = chunk_document_multi(doc, plans, deduplication_policy="keep_all")
    assert res.total_chunks > 0
    assert "recursive" in res.per_strategy_stats
    assert "hierarchical" in res.per_strategy_stats
    assert res.per_strategy_stats["recursive"]["chunk_count"] > 0
    assert res.per_strategy_stats["hierarchical"]["chunk_count"] > 0
    assert res.per_strategy_stats["recursive"]["duration_ms"] >= 0
    for c in res.chunks:
        assert c.strategy_id in ("recursive", "hierarchical")
        assert c.document_id == "multi-doc-7"


def test_scenario_8_incompatible_strategy_combinations_fail_clearly():
    raw_text = "Some document text for chunking."
    doc = parse_document("doc-fail-8", "doc.txt", raw_text.encode("utf-8"))

    plans_incompatible = [
        MultiStrategyChunkPlan(strategy_id="fixed", settings={"chunk_size": 100, "overlap": 10}),
        MultiStrategyChunkPlan(strategy_id="sentence", settings={"chunk_size": 100, "overlap": 10}),
    ]
    with pytest.raises(ChunkingError) as exc_info:
        chunk_document_multi(doc, plans_incompatible)
    assert "incompatible" in str(exc_info.value).lower()

    plans_duplicate = [
        MultiStrategyChunkPlan(strategy_id="recursive", settings={"chunk_size": 100, "overlap": 10}),
        MultiStrategyChunkPlan(strategy_id="recursive", settings={"chunk_size": 200, "overlap": 20}),
    ]
    with pytest.raises(ChunkingError) as exc_dup:
        chunk_document_multi(doc, plans_duplicate)
    assert "duplicate" in str(exc_dup.value).lower()


def test_scenario_9_unsupported_chunking_strategies_rejected():
    raw_text = "Standard document content."
    doc = parse_document("doc-fail-9", "doc.txt", raw_text.encode("utf-8"))
    plans = [{"strategy_id": "quantum_neural_chunker", "settings": {}}]
    with pytest.raises(ChunkingError) as exc_info:
        chunk_document_multi(doc, plans)
    assert "unsupported chunking strategy" in str(exc_info.value).lower()
    assert "quantum_neural_chunker" in str(exc_info.value)


def test_scenario_10_embedding_dimension_mismatch_rejected_409():
    tmpdir = tempfile.TemporaryDirectory()
    try:
        db_path = Path(tmpdir.name) / "docs.db"
        store = DocumentStore(f"sqlite:///{db_path.as_posix()}", Path(tmpdir.name) / "sources")
        doc_id = store.create("dim_test.txt", "txt", "text/plain", b"Content for dimension test.")
        parsed = parse_document(doc_id, "dim_test.txt", b"Content for dimension test.")
        store.update(
            doc_id,
            profile=parsed.profile.model_dump(mode="json"),
            status="profiled",
            selection={"strategy_id": "recursive", "settings": {"chunk_size": 200, "overlap": 20}},
        )

        v_store = MemoryVectorStore(collection_name="test_1024_col")
        dummy_chunk = chunk_document(parsed, "recursive", {"chunk_size": 200, "overlap": 20})[0]
        v_store.upsert_document("existing_doc", [dummy_chunk], [[0.0] * 1024], "model-1024", 1024)

        settings = Settings()
        emb_provider = FastLocalEmbeddingProvider()

        with pytest.raises(PipelineError) as exc_info:
            index_document(
                doc_id,
                store=store,
                vector_store=v_store,
                provider=emb_provider,
                settings=settings,
                embedding_model="fast_local",
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.code == "dimension_mismatch"
    finally:
        store.close()
        tmpdir.cleanup()


def test_scenario_11_indexing_and_querying_use_intended_collection():
    tmpdir, store, v_store, emb_provider, gen_provider, settings, doc_id = _create_test_env(
        "Custom collection content: Special maritime laws apply in territorial waters."
    )
    try:
        custom_col_name = "special_maritime_collection"
        v_store.collection_name = custom_col_name

        orchestrator = AgentOrchestrator(
            goal="Answer maritime law questions.",
            mode=AgentExecutionMode.BOUNDED_LOOP,
            max_steps=4,
        )
        res = orchestrator.run(
            query="Where do maritime laws apply?",
            vector_store=v_store,
            provider=emb_provider,
            settings=settings,
            store=store,
            generation_provider=gen_provider,
        )
        assert res["collection"] == custom_col_name
        assert res["status"] == "completed"
    finally:
        store.close()
        tmpdir.cleanup()


def test_scenario_12_provider_failures_reported_as_provider_error():
    tmpdir, store, v_store, emb_provider, gen_provider, settings, doc_id = _create_test_env(
        "Tax assessment rules are documented in chapter 4."
    )
    try:
        class FailingProvider:
            provider_name = "groq"
            def generate(self, *args, **kwargs):
                raise ProviderError("Groq 429: Rate limit quota exceeded.")

        orchestrator = AgentOrchestrator(
            goal="Answer questions.",
            mode=AgentExecutionMode.BOUNDED_LOOP,
            max_steps=5,
        )
        res = orchestrator.run(
            query="What are the tax assessment rules?",
            vector_store=v_store,
            provider=emb_provider,
            settings=settings,
            store=store,
            generation_provider=FailingProvider(),
        )
        assert res["status"] == "provider_error"
        assert res["abstained"] is False
        assert "rate limit" in str(res["error"]).lower() or "429" in str(res["error"])
        assert not any(t["tool"] == "abstain" for t in res["trace"])
        gen_trace = next(t for t in res["trace"] if t["tool"] == "answer_generation")
        assert gen_trace["status"] == "failed"
    finally:
        store.close()
        tmpdir.cleanup()


def test_scenario_13_citations_reference_real_chunks_with_metadata():
    tmpdir, store, v_store, emb_provider, gen_provider, settings, doc_id = _create_test_env(
        "Article 5 declares that tariffs shall be published in the Official Gazette on Mondays."
    )
    try:
        orchestrator = AgentOrchestrator(
            goal="Extract publication schedule.",
            mode=AgentExecutionMode.BOUNDED_LOOP,
            max_steps=4,
        )
        res = orchestrator.run(
            query="When are tariffs published in the Official Gazette?",
            vector_store=v_store,
            provider=emb_provider,
            settings=settings,
            store=store,
            generation_provider=gen_provider,
        )
        assert res["status"] == "completed"
        assert len(res["citations"]) > 0
        for citation in res["citations"]:
            assert "chunk_id" in citation
            assert citation["document_id"] == doc_id
            assert "Official Gazette" in citation["text"] or "tariffs" in citation["text"]
    finally:
        store.close()
        tmpdir.cleanup()


def test_scenario_14_execution_traces_reflect_actual_tool_calls():
    tmpdir, store, v_store, emb_provider, gen_provider, settings, doc_id = _create_test_env(
        "Section 1 defines short title and extent."
    )
    try:
        orchestrator = AgentOrchestrator(
            goal="Explain section 1.",
            mode=AgentExecutionMode.BOUNDED_LOOP,
            max_steps=5,
        )
        res = orchestrator.run(
            query="What does section 1 define?",
            vector_store=v_store,
            provider=emb_provider,
            settings=settings,
            store=store,
            generation_provider=gen_provider,
        )
        assert len(res["trace"]) >= 2
        for step in res["trace"]:
            assert step["tool"] in AgentOrchestrator.SUPPORTED_TOOLS
            assert step["duration_ms"] >= 0
            assert isinstance(step["action_input"], dict)
            assert len(step["output_summary"]) > 0
            assert step["status"] in ("success", "failed")
    finally:
        store.close()
        tmpdir.cleanup()


def test_scenario_15_agent_loops_stop_at_configured_limits():
    tmpdir, store, v_store, emb_provider, gen_provider, settings, doc_id = _create_test_env(
        "General regulatory background text."
    )
    try:
        orchestrator_timeout = AgentOrchestrator(
            goal="Test timeout limit.",
            timeout_seconds=0.0001,
        )
        res_timeout = orchestrator_timeout.run(
            query="What is the regulation?",
            vector_store=v_store,
            provider=emb_provider,
            settings=settings,
            store=store,
            generation_provider=gen_provider,
        )
        assert res_timeout["status"] == "timeout"
        assert "timeout" in str(res_timeout["error"]).lower()

        orchestrator_steps = AgentOrchestrator(
            goal="Test max steps limit.",
            max_steps=2,
        )
        res_steps = orchestrator_steps.run(
            query="What is the regulation?",
            vector_store=v_store,
            provider=emb_provider,
            settings=settings,
            store=store,
            generation_provider=gen_provider,
        )
        assert res_steps["step_count"] <= 2
    finally:
        store.close()
        tmpdir.cleanup()


def test_scenario_16_existing_workflows_100_percent_compatible():
    settings = Settings()
    templates = ["traditional-rag", "hybrid-rag", "reranking-rag", "agentic-rag"]
    for template_id in templates:
        template = TEMPLATES_BY_ID[template_id]
        assert template.graph is not None
        report = validate_traditional_graph(
            template.graph,
            settings,
            generation_provider="extractive",
            generation_model="extractive",
        )
        assert report.valid is True
        assert report.executable is True
        assert len([i for i in report.issues if i.severity == "error"]) == 0
