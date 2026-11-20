import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.contracts import (
    ChunkingChoiceMode,
    ChunkingSelection,
    ConfigFieldDefinition,
    ConfiguredProviderInstance,
    ConfiguredProviderStatus,
    ConfigValueType,
    EvidenceField,
    EvidenceState,
    ExecutionContract,
    ExecutionMode,
    GraphConnection,
    ImplementationStatus,
    NodeDefinition,
    NodeInstance,
    WorkflowGraph,
)
from app.main import app
from app.registry import (
    CATALOG_BY_ID,
    CHUNKING_STRATEGIES,
    NODE_TYPES_BY_ID,
    PRIORITY_TEMPLATE_IDS,
    REGISTRY,
    TEMPLATES_BY_ID,
)
from app.validation import validate_chunking_selection, validate_graph


def make_graph(nodes, connections=(), **kwargs):
    return WorkflowGraph(
        id="test", name="test", nodes=nodes, connections=list(connections), **kwargs
    )


def node(node_id, type_id, config=None):
    return NodeInstance(id=node_id, node_type_id=type_id, config=config or {})


def edge(source, source_port, target, target_port, **kwargs):
    return GraphConnection(
        source_node_id=source,
        source_port_id=source_port,
        target_node_id=target,
        target_port_id=target_port,
        **kwargs,
    )


def test_registry_preserves_all_architectures_and_truthful_statuses():
    assert len(REGISTRY.catalog) == 38  # 37 architecture families plus blank canvas
    assert len([entry for entry in REGISTRY.catalog if entry.kind.value == "architecture"]) == 37
    assert CATALOG_BY_ID["traditional-rag"].name == "Traditional RAG"
    assert CATALOG_BY_ID["naive-rag"].name == "Naive RAG"
    assert CATALOG_BY_ID["traditional-rag"].id != CATALOG_BY_ID["naive-rag"].id
    implemented_arch_ids = {"traditional-rag", "hybrid-rag", "reranking-rag", "multilingual-rag", "agentic-rag", "start-from-blank-canvas"}
    assert all(
        entry.implementation_status == ImplementationStatus.IMPLEMENTED
        if entry.id in implemented_arch_ids
        else entry.implementation_status == ImplementationStatus.PLANNED
        for entry in REGISTRY.catalog
    )
    assert set(TEMPLATES_BY_ID) == set(PRIORITY_TEMPLATE_IDS)
    assert TEMPLATES_BY_ID["traditional-rag"].graph is not None
    assert TEMPLATES_BY_ID["traditional-rag"].implementation_status == ImplementationStatus.IMPLEMENTED
    assert TEMPLATES_BY_ID["hybrid-rag"].graph is not None
    assert TEMPLATES_BY_ID["hybrid-rag"].implementation_status == ImplementationStatus.IMPLEMENTED
    assert TEMPLATES_BY_ID["reranking-rag"].graph is not None
    assert TEMPLATES_BY_ID["reranking-rag"].implementation_status == ImplementationStatus.IMPLEMENTED
    assert TEMPLATES_BY_ID["multilingual-rag"].graph is not None
    assert TEMPLATES_BY_ID["multilingual-rag"].implementation_status == ImplementationStatus.IMPLEMENTED
    assert TEMPLATES_BY_ID["agentic-rag"].graph is not None
    assert TEMPLATES_BY_ID["agentic-rag"].implementation_status == ImplementationStatus.IMPLEMENTED
    assert all(
        template.graph is None
        for template in REGISTRY.templates
        if template.id not in {"traditional-rag", "hybrid-rag", "reranking-rag", "multilingual-rag", "agentic-rag"}
    )
    assert all(
        definition.runtime_implementation_id
        for definition in REGISTRY.node_types
        if definition.implementation_status == ImplementationStatus.IMPLEMENTED
    )
    assert {strategy.id for strategy in REGISTRY.chunking_strategies} == set(CHUNKING_STRATEGIES)
    assert {
        strategy.id
        for strategy in REGISTRY.chunking_strategies
        if strategy.implementation_status == "implemented"
    } == {"recursive", "fixed", "sentence", "structure_aware", "hierarchical"}


def test_registry_api_exposes_searchable_catalog_and_validation():
    client = TestClient(app)
    response = client.get(
        "/api/v1/catalog", params={"category": "Foundational", "q": "traditional"}
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == ["traditional-rag"]
    assert client.get("/api/v1/catalog/no-such-entry").status_code == 404
    validation = client.post(
        "/api/v1/workflows/validate",
        json={
            "graph": {
                "id": "smoke",
                "name": "Smoke",
                "nodes": [{"id": "source", "node_type_id": "document.source"}],
                "connections": [],
            }
        },
    )
    assert validation.status_code == 200
    assert validation.json()["valid"] is True
    assert validation.json()["executable"] is True


def test_valid_dag_of_implemented_nodes_is_executable():
    document = node("source", "document.source")
    parser = node("parser", "document.parse")
    graph = make_graph([document, parser], [edge("source", "documents", "parser", "documents")])
    report = validate_graph(graph, NODE_TYPES_BY_ID)
    assert report.valid
    assert report.executable
    assert report.topological_order == ["source", "parser"]


def test_empty_graph_is_invalid():
    report = validate_graph(make_graph([]), NODE_TYPES_BY_ID)
    assert not report.valid
    assert not report.executable
    assert report.issues[0].code == "empty_graph"


def test_duplicate_node_ids_are_reported():
    report = validate_graph(
        make_graph([node("same", "document.source"), node("same", "document.source")]),
        NODE_TYPES_BY_ID,
    )
    assert "duplicate_node_id" in {issue.code for issue in report.issues}


def test_unknown_node_type_and_missing_endpoints_are_reported():
    graph = make_graph([node("mystery", "not.registered")], [edge("ghost", "x", "mystery", "y")])
    codes = {issue.code for issue in validate_graph(graph, NODE_TYPES_BY_ID).issues}
    assert {"unknown_node_type", "missing_source_node"} <= codes


def test_unknown_ports_and_duplicate_connections_are_reported():
    connection = edge("source", "nope", "parser", "documents")
    graph = make_graph(
        [node("source", "document.source"), node("parser", "document.parse")],
        [connection, connection],
    )
    codes = {issue.code for issue in validate_graph(graph, NODE_TYPES_BY_ID).issues}
    assert {"unknown_source_port", "duplicate_connection"} <= codes


def test_incompatible_ports_are_reported():
    graph = make_graph(
        [node("query", "query.input"), node("parser", "document.parse")],
        [edge("query", "query", "parser", "documents")],
    )
    assert "incompatible_port_types" in {
        issue.code for issue in validate_graph(graph, NODE_TYPES_BY_ID).issues
    }


def test_required_inputs_and_required_configuration_are_reported():
    graph = make_graph([node("chunk", "chunk.split")])
    issues = validate_graph(graph, NODE_TYPES_BY_ID).issues
    assert {issue.code for issue in issues} >= {"missing_required_input", "missing_required_config"}


def test_configuration_type_range_enum_and_unknown_keys_are_checked():
    graph = make_graph(
        [
            node("source", "document.source"),
            node("chunk", "chunk.split", {"strategy": "magical", "chunk_size": True, "junk": 1}),
        ],
        [edge("source", "documents", "parser", "documents")],
    )
    # The target id intentionally does not exist; configuration diagnostics remain deterministic.
    codes = {issue.code for issue in validate_graph(graph, NODE_TYPES_BY_ID).issues}
    assert {
        "unknown_config_field",
        "invalid_config_type",
        "invalid_config_enum",
        "missing_required_input",
    } <= codes


def test_duplicate_and_invalid_config_definition_is_rejected():
    with pytest.raises(ValidationError):
        ConfigFieldDefinition(
            value_type=ConfigValueType.INTEGER, description="x", minimum=10, maximum=1
        )


def test_deterministic_cycle_is_rejected_and_bounded_agentic_cycle_is_only_structural():
    # A minimal registered two-node cycle for validator behavior.
    source = NodeDefinition(
        node_type_id="test.source",
        display_name="Source",
        category="test",
        description="x",
        outputs=[{"id": "out", "display_name": "out", "data_type": "text"}],
        implementation_status="planned",
    )
    sink = NodeDefinition(
        node_type_id="test.sink",
        display_name="Sink",
        category="test",
        description="x",
        inputs=[{"id": "in", "display_name": "in", "data_type": "text"}],
        outputs=[{"id": "out", "display_name": "out", "data_type": "text"}],
        implementation_status="planned",
    )
    definitions = {source.node_type_id: source, sink.node_type_id: sink}
    nodes = [node("a", "test.source"), node("b", "test.sink")]
    links = [edge("a", "out", "b", "in"), edge("b", "out", "a", "in")]
    assert "unsupported_cycle" in {
        issue.code for issue in validate_graph(make_graph(nodes, links), definitions).issues
    }
    loop_links = [links[0], edge("b", "out", "a", "in", kind="loop")]
    assert "unsupported_control_flow" in {
        issue.code for issue in validate_graph(make_graph(nodes, loop_links), definitions).issues
    }
    bounded = make_graph(
        nodes,
        loop_links,
        execution=ExecutionContract(
            mode=ExecutionMode.AGENTIC, max_steps=10, max_loop_iterations=3
        ),
    )
    report = validate_graph(bounded, definitions)
    assert "bounded_control_flow_contract_only" in {issue.code for issue in report.issues}
    assert not report.executable


def test_unbounded_agentic_execution_contract_is_rejected():
    with pytest.raises(ValidationError):
        ExecutionContract(mode=ExecutionMode.AGENTIC)


def test_planned_nodes_block_execution_validation():
    graph = make_graph([node("rerank", "retrieval.rerank")])
    codes = {
        issue.code
        for issue in validate_graph(graph, NODE_TYPES_BY_ID, require_executable=True).issues
    }
    assert "node_not_executable" in codes


def test_known_provider_and_dependency_availability_are_checked():
    definition = NodeDefinition(
        node_type_id="test.provider_node",
        display_name="Provider",
        category="test",
        description="x",
        provider_requirements=["test.adapter"],
        dependencies=["test-package"],
    )
    graph = make_graph(
        [
            NodeInstance(
                id="provider", node_type_id=definition.node_type_id, provider_instance_id="p1"
            )
        ]
    )
    provider = ConfiguredProviderInstance(
        id="p1", adapter_id="wrong.adapter", status=ConfiguredProviderStatus.CONFIGURED
    )
    report = validate_graph(
        graph,
        {definition.node_type_id: definition},
        configured_providers=[provider],
        available_dependencies=set(),
    )
    codes = {issue.code for issue in report.issues}
    assert {"provider_incompatible", "dependencies_unavailable"} <= codes


def test_document_profile_allows_unknown_evidence_but_enforces_state_value():
    assert EvidenceField[int]().state == EvidenceState.UNAVAILABLE
    with pytest.raises(ValidationError):
        EvidenceField[int](state=EvidenceState.OBSERVED)
    with pytest.raises(ValidationError):
        EvidenceField[int](state=EvidenceState.UNAVAILABLE, value=12)


def test_chunking_strategy_identifiers_do_not_claim_implementation():
    selection = ChunkingSelection(
        document_id="doc", strategy_id="semantic", choice_mode=ChunkingChoiceMode.EXPLICIT_OVERRIDE
    )
    codes = {issue.code for issue in validate_chunking_selection(selection, CHUNKING_STRATEGIES)}
    assert "chunking_strategy_not_implemented" in codes
    unsupported = ChunkingSelection(
        document_id="doc", strategy_id="unknown", choice_mode=ChunkingChoiceMode.EXPLICIT_OVERRIDE
    )
    assert (
        validate_chunking_selection(unsupported, CHUNKING_STRATEGIES)[0].code
        == "unsupported_chunking_strategy"
    )


def test_accepting_recommendation_requires_reference_and_override_remains_explicit():
    with pytest.raises(ValidationError):
        ChunkingSelection(
            document_id="doc",
            strategy_id="recursive",
            choice_mode=ChunkingChoiceMode.ACCEPT_RECOMMENDATION,
        )
    override = ChunkingSelection(
        document_id="doc",
        strategy_id="recursive",
        choice_mode=ChunkingChoiceMode.EXPLICIT_OVERRIDE,
        settings={"chunk_size": 256},
    )
    assert override.choice_mode == ChunkingChoiceMode.EXPLICIT_OVERRIDE
    assert override.settings["chunk_size"] == 256
