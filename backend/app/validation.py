"""Deterministic, side-effect-free validation for workflow graphs."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence

from app.contracts import (
    ChunkingChoiceMode,
    ChunkingSelection,
    ChunkingStrategyDefinition,
    ConfigFieldDefinition,
    ConfiguredProviderInstance,
    ConfiguredProviderStatus,
    ConnectionKind,
    ExecutionMode,
    GraphValidationIssue,
    GraphValidationReport,
    ImplementationStatus,
    IssueSeverity,
    NodeDefinition,
    PortDataType,
    WorkflowGraph,
    value_matches_type,
)


def _issue(
    code: str,
    message: str,
    *,
    node_id: str | None = None,
    connection_index: int | None = None,
    severity: IssueSeverity = IssueSeverity.ERROR,
    details: dict | None = None,
) -> GraphValidationIssue:
    return GraphValidationIssue(
        code=code,
        severity=severity,
        message=message,
        node_id=node_id,
        connection_index=connection_index,
        details=details or {},
    )


def _validate_value(
    name: str, value: object, field: ConfigFieldDefinition, node_id: str
) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []
    if not value_matches_type(value, field.value_type):
        return [
            _issue(
                "invalid_config_type",
                f"Configuration '{name}' must be {field.value_type.value}.",
                node_id=node_id,
            )
        ]
    if field.enum is not None and value not in field.enum:
        if (value == "" or value is None) and field.default is not None and not field.required:
            pass  # Fallback to field.default gracefully
        else:
            issues.append(
                _issue(
                    "invalid_config_enum",
                    f"Configuration '{name}' must be one of the declared values.",
                    node_id=node_id,
                    details={"allowed": field.enum},
                )
            )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if field.minimum is not None and value < field.minimum:
            issues.append(
                _issue(
                    "config_below_minimum",
                    f"Configuration '{name}' must be at least {field.minimum}.",
                    node_id=node_id,
                )
            )
        if field.maximum is not None and value > field.maximum:
            issues.append(
                _issue(
                    "config_above_maximum",
                    f"Configuration '{name}' must be at most {field.maximum}.",
                    node_id=node_id,
                )
            )
    if isinstance(value, str):
        if field.min_length is not None and len(value) < field.min_length:
            issues.append(
                _issue(
                    "config_too_short",
                    f"Configuration '{name}' is shorter than its minimum length.",
                    node_id=node_id,
                )
            )
        if field.max_length is not None and len(value) > field.max_length:
            issues.append(
                _issue(
                    "config_too_long",
                    f"Configuration '{name}' exceeds its maximum length.",
                    node_id=node_id,
                )
            )
        if field.pattern is not None:
            import re

            if re.fullmatch(field.pattern, value) is None:
                issues.append(
                    _issue(
                        "config_pattern_mismatch",
                        f"Configuration '{name}' does not match its required format.",
                        node_id=node_id,
                    )
                )
    return issues


def secret_config_issues(graph: WorkflowGraph) -> list[GraphValidationIssue]:
    """Reject credentials embedded in vector-store node config (AD-9).

    Secrets belong in backend env or runtime credentials, never in graph JSON.
    """
    issues: list[GraphValidationIssue] = []
    for node in graph.nodes:
        if node.node_type_id != "vector.index":
            continue
        backend = str(node.config.get("backend") or "chroma").lower()
        if str(node.config.get("api_key") or "").strip():
            issues.append(
                _issue(
                    "secret_in_graph",
                    "Remove 'api_key' from this node; graphs must not contain credentials. "
                    "Set the backend environment variable (e.g. PINECONE_API_KEY) or save the "
                    f"key via /api/v1/providers/credentials/{backend}.",
                    node_id=node.id,
                )
            )
        url = str(node.config.get("url") or "")
        if "@" in url.split("://", 1)[-1].split("/", 1)[0]:
            issues.append(
                _issue(
                    "secret_in_graph",
                    "Remove credentials embedded in 'url'; use a host-only URL and set the "
                    "connection string (e.g. MONGODB_URI) in the backend environment.",
                    node_id=node.id,
                )
            )
    return issues


def _config_issues(
    node_id: str, config: dict, definition: NodeDefinition
) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []
    fields = definition.config_schema.properties
    for key in sorted(config):
        if key not in fields and not definition.config_schema.allow_additional_properties:
            issues.append(
                _issue(
                    "unknown_config_field",
                    f"Configuration '{key}' is not declared by node type '{definition.node_type_id}'.",
                    node_id=node_id,
                )
            )
        elif key in fields:
            issues.extend(_validate_value(key, config[key], fields[key], node_id))
    merged = dict(definition.config_defaults)
    for name, spec in fields.items():
        if spec.default is not None:
            merged.setdefault(name, spec.default)
    merged.update(config)
    for name, spec in fields.items():
        if spec.required and name not in merged:
            issues.append(
                _issue(
                    "missing_required_config",
                    f"Node '{definition.display_name}' requires configuration '{name}'.",
                    node_id=node_id,
                )
            )
    return issues


def validate_graph(
    graph: WorkflowGraph,
    node_definitions: Mapping[str, NodeDefinition],
    *,
    configured_providers: Sequence[ConfiguredProviderInstance] | None = None,
    available_dependencies: set[str] | None = None,
    require_executable: bool = False,
) -> GraphValidationReport:
    issues: list[GraphValidationIssue] = []
    if not graph.nodes:
        issues.append(_issue("empty_graph", "A workflow must contain at least one node."))
    nodes = {}
    for node in graph.nodes:
        if node.id in nodes:
            issues.append(
                _issue(
                    "duplicate_node_id",
                    f"Node identifier '{node.id}' is duplicated.",
                    node_id=node.id,
                )
            )
            continue
        nodes[node.id] = node

    issues.extend(secret_config_issues(graph))

    definitions: dict[str, NodeDefinition] = {}
    for node_id, node in nodes.items():
        definition = node_definitions.get(node.node_type_id)
        if definition is None:
            issues.append(
                _issue(
                    "unknown_node_type",
                    f"Node type '{node.node_type_id}' is not registered.",
                    node_id=node_id,
                )
            )
            continue
        definitions[node_id] = definition
        issues.extend(_config_issues(node_id, node.config, definition))
        if available_dependencies is not None:
            missing = sorted(set(definition.dependencies) - available_dependencies)
            if missing:
                issues.append(
                    _issue(
                        "dependencies_unavailable",
                        f"Node '{definition.display_name}' requires unavailable dependencies.",
                        node_id=node_id,
                        details={"missing_dependencies": missing},
                    )
                )
        if definition.implementation_status in {
            ImplementationStatus.PLANNED,
            ImplementationStatus.EXPERIMENTAL,
        }:
            sev = IssueSeverity.ERROR if require_executable else IssueSeverity.WARNING
            issues.append(
                _issue(
                    "node_not_executable",
                    f"Node '{definition.display_name}' has status '{definition.implementation_status.value}' and has no verified executable runtime.",
                    node_id=node_id,
                    severity=sev,
                )
            )
        if definition.provider_requirements:
            prov_id = node.provider_instance_id
            if prov_id is None and configured_providers:
                for cp in configured_providers:
                    if cp.adapter_id in definition.provider_requirements:
                        prov_id = cp.id
                        break
            if prov_id is None:
                issues.append(
                    _issue(
                        "provider_not_selected",
                        f"Node '{definition.display_name}' requires a configured provider.",
                        node_id=node_id,
                        severity=IssueSeverity.ERROR
                        if require_executable
                        else IssueSeverity.WARNING,
                    )
                )
            elif configured_providers is None:
                issues.append(
                    _issue(
                        "provider_availability_unknown",
                        f"Availability of provider instance '{prov_id}' was not supplied.",
                        node_id=node_id,
                        severity=IssueSeverity.ERROR
                        if require_executable
                        else IssueSeverity.WARNING,
                    )
                )
            elif configured_providers is not None:
                providers = {p.id: p for p in configured_providers}
                provider = providers.get(prov_id)
                if provider is None:
                    issues.append(
                        _issue(
                            "unknown_provider_instance",
                            f"Provider instance '{prov_id}' is not configured.",
                            node_id=node_id,
                            severity=IssueSeverity.ERROR
                            if require_executable
                            else IssueSeverity.WARNING,
                        )
                    )
                elif provider.status == ConfiguredProviderStatus.UNAVAILABLE:
                    issues.append(
                        _issue(
                            "provider_unavailable",
                            f"Provider instance '{provider.id}' is unavailable.",
                            node_id=node_id,
                            severity=IssueSeverity.ERROR
                            if require_executable
                            else IssueSeverity.WARNING,
                        )
                    )
                elif provider.adapter_id not in definition.provider_requirements:
                    issues.append(
                        _issue(
                            "provider_incompatible",
                            f"Provider adapter '{provider.adapter_id}' is not compatible with this node.",
                            node_id=node_id,
                            details={"required": definition.provider_requirements},
                            severity=IssueSeverity.ERROR
                            if require_executable
                            else IssueSeverity.WARNING,
                        )
                    )

    incoming: dict[tuple[str, str], int] = defaultdict(int)
    edge_pairs: set[tuple[str, str, str, str, str]] = set()
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    indegree: dict[str, int] = {node_id: 0 for node_id in nodes}
    cycle_edges: list[int] = []
    for index, edge in enumerate(graph.connections):
        identity = (
            edge.source_node_id,
            edge.source_port_id,
            edge.target_node_id,
            edge.target_port_id,
            edge.kind.value,
        )
        if identity in edge_pairs:
            issues.append(
                _issue(
                    "duplicate_connection", "This connection is duplicated.", connection_index=index
                )
            )
            continue
        edge_pairs.add(identity)
        source = nodes.get(edge.source_node_id)
        target = nodes.get(edge.target_node_id)
        if source is None:
            issues.append(
                _issue(
                    "missing_source_node",
                    f"Source node '{edge.source_node_id}' does not exist.",
                    connection_index=index,
                )
            )
        if target is None:
            issues.append(
                _issue(
                    "missing_target_node",
                    f"Destination node '{edge.target_node_id}' does not exist.",
                    connection_index=index,
                )
            )
        source_def = definitions.get(edge.source_node_id)
        target_def = definitions.get(edge.target_node_id)
        output = (
            next((p for p in source_def.outputs if p.id == edge.source_port_id), None)
            if source_def
            else None
        )
        input_port = (
            next((p for p in target_def.inputs if p.id == edge.target_port_id), None)
            if target_def
            else None
        )
        if source_def and output is None:
            issues.append(
                _issue(
                    "unknown_source_port",
                    f"Output port '{edge.source_port_id}' is not defined for node '{source_def.node_type_id}'.",
                    connection_index=index,
                )
            )
        if target_def and input_port is None:
            issues.append(
                _issue(
                    "unknown_target_port",
                    f"Input port '{edge.target_port_id}' is not defined for node '{target_def.node_type_id}'.",
                    connection_index=index,
                )
            )
        if output and input_port:
            accepted = input_port.accepts or [input_port.data_type]
            if (
                output.data_type != PortDataType.ANY
                and PortDataType.ANY not in accepted
                and output.data_type not in accepted
            ):
                issues.append(
                    _issue(
                        "incompatible_port_types",
                        f"Output type '{output.data_type.value}' cannot connect to input type '{input_port.data_type.value}'.",
                        connection_index=index,
                        details={
                            "source_type": output.data_type.value,
                            "accepted_types": [t.value for t in accepted],
                        },
                    )
                )
            incoming[(edge.target_node_id, edge.target_port_id)] += 1
            if (
                input_port.cardinality.value == "one"
                and incoming[(edge.target_node_id, edge.target_port_id)] > 1
            ):
                issues.append(
                    _issue(
                        "input_cardinality_exceeded",
                        f"Input '{input_port.id}' accepts only one connection.",
                        node_id=edge.target_node_id,
                        connection_index=index,
                    )
                )

        if edge.kind != ConnectionKind.DATA:
            if graph.execution.mode == ExecutionMode.DETERMINISTIC_DAG:
                issues.append(
                    _issue(
                        "unsupported_control_flow",
                        "Conditional and loop edges require an explicit bounded stateful or agentic execution mode.",
                        connection_index=index,
                    )
                )
            elif edge.kind == ConnectionKind.LOOP and graph.execution.max_loop_iterations is None:
                issues.append(
                    _issue(
                        "unbounded_loop",
                        "Loop edges require a declared max_loop_iterations bound.",
                        connection_index=index,
                    )
                )
            else:
                cycle_edges.append(index)
        if (
            source is not None
            and target is not None
            and edge.kind != ConnectionKind.LOOP
            and target.id not in adjacency[source.id]
        ):
            adjacency[source.id].add(target.id)
            indegree[target.id] += 1

    for node_id, definition in definitions.items():
        for input_port in definition.inputs:
            if input_port.required and incoming[(node_id, input_port.id)] == 0:
                issues.append(
                    _issue(
                        "missing_required_input",
                        f"Required input '{input_port.id}' is not connected.",
                        node_id=node_id,
                    )
                )

    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    order: list[str] = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for next_node in sorted(adjacency[current]):
            indegree[next_node] -= 1
            if indegree[next_node] == 0:
                queue.append(next_node)
    if len(order) != len(nodes):
        affected = sorted(node_id for node_id, degree in indegree.items() if degree > 0)
        supports_bounded_loop = (
            graph.execution.mode in {ExecutionMode.STATEFUL_CONDITIONAL, ExecutionMode.AGENTIC}
            and graph.execution.max_steps is not None
            and graph.execution.max_loop_iterations is not None
        )
        if supports_bounded_loop and cycle_edges:
            issues.append(
                _issue(
                    "bounded_cycle_contract_only",
                    "Graph contains cycles; bounds are declared, but Story 2 does not execute stateful graphs.",
                    severity=IssueSeverity.WARNING,
                    details={"node_ids": affected, "connection_indexes": cycle_edges},
                )
            )
        else:
            issues.append(
                _issue(
                    "unsupported_cycle",
                    "Graph contains a cycle unsupported by its execution contract.",
                    details={"node_ids": affected},
                )
            )
    elif cycle_edges:
        issues.append(
            _issue(
                "bounded_control_flow_contract_only",
                "Conditional or loop control flow is represented with explicit bounds, but Story 2 does not execute it.",
                severity=IssueSeverity.WARNING,
                details={
                    "connection_indexes": cycle_edges,
                    "max_steps": graph.execution.max_steps,
                    "max_loop_iterations": graph.execution.max_loop_iterations,
                },
            )
        )

    errors = [issue for issue in issues if issue.severity == IssueSeverity.ERROR]
    blocked_by_warning = any(
        issue.code
        in {
            "provider_not_selected",
            "provider_availability_unknown",
            "provider_unavailable",
            "provider_incompatible",
            "unknown_provider_instance",
            "node_not_executable",
        }
        for issue in issues
    )
    executable = (
        bool(nodes)
        and not errors
        and not blocked_by_warning
        and all(
            definition.implementation_status
            in {ImplementationStatus.IMPLEMENTED, ImplementationStatus.TESTED}
            and definition.runtime_implementation_id
            for definition in definitions.values()
        )
        and all(
            definition.implementation_status
            in {ImplementationStatus.IMPLEMENTED, ImplementationStatus.TESTED}
            for definition in definitions.values()
        )
    )
    valid = not errors
    if valid and executable:
        workflow_state = "executable"
    elif valid:
        workflow_state = "valid"
    else:
        workflow_state = "draft"

    return GraphValidationReport(
        valid=valid,
        executable=bool(executable),
        workflow_state=workflow_state,
        issues=issues,
        topological_order=order if len(order) == len(nodes) else [],
    )


def validate_chunking_selection(
    selection: ChunkingSelection,
    strategies: Mapping[str, ChunkingStrategyDefinition],
    *,
    available_dependencies: set[str] | None = None,
) -> list[GraphValidationIssue]:
    strategy = strategies.get(selection.strategy_id)
    if strategy is None:
        return [
            _issue(
                "unsupported_chunking_strategy",
                f"Chunking strategy '{selection.strategy_id}' is not registered.",
            )
        ]
    issues: list[GraphValidationIssue] = []
    if (
        selection.choice_mode == ChunkingChoiceMode.ACCEPT_RECOMMENDATION
        and not selection.source_recommendation_id
    ):
        issues.append(
            _issue(
                "missing_recommendation_reference",
                "Accepting a recommendation requires its recommendation identifier.",
            )
        )
    if strategy.implementation_status in {
        ImplementationStatus.PLANNED,
        ImplementationStatus.EXPERIMENTAL,
    }:
        issues.append(
            _issue(
                "chunking_strategy_not_implemented",
                f"Strategy '{strategy.id}' is registered as metadata but has no verified runtime.",
            )
        )
    if available_dependencies is not None:
        missing = sorted(set(strategy.required_dependencies) - available_dependencies)
        if missing:
            issues.append(
                _issue(
                    "chunking_dependencies_unavailable",
                    f"Strategy '{strategy.id}' requires unavailable dependencies.",
                    details={"missing_dependencies": missing},
                )
            )
    return issues
