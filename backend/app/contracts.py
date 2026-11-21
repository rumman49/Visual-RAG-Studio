"""Versioned Pydantic contracts for the capability registry and workflow graphs."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

JsonValue = Any
EvidenceT = TypeVar("EvidenceT")


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ImplementationStatus(StrEnum):
    PLANNED = "planned"
    EXPERIMENTAL = "experimental"
    IMPLEMENTED = "implemented"
    TESTED = "tested"


class CatalogEntryKind(StrEnum):
    ARCHITECTURE = "architecture"
    BLANK_CANVAS = "blank_canvas"


class PortDataType(StrEnum):
    ANY = "any"
    DOCUMENT_SET = "document_set"
    PARSED_DOCUMENTS = "parsed_documents"
    CHUNKS = "chunks"
    EMBEDDINGS = "embeddings"
    VECTOR_INDEX = "vector_index"
    QUERY = "query"
    RETRIEVED_CHUNKS = "retrieved_chunks"
    CONTEXT = "context"
    ANSWER = "answer"
    BOOLEAN = "boolean"
    TEXT = "text"


class PortCardinality(StrEnum):
    ONE = "one"
    MANY = "many"


class ExecutionMode(StrEnum):
    DETERMINISTIC_DAG = "deterministic_dag"
    STATEFUL_CONDITIONAL = "stateful_conditional"
    AGENTIC = "agentic"


class ConnectionKind(StrEnum):
    DATA = "data"
    CONDITIONAL = "conditional"
    LOOP = "loop"


class IssueSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ConfigValueType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ConfigFieldDefinition(ContractModel):
    value_type: ConfigValueType
    description: str
    required: bool = False
    default: JsonValue | None = None
    enum: list[JsonValue] | None = None
    minimum: float | None = None
    maximum: float | None = None
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=0)
    pattern: str | None = None

    @model_validator(mode="after")
    def validate_field_definition(self) -> ConfigFieldDefinition:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum must be less than or equal to maximum")
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError("min_length must be less than or equal to max_length")
        if self.default is not None and not value_matches_type(self.default, self.value_type):
            raise ValueError(f"default value does not match {self.value_type}")
        if self.enum is not None and any(
            not value_matches_type(v, self.value_type) for v in self.enum
        ):
            raise ValueError("every enum value must match the field value_type")
        return self


class NodeConfigSchema(ContractModel):
    properties: dict[str, ConfigFieldDefinition] = Field(default_factory=dict)
    allow_additional_properties: bool = False


class InputPortDefinition(ContractModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str
    data_type: PortDataType
    required: bool = True
    cardinality: PortCardinality = PortCardinality.ONE
    accepts: list[PortDataType] = Field(default_factory=list)
    description: str = ""


class OutputPortDefinition(ContractModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str
    data_type: PortDataType
    cardinality: PortCardinality = PortCardinality.ONE
    description: str = ""


class ExecutionCapabilityMetadata(ContractModel):
    supported_modes: list[ExecutionMode] = Field(
        default_factory=lambda: [ExecutionMode.DETERMINISTIC_DAG]
    )
    supports_cycles: bool = False
    supports_conditional_edges: bool = False
    max_steps_required: bool = False
    description: str = ""


class CapabilityEvidence(ContractModel):
    source_url: str | None = None
    api_reference: str | None = None
    verified_at: str | None = None
    test_references: list[str] = Field(default_factory=list)
    notes: str = ""


class NodeDefinition(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    node_type_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    display_name: str
    category: str
    description: str
    implementation_status: ImplementationStatus = ImplementationStatus.PLANNED
    inputs: list[InputPortDefinition] = Field(default_factory=list)
    outputs: list[OutputPortDefinition] = Field(default_factory=list)
    config_schema: NodeConfigSchema = Field(default_factory=NodeConfigSchema)
    config_defaults: dict[str, JsonValue] = Field(default_factory=dict)
    validation_rules: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    provider_requirements: list[str] = Field(default_factory=list)
    compatibility_constraints: list[str] = Field(default_factory=list)
    supported_document_types: list[str] = Field(default_factory=list)
    supported_languages: list[str] = Field(default_factory=list)
    supported_modalities: list[str] = Field(default_factory=lambda: ["text"])
    execution: ExecutionCapabilityMetadata = Field(default_factory=ExecutionCapabilityMetadata)
    runtime_implementation_id: str | None = None
    known_limitations: list[str] = Field(default_factory=list)
    evidence: list[CapabilityEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_definition_contract(self) -> NodeDefinition:
        ensure_unique([p.id for p in self.inputs], "input port id")
        ensure_unique([p.id for p in self.outputs], "output port id")
        schema_fields = self.config_schema.properties
        for key in self.config_defaults:
            if key not in schema_fields:
                raise ValueError(f"config default '{key}' has no schema property")
        for key, field_def in schema_fields.items():
            if key in self.config_defaults and field_def.default is not None:
                raise ValueError(
                    f"config default '{key}' is defined both in the field and config_defaults"
                )
        if (
            self.implementation_status
            in {ImplementationStatus.IMPLEMENTED, ImplementationStatus.TESTED}
            and not self.runtime_implementation_id
        ):
            raise ValueError("implemented or tested nodes require a runtime_implementation_id")
        if self.implementation_status == ImplementationStatus.TESTED and not self.evidence:
            raise ValueError("tested nodes require recorded implementation evidence")
        return self


class ArchitectureCatalogEntry(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    name: str
    category: str
    kind: CatalogEntryKind = CatalogEntryKind.ARCHITECTURE
    description: str
    implementation_status: ImplementationStatus = ImplementationStatus.PLANNED
    composition_facets: list[str] = Field(default_factory=list)
    required_component_ids: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    execution_model: list[ExecutionMode] = Field(default_factory=list)
    supported_modalities: list[str] = Field(default_factory=list)
    supported_languages: list[str] = Field(default_factory=list)
    configuration_notes: list[str] = Field(default_factory=list)
    validation_rules: list[str] = Field(default_factory=list)
    compatibility_constraints: list[str] = Field(default_factory=list)
    provider_status: str = "unverified"
    known_limitations: list[str] = Field(default_factory=list)
    evidence: list[CapabilityEvidence] = Field(default_factory=list)
    template_id: str | None = None


class ArchitectureTemplateDefinition(ContractModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    architecture_id: str
    name: str
    implementation_status: ImplementationStatus = ImplementationStatus.PLANNED
    graph: WorkflowGraph | None = None
    known_limitations: list[str] = Field(default_factory=list)
    evidence: list[CapabilityEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def tested_template_has_graph_and_evidence(self) -> ArchitectureTemplateDefinition:
        if self.implementation_status == ImplementationStatus.TESTED and (
            self.graph is None or not self.evidence
        ):
            raise ValueError("tested templates require a graph and verification evidence")
        return self


class ProviderAdapterDefinition(ContractModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    display_name: str
    provider_kind: str
    implementation_status: ImplementationStatus = ImplementationStatus.PLANNED
    capabilities: list[str] = Field(default_factory=list)
    config_schema: NodeConfigSchema = Field(default_factory=NodeConfigSchema)
    dependencies: list[str] = Field(default_factory=list)
    runtime_implementation_id: str | None = None
    known_limitations: list[str] = Field(default_factory=list)
    # "verified" only with live end-to-end evidence (Story 1.6); otherwise "unverified".
    verification_status: str = "unverified"
    evidence: list[CapabilityEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def verified_requires_dated_evidence(self) -> ProviderAdapterDefinition:
        if self.verification_status == "verified" and not any(e.verified_at for e in self.evidence):
            raise ValueError("verified providers require dated live evidence")
        return self


class ConfiguredProviderStatus(StrEnum):
    CONFIGURED = "configured"
    UNAVAILABLE = "unavailable"


class ConfiguredProviderInstance(ContractModel):
    id: str
    adapter_id: str
    status: ConfiguredProviderStatus = ConfiguredProviderStatus.CONFIGURED
    config: dict[str, JsonValue] = Field(default_factory=dict)
    secret_references: dict[str, str] = Field(default_factory=dict)


class NodeInstance(ContractModel):
    id: str
    node_type_id: str
    config: dict[str, JsonValue] = Field(default_factory=dict)
    provider_instance_id: str | None = None


class GraphConnection(ContractModel):
    source_node_id: str
    source_port_id: str
    target_node_id: str
    target_port_id: str
    kind: ConnectionKind = ConnectionKind.DATA
    label: str | None = None


class ExecutionContract(ContractModel):
    mode: ExecutionMode = ExecutionMode.DETERMINISTIC_DAG
    max_steps: int | None = Field(default=None, ge=1, le=10000)
    max_loop_iterations: int | None = Field(default=None, ge=1, le=1000)
    timeout_seconds: int = Field(default=120, ge=1, le=3600)

    @model_validator(mode="after")
    def bounded_stateful_mode(self) -> ExecutionContract:
        if (
            self.mode in {ExecutionMode.STATEFUL_CONDITIONAL, ExecutionMode.AGENTIC}
            and self.max_steps is None
        ):
            raise ValueError("stateful and agentic execution modes require a bounded max_steps")
        if self.mode == ExecutionMode.DETERMINISTIC_DAG and self.max_loop_iterations is not None:
            raise ValueError("deterministic DAG mode cannot declare loop iterations")
        return self


class WorkflowGraph(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    id: str
    name: str
    architecture_id: str | None = None
    template_id: str | None = None
    nodes: list[NodeInstance] = Field(default_factory=list)
    connections: list[GraphConnection] = Field(default_factory=list)
    execution: ExecutionContract = Field(default_factory=ExecutionContract)


class GraphValidationIssue(ContractModel):
    code: str
    severity: IssueSeverity
    message: str
    node_id: str | None = None
    connection_index: int | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class GraphValidationReport(ContractModel):
    valid: bool
    executable: bool
    workflow_state: str = "draft"
    issues: list[GraphValidationIssue] = Field(default_factory=list)
    topological_order: list[str] = Field(default_factory=list)


class WorkflowValidationRequest(ContractModel):
    graph: WorkflowGraph
    configured_providers: list[ConfiguredProviderInstance] | None = None
    available_dependencies: list[str] | None = None
    generation_provider: str | None = None
    generation_model: str | None = None
    require_executable: bool = False


class SavedWorkflow(ContractModel):
    id: str
    name: str
    graph: WorkflowGraph
    created_at: str
    updated_at: str


class RegistrySnapshot(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    catalog: list[ArchitectureCatalogEntry]
    templates: list[ArchitectureTemplateDefinition]
    node_types: list[NodeDefinition]
    provider_adapters: list[ProviderAdapterDefinition] = Field(default_factory=list)
    chunking_strategies: list[ChunkingStrategyDefinition] = Field(default_factory=list)


class EvidenceState(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    UNAVAILABLE = "unavailable"


class EvidenceField(ContractModel, Generic[EvidenceT]):
    state: EvidenceState = EvidenceState.UNAVAILABLE
    value: EvidenceT | None = None
    source: str | None = None

    @model_validator(mode="after")
    def evidence_value_matches_state(self) -> EvidenceField[EvidenceT]:
        if self.state == EvidenceState.UNAVAILABLE and self.value is not None:
            raise ValueError("unavailable evidence cannot carry a value")
        if self.state != EvidenceState.UNAVAILABLE and self.value is None:
            raise ValueError("observed or inferred evidence must carry a value")
        return self


class ExtractionStatus(StrEnum):
    NOT_STARTED = "not_started"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class DocumentStructureProfile(ContractModel):
    has_headings: EvidenceField[bool] = Field(default_factory=EvidenceField)
    has_sections: EvidenceField[bool] = Field(default_factory=EvidenceField)
    has_paragraphs: EvidenceField[bool] = Field(default_factory=EvidenceField)
    has_sentence_boundaries: EvidenceField[bool] = Field(default_factory=EvidenceField)
    has_tables: EvidenceField[bool] = Field(default_factory=EvidenceField)
    has_code_blocks: EvidenceField[bool] = Field(default_factory=EvidenceField)
    page_count: EvidenceField[int] = Field(default_factory=EvidenceField)


class DocumentProfile(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    document_id: str
    filename: str
    file_type: EvidenceField[str] = Field(default_factory=EvidenceField)
    parser_id: str | None = None
    extraction_status: ExtractionStatus = ExtractionStatus.NOT_STARTED
    meaningful_text_extracted: EvidenceField[bool] = Field(default_factory=EvidenceField)
    languages: EvidenceField[list[str]] = Field(default_factory=EvidenceField)
    approximate_text_length: EvidenceField[int] = Field(default_factory=EvidenceField)
    structure: DocumentStructureProfile = Field(default_factory=DocumentStructureProfile)
    extraction_warnings: list[str] = Field(default_factory=list)


class ChunkingStrategyStatus(StrEnum):
    PLANNED = "planned"
    EXPERIMENTAL = "experimental"
    IMPLEMENTED = "implemented"
    TESTED = "tested"


class ChunkingStrategyDefinition(ContractModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    display_name: str
    implementation_status: ChunkingStrategyStatus = ChunkingStrategyStatus.PLANNED
    config_schema: NodeConfigSchema = Field(default_factory=NodeConfigSchema)
    required_models: list[str] = Field(default_factory=list)
    required_dependencies: list[str] = Field(default_factory=list)
    compatible_modalities: list[str] = Field(default_factory=lambda: ["text"])
    known_limitations: list[str] = Field(default_factory=list)
    runtime_implementation_id: str | None = None


class RecommendationSource(StrEnum):
    AUTOMATIC = "automatic"
    USER_REQUESTED = "user_requested"


class CompatibilityState(StrEnum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class StrategyCompatibility(ContractModel):
    component_id: str
    state: CompatibilityState
    reason: str


class DocumentEvidenceReference(ContractModel):
    field_path: str
    state: EvidenceState
    value: JsonValue | None = None


class ChunkingStrategyRecommendation(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    recommendation_id: str
    document_id: str
    recommended_strategy_id: str
    rationale: str
    evidence: list[DocumentEvidenceReference] = Field(default_factory=list)
    proposed_settings: dict[str, JsonValue] = Field(default_factory=dict)
    required_models: list[str] = Field(default_factory=list)
    required_dependencies: list[str] = Field(default_factory=list)
    compatibility: list[StrategyCompatibility] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    alternative_strategy_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    unresolved_assumptions: list[str] = Field(default_factory=list)
    source: RecommendationSource = RecommendationSource.AUTOMATIC


class ChunkingChoiceMode(StrEnum):
    ACCEPT_RECOMMENDATION = "accept_recommendation"
    EXPLICIT_OVERRIDE = "explicit_override"


class ChunkingSelection(ContractModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    document_id: str
    strategy_id: str
    choice_mode: ChunkingChoiceMode
    source_recommendation_id: str | None = None
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    plans: list[dict[str, Any]] | None = None
    deduplication_policy: str | None = None

    @model_validator(mode="after")
    def recommendation_reference_required_for_accept(self) -> ChunkingSelection:
        if (
            self.choice_mode == ChunkingChoiceMode.ACCEPT_RECOMMENDATION
            and not self.source_recommendation_id
        ):
            raise ValueError("accepting a recommendation requires source_recommendation_id")
        return self


class ChunkingExecutionRecord(ContractModel):
    document_id: str
    strategy_id: str
    strategy_schema_version: int = Field(ge=1)
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    source_recommendation_id: str | None = None
    selection_mode: ChunkingChoiceMode
    executed_at: str
    chunk_ids: list[str] = Field(default_factory=list)


def value_matches_type(value: JsonValue, value_type: ConfigValueType) -> bool:
    if value_type == ConfigValueType.STRING:
        return isinstance(value, str)
    if value_type == ConfigValueType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if value_type == ConfigValueType.NUMBER:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if value_type == ConfigValueType.BOOLEAN:
        return isinstance(value, bool)
    if value_type == ConfigValueType.ARRAY:
        return isinstance(value, list)
    return isinstance(value, dict)


def ensure_unique(values: list[str], label: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {label}(s): {', '.join(duplicates)}")


ArchitectureTemplateDefinition.model_rebuild()
RegistrySnapshot.model_rebuild()
