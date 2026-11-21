"""Autonomous Ingestion Strategy Planner for Visual RAG Studio.

Analyzes document structure, multilingual script, size, headings, tables,
and code blocks to plan verified chunking, embedding, and vector store configurations.
Supports:
- Modes: manual, recommendation_only, user_approved_auto, fully_autonomous.
- Single-strategy and parallel multi-strategy chunking with explicit deduplication policy.
- Embedding and vector store compatibility verification (dimensions, models, language support).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.contracts import (
    DocumentProfile,
)
from app.rag.chunking import DEFAULT_SETTINGS, SUPPORTED_STRATEGIES
from app.rag.multilingual import detect_language
from app.rag.parser import ParsedDocument


class PlanningMode(StrEnum):
    MANUAL = "manual"
    RECOMMENDATION_ONLY = "recommendation_only"
    USER_APPROVED_AUTO = "user_approved_auto"
    FULLY_AUTONOMOUS = "fully_autonomous"


class StrategyProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_id: str
    settings: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_SETTINGS))
    priority: int = 1
    reason: str = ""


class EmbeddingProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    dimension: int
    languages: list[str]
    reason: str


class VectorStoreProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str
    collection_name: str
    compatible: bool
    existing_dimension: int | None = None
    target_dimension: int
    notes: str


class IngestionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    document_id: str
    filename: str
    mode: PlanningMode
    selected_strategies: list[StrategyProposal]
    deduplication_policy: str = "keep_all"
    rationale: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    embedding_proposal: EmbeddingProposal
    vector_store_proposal: VectorStoreProposal
    requires_approval: bool = False
    approved: bool = False
    warnings: list[str] = Field(default_factory=list)
    alternative_strategies: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: str(uuid4()))


class IngestionPlanner:
    """Plans ingestion strategies deterministically based on observed document profile signals."""

    VERIFIED_EMBEDDING_MODELS: ClassVar[dict[str, dict[str, Any]]] = {
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
            "provider": "sentence_transformers",
            "dimension": 384,
            "languages": ["en", "hi", "hinglish", "es", "fr", "de", "zh", "multilingual"],
            "max_tokens": 512,
        },
        "all-MiniLM-L6-v2": {
            "provider": "sentence_transformers",
            "dimension": 384,
            "languages": ["en"],
            "max_tokens": 512,
        },
        "fast_local": {
            "provider": "fast_local",
            "dimension": 128,
            "languages": ["en", "universal"],
            "max_tokens": 256,
        },
    }

    @classmethod
    def plan(
        cls,
        document: ParsedDocument,
        profile: DocumentProfile | None = None,
        mode: PlanningMode = PlanningMode.USER_APPROVED_AUTO,
        settings: Settings | None = None,
        existing_vector_store: Any | None = None,
        approval_override: bool = False,
    ) -> IngestionPlan:
        app_settings = settings or Settings()
        doc_profile = profile or document.profile
        doc_id = document.document_id
        filename = document.filename
        text = document.text or ""

        rationale: list[str] = []
        warnings: list[str] = []
        strategies: list[StrategyProposal] = []
        confidence = 0.90

        # 1. Structure Analysis
        struct = doc_profile.structure
        has_headings = struct.has_headings.value is True
        has_tables = struct.has_tables.value is True
        has_code = struct.has_code_blocks.value is True
        has_sentences = struct.has_sentence_boundaries.value is True
        text_len = len(text)

        # 2. Language Detection
        detected_lang = detect_language(text[:2000]) if text else "en"
        is_multilingual = detected_lang in {"hi", "hinglish", "es", "fr", "de"}

        if is_multilingual:
            rationale.append(f"Detected multilingual content in {detected_lang}; prioritizing multilingual-compatible embeddings and boundary preservation.")

        # 3. Strategy Selection (supporting single and multi-strategy)
        if has_tables and has_code:
            strategies.append(
                StrategyProposal(
                    strategy_id="ragspark.table_aware" if "ragspark.table_aware" in SUPPORTED_STRATEGIES else "structure_aware",
                    settings={"chunk_size": 800, "overlap": 80},
                    priority=1,
                    reason="Document contains structured markdown tables; retaining row/column alignment.",
                )
            )
            strategies.append(
                StrategyProposal(
                    strategy_id="ragspark.code_aware" if "ragspark.code_aware" in SUPPORTED_STRATEGIES else "fixed",
                    settings={"chunk_size": 600, "overlap": 60},
                    priority=2,
                    reason="Document contains code blocks; preserving syntax and block enclosure.",
                )
            )
            dedup_policy = "prefer_structured"
            rationale.append("Multi-strategy chunking enabled: pairing specialized table-aware and code-aware chunkers with prefer_structured deduplication.")
        elif has_headings and text_len > 3000:
            strategies.append(
                StrategyProposal(
                    strategy_id="hierarchical",
                    settings={"chunk_size": 600, "overlap": 100},
                    priority=1,
                    reason="Observed hierarchical document structure with headings and substantial length; parent-child chunking preserves both macro context and precise retrieval chunks.",
                )
            )
            dedup_policy = "keep_all"
            rationale.append("Hierarchical chunking selected to preserve section hierarchy and multi-scale context.")
        elif has_sentences and text_len > 1500:
            strategies.append(
                StrategyProposal(
                    strategy_id="sentence",
                    settings={"chunk_size": 500, "overlap": 80},
                    priority=1,
                    reason="Continuous prose with clear sentence boundaries detected; sentence chunking prevents mid-sentence fragmentation.",
                )
            )
            dedup_policy = "exact_dedup"
            rationale.append("Sentence-boundary chunking selected to preserve semantic completeness.")
        else:
            strategies.append(
                StrategyProposal(
                    strategy_id="recursive",
                    settings={"chunk_size": 500, "overlap": 60},
                    priority=1,
                    reason="Standard recursive character splitting provides reliable fallback across paragraph and line boundaries.",
                )
            )
            dedup_policy = "exact_dedup"
            rationale.append("Recursive chunking selected as the robust default splitter.")

        # 4. Embedding Model Recommendation
        if is_multilingual or app_settings.embedding_provider == "sentence_transformers":
            target_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            target_dim = 384
            emb_provider = "sentence_transformers"
            emb_reason = "Verified multilingual MiniLM model supporting English, Hindi, Hinglish, and 50+ languages at 384 dimensions."
        elif app_settings.embedding_provider == "ollama":
            target_model = app_settings.ollama_embedding_model or "nomic-embed-text"
            target_dim = 768
            emb_provider = "ollama"
            emb_reason = "Local Ollama embedding model."
        else:
            target_model = "fast_local"
            target_dim = 128
            emb_provider = "fast_local"
            emb_reason = "Zero-dependency lightweight local embedding provider."

        embedding_prop = EmbeddingProposal(
            provider=emb_provider,
            model=target_model,
            dimension=target_dim,
            languages=[detected_lang, "multilingual" if is_multilingual else "en"],
            reason=emb_reason,
        )

        # 5. Vector Store Compatibility Check
        v_backend = "chroma" if app_settings.chroma_path else "memory"
        existing_dim = None
        compatible = True
        notes = "Vector store is ready for target embedding dimensions."

        if existing_vector_store is not None:
            if hasattr(existing_vector_store, "_metadata"):
                col_meta = existing_vector_store._metadata.get(existing_vector_store.collection_name, {})
                existing_dim = col_meta.get("embedding_dimension")
            elif hasattr(existing_vector_store, "collection"):
                try:
                    meta = existing_vector_store.collection.metadata or {}
                    existing_dim = meta.get("dimension")
                except Exception:
                    existing_dim = None

            if existing_dim is not None and existing_dim != target_dim:
                compatible = False
                confidence = max(0.40, confidence - 0.40)
                notes = (
                    f"Vector dimension mismatch! Existing collection has dimension {existing_dim}, "
                    f"but proposed model '{target_model}' outputs dimension {target_dim}. "
                    "Reindexing or a new collection name is strictly required to prevent vector corruption."
                )
                warnings.append(notes)

        vstore_prop = VectorStoreProposal(
            backend=v_backend,
            collection_name="visual_rag_studio_documents",
            compatible=compatible,
            existing_dimension=existing_dim,
            target_dimension=target_dim,
            notes=notes,
        )

        # 6. Mode & Approval Policy
        if mode == PlanningMode.MANUAL:
            requires_approval = True
            approved = approval_override
        elif mode == PlanningMode.RECOMMENDATION_ONLY:
            requires_approval = True
            approved = False
        elif mode == PlanningMode.USER_APPROVED_AUTO:
            requires_approval = not approval_override
            approved = approval_override
        elif mode == PlanningMode.FULLY_AUTONOMOUS:
            if not compatible or confidence < 0.70:
                requires_approval = True
                approved = False
                warnings.append("Autonomous execution paused: confidence threshold or vector store incompatibility requires explicit user review.")
            else:
                requires_approval = False
                approved = True
        else:
            requires_approval = True
            approved = False

        alt_strategies = [s for s in ("recursive", "sentence", "fixed", "hierarchical") if s not in {sp.strategy_id for sp in strategies}]

        return IngestionPlan(
            plan_id=str(uuid4()),
            document_id=doc_id,
            filename=filename,
            mode=mode,
            selected_strategies=strategies,
            deduplication_policy=dedup_policy,
            rationale=rationale,
            confidence=round(confidence, 2),
            embedding_proposal=embedding_prop,
            vector_store_proposal=vstore_prop,
            requires_approval=requires_approval,
            approved=approved,
            warnings=warnings,
            alternative_strategies=alt_strategies,
        )
