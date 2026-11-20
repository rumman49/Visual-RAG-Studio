"""Explainable deterministic chunk-strategy selection from observed parser signals."""

from __future__ import annotations

from uuid import uuid4

from app.contracts import (
    ChunkingStrategyRecommendation,
    CompatibilityState,
    DocumentEvidenceReference,
    DocumentProfile,
    EvidenceState,
    StrategyCompatibility,
)
from app.rag.chunking import DEFAULT_SETTINGS, SUPPORTED_STRATEGIES


def recommend_chunking(profile: DocumentProfile) -> ChunkingStrategyRecommendation:
    file_type = (
        profile.file_type.value if profile.file_type.state != EvidenceState.UNAVAILABLE else None
    )
    text_length = (
        profile.approximate_text_length.value
        if profile.approximate_text_length.state != EvidenceState.UNAVAILABLE
        else None
    )
    structure = profile.structure
    headings = (
        structure.has_headings.value
        if structure.has_headings.state != EvidenceState.UNAVAILABLE
        else None
    )
    tables = (
        structure.has_tables.value
        if structure.has_tables.state != EvidenceState.UNAVAILABLE
        else None
    )
    code = (
        structure.has_code_blocks.value
        if structure.has_code_blocks.state != EvidenceState.UNAVAILABLE
        else None
    )
    sentences = (
        structure.has_sentence_boundaries.value
        if structure.has_sentence_boundaries.state != EvidenceState.UNAVAILABLE
        else None
    )

    warnings = list(profile.extraction_warnings)
    evidence = [
        DocumentEvidenceReference(
            field_path="file_type", state=profile.file_type.state, value=profile.file_type.value
        ),
        DocumentEvidenceReference(
            field_path="approximate_text_length",
            state=profile.approximate_text_length.state,
            value=profile.approximate_text_length.value,
        ),
        DocumentEvidenceReference(
            field_path="structure.has_headings",
            state=structure.has_headings.state,
            value=structure.has_headings.value,
        ),
        DocumentEvidenceReference(
            field_path="structure.has_sentence_boundaries",
            state=structure.has_sentence_boundaries.state,
            value=structure.has_sentence_boundaries.value,
        ),
        DocumentEvidenceReference(
            field_path="structure.has_tables",
            state=structure.has_tables.state,
            value=structure.has_tables.value,
        ),
        DocumentEvidenceReference(
            field_path="structure.has_code_blocks",
            state=structure.has_code_blocks.state,
            value=structure.has_code_blocks.value,
        ),
    ]

    if file_type == "ma rkdown" and (headings or tables or code):
        strategy = "structure_aware"
        rationale = "Observed Markdown structure is present; section boundaries and heading paths can be retained by the supported structure-aware splitter."
    elif text_length is not None and text_length >= 4000 and sentences:
        strategy = "sentence"
        rationale = "The extracted text is long and sentence boundaries were observed, so sentence-aware grouping can preserve complete sentence boundaries."
    else:
        strategy = "recursive"
        rationale = "Recursive splitting is the supported default for the available text. It handles paragraphs and falls back to smaller separators while preserving extracted page/section metadata."
        if file_type == "pdf" and text_length is not None and text_length < 200:
            warnings.append(
                "PDF extraction produced little text; no specialized structure was assumed."
            )

    if text_length is None or headings is None or sentences is None:
        warnings.append(
            "Some document profile signals are unavailable; the recommendation uses only observed fields and the supported recursive default."
        )

    alternatives = [
        item for item in ("recursive", "fixed", "sentence", "structure_aware") if item != strategy
    ]
    compatibility = [
        StrategyCompatibility(
            component_id=item,
            state=CompatibilityState.COMPATIBLE
            if item in SUPPORTED_STRATEGIES
            else CompatibilityState.INCOMPATIBLE,
            reason="Implemented for extracted text in this milestone."
            if item in SUPPORTED_STRATEGIES
            else "Strategy runtime is not available.",
        )
        for item in [strategy, *alternatives]
    ]
    limitations = ["Language was not detected; no language-specific chunking behavior is applied."]
    if file_type == "pdf":
        limitations.append(
            "PDF layout/OCR reconstruction is not performed; page provenance comes from selectable text extraction."
        )
    return ChunkingStrategyRecommendation(
        recommendation_id=str(uuid4()),
        document_id=profile.document_id,
        recommended_strategy_id=strategy,
        rationale=rationale,
        evidence=evidence,
        proposed_settings=DEFAULT_SETTINGS,
        compatibility=compatibility,
        known_limitations=limitations,
        alternative_strategy_ids=alternatives,
        warnings=warnings,
        unresolved_assumptions=[],
    )
