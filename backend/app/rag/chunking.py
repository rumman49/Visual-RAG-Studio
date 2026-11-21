"""Deterministic provenance-preserving text chunkers used by the local pipeline."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from app.rag.parser import ParsedDocument, TextSegment

RAGSPARK_CORE_STRATEGIES = {
    "auto",
    "recursive",
    "fixed",
    "sentence",
    "token",
    "paragraph",
    "markdown",
    "semantic",
    "sliding_window",
    "parent_child",
    "legal_clause",
    "table_aware",
    "code_aware",
    "contextual",
    "adaptive",
}
RAGSPARK_STRATEGIES = {
    *{f"ragspark_{s}" for s in RAGSPARK_CORE_STRATEGIES},
    *{f"ragspark.{s}" for s in RAGSPARK_CORE_STRATEGIES},
    *{f"ragspark-{s}" for s in RAGSPARK_CORE_STRATEGIES},
}
SUPPORTED_STRATEGIES = {
    "recursive",
    "fixed",
    "sentence",
    "structure_aware",
    "hierarchical",
    *RAGSPARK_STRATEGIES,
}
DEFAULT_SETTINGS = {"chunk_size": 1000, "overlap": 120}



class ChunkingError(ValueError):
    pass


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    document_name: str
    ordinal: int
    text: str
    strategy_id: str
    settings: dict[str, int]
    page_number: int | None
    section_path: tuple[str, ...]
    source_start: int | None
    source_end: int | None
    parser_id: str
    warnings: tuple[str, ...] = ()
    parent_id: str | None = None
    parent_text: str | None = None

    def chroma_metadata(self) -> dict[str, str | int]:
        metadata: dict[str, str | int] = {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "ordinal": self.ordinal,
            "strategy_id": self.strategy_id,
            "parser_id": self.parser_id,
            "section_path": " / ".join(self.section_path),
        }
        if self.page_number is not None:
            metadata["page_number"] = self.page_number
        if self.source_start is not None:
            metadata["source_start"] = self.source_start
        if self.source_end is not None:
            metadata["source_end"] = self.source_end
        if self.parent_id is not None:
            metadata["parent_id"] = self.parent_id
        if self.parent_text is not None:
            metadata["parent_text"] = self.parent_text[:1000]
        return metadata


def validate_settings(strategy_id: str, settings: dict | None) -> dict[str, int]:
    if strategy_id not in SUPPORTED_STRATEGIES:
        raise ChunkingError(f"Chunking strategy '{strategy_id}' is not implemented.")
    values = {**DEFAULT_SETTINGS, **(settings or {})}
    if set(values) != {"chunk_size", "overlap"}:
        raise ChunkingError("Chunk settings support only chunk_size and overlap.")
    size, overlap = values["chunk_size"], values["overlap"]
    if isinstance(size, bool) or not isinstance(size, int) or not 64 <= size <= 8000:
        raise ChunkingError("chunk_size must be an integer from 64 to 8000 characters.")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < size:
        raise ChunkingError(
            "overlap must be an integer from 0 up to (but not including) chunk_size."
        )
    return {"chunk_size": size, "overlap": overlap}


def _recursive_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        hard_end = min(start + size, len(text))
        end = hard_end
        if hard_end < len(text):
            floor = start + max(1, size // 2)
            for separator in ("\n\n", "\n", ". ", "; ", " "):
                point = text.rfind(separator, floor, hard_end)
                if point >= floor:
                    end = point + len(separator)
                    break
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            end = hard_end
        spans.append((start, end))
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return spans


def _fixed_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        spans.append((start, end))
        if end == len(text):
            break
        start = end - overlap
    return spans


def _sentence_spans(text: str, size: int, overlap: int) -> tuple[list[tuple[int, int]], bool]:
    matches = list(re.finditer(r".*?(?:[.!?](?=\s|$)|$)", text, re.DOTALL))
    sentences = [(m.start(), m.end()) for m in matches if text[m.start() : m.end()].strip()]
    if not sentences:
        return _recursive_spans(text, size, overlap), True
    spans: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end = 0
    fallback = False
    for sentence_start, sentence_end in sentences:
        sentence = text[sentence_start:sentence_end]
        if len(sentence) > size:
            fallback = True
            if current_start is not None:
                spans.append((current_start, current_end))
                current_start = None
            spans.extend(
                (sentence_start + start, sentence_start + end)
                for start, end in _recursive_spans(sentence, size, overlap)
            )
        elif current_start is None:
            current_start, current_end = sentence_start, sentence_end
        elif sentence_end - current_start <= size:
            current_end = sentence_end
        else:
            spans.append((current_start, current_end))
            next_start = max(current_start, current_end - overlap)
            current_start, current_end = next_start, sentence_end
    if current_start is not None:
        spans.append((current_start, current_end))
    return spans, fallback


def _structure_groups(segments: Iterable[TextSegment]) -> list[list[TextSegment]]:
    groups: list[list[TextSegment]] = []
    previous_key = None
    for segment in segments:
        key = (segment.page_number, segment.section_path)
        if key != previous_key:
            groups.append([])
            previous_key = key
        groups[-1].append(segment)
    return groups


def _hierarchical_chunks(
    document: ParsedDocument, strategy_id: str, config: dict[str, int]
) -> list[TextChunk]:
    parent_size = max(400, config["chunk_size"])
    child_size = max(120, parent_size // 3)
    parent_overlap = min(config["overlap"], parent_size - 1)
    child_overlap = max(10, parent_overlap // 3)

    source_segments = document.segments or [
        TextSegment(
            text=document.text,
            section_path=(),
            page_number=1,
            source_start=0,
            source_end=len(document.text),
        )
    ]
    groups = _structure_groups(source_segments)
    chunks: list[TextChunk] = []

    for g_idx, group in enumerate(groups, start=1):
        group_text = "\n\n".join(segment.text for segment in group).strip()
        if not group_text:
            continue
        section_path = group[-1].section_path
        page_number = (
            group[0].page_number
            if all(s.page_number == group[0].page_number for s in group)
            else None
        )

        parent_spans = _recursive_spans(group_text, parent_size, parent_overlap)
        if not parent_spans:
            parent_spans = [(0, len(group_text))]

        for p_idx, (p_start, p_end) in enumerate(parent_spans, start=1):
            parent_body = group_text[p_start:p_end].strip()
            if not parent_body:
                continue
            parent_id = f"{document.document_id}:parent_{g_idx}_{p_idx}"

            child_spans, _ = _sentence_spans(parent_body, child_size, child_overlap)
            if not child_spans:
                child_spans = _recursive_spans(parent_body, child_size, child_overlap)
            if not child_spans:
                child_spans = [(0, len(parent_body))]

            for c_start, c_end in child_spans:
                child_body = parent_body[c_start:c_end].strip()
                if not child_body:
                    continue
                ordinal = len(chunks)
                digest = hashlib.sha256(
                    f"{document.document_id}|{strategy_id}|{ordinal}|{child_body}".encode()
                ).hexdigest()[:20]
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document.document_id}:{digest}",
                        document_id=document.document_id,
                        document_name=document.filename,
                        ordinal=ordinal,
                        text=child_body,
                        strategy_id=strategy_id,
                        settings=config,
                        page_number=page_number,
                        section_path=(*section_path, f"Parent-{p_idx}"),
                        source_start=None,
                        source_end=None,
                        parser_id=document.parser_id,
                        warnings=(),
                        parent_id=parent_id,
                        parent_text=parent_body,
                    )
                )
    if not chunks:
        raise ChunkingError("Hierarchical chunking produced no usable chunks.")
    return chunks


def chunk_document(
    document: ParsedDocument, strategy_id: str, settings: dict | None = None
) -> list[TextChunk]:
    config = validate_settings(strategy_id, settings)
    core = strategy_id
    for prefix in ("ragspark.", "ragspark_", "ragspark-"):
        if core.startswith(prefix):
            core = core[len(prefix) :]
            break

    if core in {"hierarchical", "parent_child"}:
        return _hierarchical_chunks(document, strategy_id, config)

    if core in {
        "auto",
        "token",
        "paragraph",
        "markdown",
        "semantic",
        "sliding_window",
        "legal_clause",
        "table_aware",
        "code_aware",
        "contextual",
        "adaptive",
    } or strategy_id.startswith(("ragspark.", "ragspark_", "ragspark-")):
        from app.rag.ragspark_adapters import RagSparkChunker

        return RagSparkChunker.chunk(document, strategy_id, config)
    size, overlap = config["chunk_size"], config["overlap"]

    source_segments = document.segments
    groups = (
        _structure_groups(source_segments)
        if strategy_id == "structure_aware"
        else [[s] for s in source_segments]
    )
    chunks: list[TextChunk] = []

    for group in groups:
        group_text = "\n\n".join(segment.text for segment in group)
        if strategy_id == "sentence":
            spans, sentence_fallback = _sentence_spans(group_text, size, overlap)
        elif strategy_id == "fixed":
            spans, sentence_fallback = _fixed_spans(group_text, size, overlap), False
        else:
            spans, sentence_fallback = _recursive_spans(group_text, size, overlap), False
        offsets_known = len(group) == 1 and group[0].source_start is not None
        base_offset = group[0].source_start if offsets_known else None
        warnings = (
            ("A sentence exceeded the limit and was split at a text boundary.",)
            if sentence_fallback
            else ()
        )
        section_path = group[-1].section_path
        page_number = (
            group[0].page_number
            if all(segment.page_number == group[0].page_number for segment in group)
            else None
        )
        for local_start, local_end in spans:
            raw_body = group_text[local_start:local_end]
            body = raw_body.strip()
            if not body:
                continue
            leading = len(raw_body) - len(raw_body.lstrip())
            trailing = len(raw_body) - len(raw_body.rstrip())
            start_offset = base_offset + local_start + leading if base_offset is not None else None
            end_offset = base_offset + local_end - trailing if base_offset is not None else None
            ordinal = len(chunks)
            digest = hashlib.sha256(
                f"{document.document_id}|{strategy_id}|{config}|{ordinal}|{body}".encode()
            ).hexdigest()[:20]
            chunks.append(
                TextChunk(
                    chunk_id=f"{document.document_id}:{digest}",
                    document_id=document.document_id,
                    document_name=document.filename,
                    ordinal=ordinal,
                    text=body,
                    strategy_id=strategy_id,
                    settings=config,
                    page_number=page_number,
                    section_path=section_path,
                    source_start=start_offset,
                    source_end=end_offset,
                    parser_id=document.parser_id,
                    warnings=warnings,
                )
            )
    if not chunks:
        raise ChunkingError("The selected strategy produced no usable chunks.")
    return chunks


@dataclass
class MultiStrategyChunkPlan:
    strategy_id: str
    settings: dict[str, int] | None = None
    priority: int = 1


@dataclass
class MultiStrategyChunkResult:
    chunks: list[TextChunk]
    per_strategy_stats: dict[str, dict[str, Any]]
    total_chunks: int
    deduplication_policy: str
    warnings: list[str]


def chunk_document_multi(
    document: ParsedDocument,
    plans: list[MultiStrategyChunkPlan | dict[str, Any]],
    deduplication_policy: str = "keep_all",
) -> MultiStrategyChunkResult:
    """Execute multiple chunking strategies in parallel, preserving provenance and handling deduplication."""
    if not plans:
        raise ChunkingError("At least one chunking strategy plan is required.")

    valid_dedup_policies = {"keep_all", "exact_dedup", "prefer_structured", "cross_strategy_merge"}
    if deduplication_policy not in valid_dedup_policies:
        raise ChunkingError(
            f"Invalid deduplication_policy '{deduplication_policy}'. Supported: {sorted(valid_dedup_policies)}"
        )

    # 1. Upfront validation of all strategy IDs and duplicate detection
    seen_strategies: list[str] = []
    for item in plans:
        strat_id = (
            item.strategy_id
            if isinstance(item, MultiStrategyChunkPlan)
            else (item.get("strategy_id", "recursive") if isinstance(item, dict) else str(item))
        )
        if strat_id not in SUPPORTED_STRATEGIES:
            raise ChunkingError(
                f"Unsupported chunking strategy '{strat_id}'. Supported strategies: {sorted(SUPPORTED_STRATEGIES)}"
            )
        if strat_id in seen_strategies:
            raise ChunkingError(
                f"Incompatible chunking plan: duplicate strategy '{strat_id}' cannot be combined redundantly."
            )
        seen_strategies.append(strat_id)

    # 2. Check for mutually incompatible strategy combinations
    incompatible_pairs = [
        ({"fixed", "sentence"}, "Combining 'fixed' character slicing with 'sentence' splitting produces conflicting chunk boundaries."),
        ({"ragspark.code_aware", "ragspark.table_aware"}, "Select either table-aware or code-aware chunking as primary structural strategy, not both simultaneously on the same segment."),
    ]
    strat_set = set(seen_strategies)
    for pair, reason in incompatible_pairs:
        if pair.issubset(strat_set):
            raise ChunkingError(f"Incompatible strategy combination {sorted(pair)}: {reason}")

    all_chunks: list[TextChunk] = []
    stats: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for item in plans:
        if isinstance(item, MultiStrategyChunkPlan):
            strat_id = item.strategy_id
            strat_settings = item.settings or {}
            priority = item.priority
        elif isinstance(item, dict):
            strat_id = item.get("strategy_id", "recursive")
            strat_settings = item.get("settings") or {}
            priority = item.get("priority", 1)
        else:
            strat_id = str(item)
            strat_settings = {}
            priority = 1

        t0 = time.perf_counter()
        try:
            strat_chunks = chunk_document(document, strat_id, strat_settings)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            stats[strat_id] = {
                "chunk_count": len(strat_chunks),
                "duration_ms": round(elapsed_ms, 2),
                "error": None,
                "priority": priority,
                "settings": strat_settings,
            }
            all_chunks.extend(strat_chunks)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            err_msg = str(exc)
            stats[strat_id] = {
                "chunk_count": 0,
                "duration_ms": round(elapsed_ms, 2),
                "error": err_msg,
                "priority": priority,
                "settings": strat_settings,
            }
            warnings.append(f"Strategy '{strat_id}' failed: {err_msg}")

    if not all_chunks:
        raise ChunkingError(
            f"All multi-strategy chunking attempts failed: {'; '.join(warnings) if warnings else 'No chunks produced.'}"
        )

    # Apply Deduplication & Merge Policy
    final_chunks: list[TextChunk] = []

    if deduplication_policy == "keep_all":
        # Keep all generated chunks verbatim, preserving each strategy's perspective
        final_chunks = all_chunks

    elif deduplication_policy in ("exact_dedup", "cross_strategy_merge"):
        seen_texts: set[str] = set()
        for chunk in all_chunks:
            normalized = " ".join(chunk.text.split()).lower()
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            final_chunks.append(chunk)

    elif deduplication_policy == "prefer_structured":
        # If a chunk comes from a specialized structure-aware, code-aware, table-aware,
        # or hierarchical parent-child strategy, prioritize it over flat recursive/fixed chunks.
        structured_strategies = {
            "structure_aware", "hierarchical", "parent_child",
            "ragspark.table_aware", "ragspark.code_aware", "ragspark.markdown"
        }
        structured_chunks: list[TextChunk] = []
        unstructured_chunks: list[TextChunk] = []

        for chunk in all_chunks:
            if chunk.strategy_id in structured_strategies or chunk.parent_id is not None:
                structured_chunks.append(chunk)
            else:
                unstructured_chunks.append(chunk)

        seen_snippets: set[str] = set()
        # Add structured first
        for chunk in structured_chunks:
            snip = " ".join(chunk.text.split()[:15]).lower()
            if snip:
                seen_snippets.add(snip)
            final_chunks.append(chunk)

        # Only add unstructured chunks whose text doesn't substantially overlap
        for chunk in unstructured_chunks:
            snip = " ".join(chunk.text.split()[:15]).lower()
            if snip and snip not in seen_snippets:
                seen_snippets.add(snip)
                final_chunks.append(chunk)

    # Re-assign continuous ordinals
    final_ordered = [
        TextChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_name=c.document_name,
            ordinal=idx,
            text=c.text,
            strategy_id=c.strategy_id,
            settings=c.settings,
            page_number=c.page_number,
            section_path=c.section_path,
            source_start=c.source_start,
            source_end=c.source_end,
            parser_id=c.parser_id,
            warnings=c.warnings,
            parent_id=c.parent_id,
            parent_text=c.parent_text,
        )
        for idx, c in enumerate(final_chunks)
    ]

    return MultiStrategyChunkResult(
        chunks=final_ordered,
        per_strategy_stats=stats,
        total_chunks=len(final_ordered),
        deduplication_policy=deduplication_policy,
        warnings=warnings,
    )

