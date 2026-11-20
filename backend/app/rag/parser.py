"""UTF-8 text/Markdown and text-based PDF parsing with conservative provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

from app.contracts import (
    DocumentProfile,
    DocumentStructureProfile,
    EvidenceField,
    EvidenceState,
    ExtractionStatus,
)


@dataclass(frozen=True)
class TextSegment:
    text: str
    page_number: int | None = None
    section_path: tuple[str, ...] = ()
    source_start: int | None = None
    source_end: int | None = None


@dataclass
class ParsedDocument:
    document_id: str
    filename: str
    file_type: str
    parser_id: str
    text: str
    segments: list[TextSegment]
    profile: DocumentProfile
    warnings: list[str] = field(default_factory=list)


class DocumentParseError(ValueError):
    """Input could not be parsed into usable text."""


def _evidence(value, source: str) -> EvidenceField:
    return EvidenceField(state=EvidenceState.OBSERVED, value=value, source=source)


def _unavailable() -> EvidenceField:
    return EvidenceField(state=EvidenceState.UNAVAILABLE)


def _text_segments(
    text: str, *, markdown: bool, page_number: int | None = None
) -> list[TextSegment]:
    lines = text.splitlines(keepends=True)
    headings: list[tuple[int, str]] = []
    blocks: list[TextSegment] = []
    block_lines: list[str] = []
    block_start: int | None = None
    offset = 0

    def flush(end_offset: int) -> None:
        nonlocal block_lines, block_start
        raw_block = "".join(block_lines)
        content = raw_block.strip()
        if content:
            leading = len(raw_block) - len(raw_block.lstrip())
            trailing = len(raw_block) - len(raw_block.rstrip())
            content_start = block_start + leading if block_start is not None else None
            content_end = end_offset - trailing
            blocks.append(
                TextSegment(
                    content,
                    page_number,
                    tuple(title for _, title in headings),
                    content_start,
                    content_end,
                )
            )
        block_lines = []
        block_start = None

    for line in lines:
        raw = line.rstrip("\r\n")
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", raw) if markdown else None
        if not markdown:
            match = re.match(
                r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?([A-Z][A-Z0-9 /:&()_-]{3,80})\s*$", raw
            )
        if match:
            flush(offset)
            if markdown:
                depth, title = len(match.group(1)), match.group(2).strip()
            else:
                depth, title = 1, match.group(1).strip()
            headings = [(level, value) for level, value in headings if level < depth]
            headings.append((depth, title))
            offset += len(line)
            continue
        if not raw.strip():
            flush(offset)
            offset += len(line)
            continue
        if block_start is None:
            block_start = offset
        block_lines.append(line)
        offset += len(line)
    flush(offset)
    return blocks


def parse_document(document_id: str, filename: str, content: bytes) -> ParsedDocument:
    safe_name = filename.replace("\\", "/").split("/")[-1]
    suffix = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    warnings: list[str] = []
    page_count: int | None = None

    if suffix in {"txt", "md", "markdown"}:
        try:
            text = content.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as exc:
            raise DocumentParseError("Text documents must be valid UTF-8.") from exc
        file_type = "markdown" if suffix in {"md", "markdown"} else "text"
        parser_id = (
            "builtin-utf8-markdown-v1" if file_type == "markdown" else "builtin-utf8-text-v1"
        )
        segments = _text_segments(text, markdown=file_type == "markdown")
    elif suffix == "pdf":
        try:
            pdf = pymupdf.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise DocumentParseError(
                "The PDF could not be opened. Check that it is a valid PDF file."
            ) from exc
        if pdf.needs_pass:
            pdf.close()
            raise DocumentParseError("Encrypted PDFs are not supported.")
        page_count = pdf.page_count
        pages: list[str] = []
        segments = []
        for number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text")
            pages.append(page_text)
            segments.extend(_text_segments(page_text, markdown=False, page_number=number))
        pdf.close()
        text = "\n\n".join(pages)
        file_type = "pdf"
        parser_id = "pymupdf-text-v1"
        if not text.strip():
            raise DocumentParseError(
                "No selectable text was found in this PDF. OCR is not enabled."
            )
        if len(text.strip()) < 80:
            warnings.append(
                "Very little text was extracted; this may be a scanned or image-only PDF. OCR is not enabled."
            )
    else:
        raise DocumentParseError(
            "Supported file types are .txt, .md, .markdown, and text-based .pdf."
        )

    normalized = text.strip()
    if not normalized:
        raise DocumentParseError("The document contains no usable text.")
    if not segments:
        segments = [TextSegment(normalized, page_number=1 if file_type == "pdf" else None)]

    heading_signal = any(segment.section_path for segment in segments)
    has_tables = bool(re.search(r"(?m)^\s*\|.*\|\s*$", normalized))
    has_code = "```" in normalized or bool(re.search(r"(?m)^\s{4,}\S", normalized))
    profile = DocumentProfile(
        document_id=document_id,
        filename=safe_name,
        file_type=_evidence(file_type, parser_id),
        parser_id=parser_id,
        extraction_status=ExtractionStatus.SUCCEEDED,
        meaningful_text_extracted=_evidence(True, parser_id),
        languages=_unavailable(),
        approximate_text_length=_evidence(len(normalized), parser_id),
        structure=DocumentStructureProfile(
            has_headings=_evidence(heading_signal, parser_id),
            has_sections=_evidence(heading_signal, parser_id),
            has_paragraphs=_evidence("\n\n" in normalized, parser_id),
            has_sentence_boundaries=_evidence(
                bool(re.search(r"[.!?](?:\s|$)", normalized)), parser_id
            ),
            has_tables=_evidence(has_tables, parser_id),
            has_code_blocks=_evidence(has_code, parser_id),
            page_count=_evidence(page_count, parser_id)
            if page_count is not None
            else _unavailable(),
        ),
        extraction_warnings=warnings,
    )
    return ParsedDocument(
        document_id, safe_name, file_type, parser_id, normalized, segments, profile, warnings
    )
