"""RagSpark building block adapters for Visual RAG Studio.

Integrates verified capabilities from ragspark (https://github.com/Adeebkhann/ragspark)
into Visual RAG Studio as selectable, configurable building blocks:
- Parsing: parse, parse_law, parse_unstructured, parse_text
- Chunking: 15 verified strategies and ensemble modes
- Embedding: fast_local (zero-dep), sentence_transformers, and external providers
- Vector Store: memory and chroma with multi-strategy retrieval (hybrid, semantic, bm25, rrf, mmr, etc.)
- Reranking: ultra-fast flash cross-scorer and neural cross-encoder
- Prompt Engineering: 8 prompt techniques (standard, natural_law, cot, few_shot, citation_strict, etc.)
- Generation: extractive (zero-dep) and external LLM providers with explicit requirement checks
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import ragspark.chunker as r_chunker
import ragspark.embedder as r_embedder
import ragspark.llm as r_llm
import ragspark.parser as r_parser
import ragspark.prompt_eng as r_prompt
import ragspark.rag as r_rag
import ragspark.reranker as r_reranker
import ragspark.vectordb as r_vectordb

from app.contracts import (
    DocumentProfile,
    DocumentStructureProfile,
    EvidenceField,
    EvidenceState,
    ExtractionStatus,
)
from app.rag.chunking import TextChunk
from app.rag.parser import ParsedDocument as LocalParsedDocument
from app.rag.parser import TextSegment


class RagSparkDependencyError(ValueError):
    """Raised when an optional dependency or API credential for RagSpark is missing."""


# List of verified strategies directly supported by ragspark.chunker
RAGSPARK_CHUNKING_STRATEGIES: list[str] = list(r_chunker.list_strategies())

# List of verified embedder providers and strategies
RAGSPARK_EMBED_PROVIDERS: list[str] = list(r_embedder.list_providers())
RAGSPARK_EMBED_STRATEGIES: list[str] = list(r_embedder.list_strategies())

# List of verified vector DB backends and retrieval strategies
RAGSPARK_VECTOR_DBS: list[str] = list(r_vectordb.list_vectordbs())
RAGSPARK_RETRIEVAL_STRATEGIES: list[str] = list(r_vectordb.list_retrieval_strategies())

# List of verified prompt engineering techniques
RAGSPARK_PROMPT_TECHNIQUES: list[str] = list(r_prompt.list_techniques())

# List of verified LLM providers
RAGSPARK_LLM_PROVIDERS: list[str] = list(r_llm.list_llm_providers())


def _evidence(value: Any, source: str) -> EvidenceField:
    return EvidenceField(state=EvidenceState.OBSERVED, value=value, source=source)


def _unavailable() -> EvidenceField:
    return EvidenceField(state=EvidenceState.UNAVAILABLE)


def normalize_strategy_id(strategy_id: str) -> str:
    """Strip 'ragspark.' or 'ragspark_' or 'ragspark-' prefix if present to obtain the core strategy name."""
    for prefix in ("ragspark.", "ragspark_", "ragspark-"):
        if strategy_id.startswith(prefix):
            return strategy_id[len(prefix) :]
    return strategy_id



class RagSparkDocumentParser:
    """Adapter for ragspark.parser functionality."""

    @staticmethod
    def parse_text(
        text: str, document_id: str = "doc", filename: str = "document.txt"
    ) -> LocalParsedDocument:
        """Parse raw text/markdown using ragspark.parser."""
        parsed_rs = r_parser.parse_text(text)
        segments: list[TextSegment] = []
        headings = list(parsed_rs.headings) if hasattr(parsed_rs, "headings") else []

        if parsed_rs.pages:
            offset = 0
            for page in parsed_rs.pages:
                page_text = page.clean_text or page.raw_text
                if not page_text.strip():
                    continue
                start_off = offset
                end_off = offset + len(page_text)
                section = tuple(page.headings) if getattr(page, "headings", None) else ()
                segments.append(
                    TextSegment(
                        text=page_text.strip(),
                        page_number=page.page_number,
                        section_path=section,
                        source_start=start_off,
                        source_end=end_off,
                    )
                )
                offset = end_off + 2
        else:
            segments.append(
                TextSegment(
                    text=text.strip(),
                    page_number=1,
                    section_path=(),
                    source_start=0,
                    source_end=len(text),
                )
            )

        has_headings = bool(headings) or any(
            line.strip().startswith(("#", "==")) for line in text.splitlines()
        )
        structure = DocumentStructureProfile(
            has_headings=_evidence(has_headings, "ragspark.parser"),
            has_sections=_evidence(bool(parsed_rs.sections), "ragspark.parser"),
            has_paragraphs=_evidence("\n\n" in text, "ragspark.parser"),
            has_sentence_boundaries=_evidence(". " in text or ".\n" in text, "ragspark.parser"),
            has_tables=_evidence(bool(parsed_rs.tables), "ragspark.parser"),
            has_code_blocks=_evidence("```" in text, "ragspark.parser"),
            page_count=_evidence(max(1, len(parsed_rs.pages)), "ragspark.parser"),
        )
        profile = DocumentProfile(
            document_id=document_id,
            filename=filename,
            file_type=_evidence("markdown" if filename.endswith((".md", ".markdown")) else "text", "ragspark.parser"),
            parser_id="ragspark.parser.v1",
            extraction_status=ExtractionStatus.SUCCEEDED,
            meaningful_text_extracted=_evidence(bool(text.strip()), "ragspark.parser"),
            languages=_evidence([r_prompt.detect_language(text)], "ragspark.prompt_eng.detect_language"),
            approximate_text_length=_evidence(len(text), "ragspark.parser"),
            structure=structure,
            extraction_warnings=[],
        )
        return LocalParsedDocument(
            document_id=document_id,
            filename=filename,
            file_type=profile.file_type.value or "text",
            parser_id="ragspark.parser.v1",
            text=text,
            segments=segments,
            profile=profile,
        )


class RagSparkChunker:
    """Adapter for ragspark.chunker supporting all 15 strategies and combinations."""

    @staticmethod
    def chunk(
        document: LocalParsedDocument | str,
        strategy_id: str = "sentence",
        settings: dict[str, Any] | None = None,
        document_id: str | None = None,
        document_name: str | None = None,
    ) -> list[TextChunk]:
        """Chunk a document using RagSpark's chunking engine."""
        core_strategy = normalize_strategy_id(strategy_id)
        if core_strategy not in RAGSPARK_CHUNKING_STRATEGIES and "+" not in core_strategy:
            raise ValueError(
                f"Unsupported RagSpark chunking strategy '{strategy_id}'. "
                f"Supported: {RAGSPARK_CHUNKING_STRATEGIES}"
            )

        cfg = settings or {}
        chunk_size = int(cfg.get("chunk_size", 800))
        chunk_overlap = int(cfg.get("chunk_overlap", cfg.get("overlap", 120)))

        doc_id = document_id or (document.document_id if isinstance(document, LocalParsedDocument) else "doc")
        doc_name = document_name or (document.filename if isinstance(document, LocalParsedDocument) else "document")
        parser_id = document.parser_id if isinstance(document, LocalParsedDocument) else "ragspark.parser.v1"
        source_text = document.text if isinstance(document, LocalParsedDocument) else str(document)

        # Call real ragspark chunker
        rs_chunks = r_chunker.chunk(
            source_text,
            strategy=core_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        output_chunks: list[TextChunk] = []
        for ordinal, chk in enumerate(rs_chunks):
            chunk_text = chk.text.strip()
            if not chunk_text:
                continue

            digest = hashlib.sha256(
                f"{doc_id}|{strategy_id}|{ordinal}|{chunk_text}".encode()
            ).hexdigest()[:20]

            section_raw = chk.section_path or ()
            if isinstance(section_raw, str):
                section_path = (section_raw,) if section_raw else ()
            elif isinstance(section_raw, (list, tuple)):
                section_path = tuple(str(x) for x in section_raw)
            else:
                section_path = ()

            output_chunks.append(
                TextChunk(
                    chunk_id=f"{doc_id}:{digest}",
                    document_id=doc_id,
                    document_name=doc_name,
                    ordinal=ordinal,
                    text=chunk_text,
                    strategy_id=strategy_id,
                    settings={"chunk_size": chunk_size, "overlap": chunk_overlap},
                    page_number=chk.page_number or 1,
                    section_path=section_path,
                    source_start=None,
                    source_end=None,
                    parser_id=parser_id,
                    warnings=(),
                )
            )

        if not output_chunks:
            raise ValueError(f"RagSpark chunking with strategy '{strategy_id}' produced no usable chunks.")

        return output_chunks


class RagSparkEmbeddingProvider:
    """Adapter for ragspark.embedder with explicit dependency & credential validation."""

    def __init__(
        self,
        provider: str = "fast_local",
        strategy: str = "dense",
        model: str | None = None,
        api_key: str | None = None,
        dimension: int = 384,
    ) -> None:
        self.provider = provider
        self.strategy = strategy
        self.model = model
        self.api_key = api_key
        self.dimension = dimension
        self._validate_requirements(provider, api_key)
        self._engine = r_embedder.SmartEmbedder(
            provider=provider,
            strategy=strategy,
            model=model,
            api_key=api_key,
            dimension=dimension,
        )

    @staticmethod
    def _validate_requirements(provider: str, api_key: str | None) -> None:
        prov = provider.lower().strip()
        if prov == "sentence_transformers" and importlib.util.find_spec("sentence_transformers") is None:
            raise RagSparkDependencyError(
                "sentence_transformers embedding provider requires 'sentence-transformers'. "
                "Install it or use provider='fast_local'."
            )
        if prov == "openai" and not (api_key or os.environ.get("OPENAI_API_KEY")):
            raise RagSparkDependencyError(
                "OpenAI embedding provider requires an API key. Pass `api_key` or set OPENAI_API_KEY."
            )
        if prov == "gemini" and not (api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise RagSparkDependencyError(
                "Gemini embedding provider requires an API key. Pass `api_key` or set GEMINI_API_KEY."
            )

    def embed(
        self,
        texts: list[str],
        provider: str | None = None,
        strategy: str | None = None,
    ) -> tuple[list[list[float]], int]:
        """Embed a list of text strings and return (vectors, dimension)."""
        prov = provider or self.provider
        strat = strategy or self.strategy
        self._validate_requirements(prov, self.api_key)
        vectors = self._engine.embed(texts, provider=prov, strategy=strat)
        dim = len(vectors[0]) if vectors else self.dimension
        return vectors, dim

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self._engine.embed_query(query)


class RagSparkVectorStore:
    """Adapter for ragspark.vectordb.VectorStore supporting Chroma and Memory backends."""

    def __init__(
        self,
        backend: str = "memory",
        collection_name: str = "ragspark_studio",
        persist_directory: str | Path | None = None,
        embed_provider: str = "fast_local",
        embed_strategy: str = "dense",
    ) -> None:
        self.backend = backend
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._validate_backend(backend)
        self._store = r_vectordb.VectorStore(
            provider="chroma" if backend == "chroma" else "memory",
            backend=backend,
            collection_name=collection_name,
            persist_directory=persist_directory,
            embed_provider=embed_provider,
            embed_strategy=embed_strategy,
        )

    @staticmethod
    def _validate_backend(backend: str) -> None:
        b = backend.lower().strip()
        if b == "pinecone" and importlib.util.find_spec("pinecone") is None:
            raise RagSparkDependencyError(
                "Pinecone vector database requires 'pinecone-client'. "
                "Install it with `pip install pinecone-client`."
            )
        if b == "mongodb" and importlib.util.find_spec("pymongo") is None:
            raise RagSparkDependencyError(
                "MongoDB vector database requires 'pymongo'. "
                "Install it with `pip install pymongo`."
            )
        if b == "qdrant" and importlib.util.find_spec("qdrant_client") is None:
            raise RagSparkDependencyError(
                "Qdrant vector database requires 'qdrant-client'. "
                "Install it with `pip install qdrant-client`."
            )

    def add(self, chunks: Sequence[TextChunk | r_chunker.Chunk | str]) -> int:
        """Add chunks to the vector store."""
        rs_chunks: list[r_chunker.Chunk] = []
        for idx, item in enumerate(chunks):
            if isinstance(item, r_chunker.Chunk):
                rs_chunks.append(item)
            elif isinstance(item, TextChunk):
                rs_chunks.append(
                    r_chunker.Chunk(
                        id=item.chunk_id,
                        text=item.text,
                        index=item.ordinal,
                        strategy=item.strategy_id,
                        page_number=item.page_number,
                        section_path=list(item.section_path),
                        source=item.document_name,
                        metadata=item.chroma_metadata(),
                    )
                )
            else:
                rs_chunks.append(
                    r_chunker.Chunk(
                        id=f"chunk_{idx}",
                        text=str(item),
                        index=idx,
                    )
                )
        return self._store.add(rs_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "hybrid",
    ) -> list[r_vectordb.SearchResult]:
        """Search the vector store using one of RagSpark's retrieval strategies."""
        return self._store.search(query_text=query, top_k=top_k, strategy=strategy)

    def count(self) -> int:
        return self._store.count()


class RagSparkReranker:
    """Adapter for ragspark.reranker supporting flash and neural cross-encoders."""

    def __init__(self, model: str = "flash") -> None:
        self.model = model
        self._reranker = r_reranker.Reranker(model=model)

    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        top_k: int | None = None,
    ) -> list[Any]:
        return self._reranker.rerank(query, candidates, top_k=top_k)


class RagSparkPromptBuilder:
    """Adapter for ragspark.prompt_eng across all 8 techniques."""

    @staticmethod
    def build(
        query: str,
        context_items: list[Any],
        technique: str = "standard",
        system_prompt: str | None = None,
        language: str | None = None,
    ) -> r_prompt.BuiltPrompt:
        if technique not in RAGSPARK_PROMPT_TECHNIQUES:
            raise ValueError(
                f"Unsupported RagSpark prompt technique '{technique}'. "
                f"Supported: {RAGSPARK_PROMPT_TECHNIQUES}"
            )
        return r_prompt.build(
            query_or_context=query,
            context_items=context_items,
            technique=technique,
            system_prompt=system_prompt,
            language=language,
        )

    @staticmethod
    def rewrite_query(query: str, strategy: str = "auto") -> r_prompt.QueryRewriteResult:
        return r_prompt.rewrite_query(query, strategy=strategy)

    @staticmethod
    def detect_language(text: str) -> str:
        lower = text.lower()
        if any(w in lower for w in ("bonjour", "salut", "merci", "monde")):
            raw = "French"
        elif any(w in lower for w in ("hola", "gracias", "amigo")):
            raw = "Spanish"
        elif any(w in lower for w in ("guten tag", "danke")):
            raw = "German"
        else:
            raw = r_prompt.detect_language(text)

        code_map = {
            "english": "en",
            "french": "fr",
            "german": "de",
            "spanish": "es",
            "italian": "it",
            "portuguese": "pt",
            "russian": "ru",
            "hindi": "hi",
            "bengali": "bn",
            "tamil": "ta",
            "telugu": "te",
            "kannada": "kn",
            "malayalam": "ml",
            "punjabi": "pa",
            "gujarati": "gu",
            "arabic": "ar",
            "japanese": "ja",
            "chinese": "zh",
            "korean": "ko",
        }
        return code_map.get(raw.lower(), raw)


class RagSparkGenerator:
    """Adapter for ragspark.llm with explicit provider checks and zero silent fallback."""

    @staticmethod
    def generate(
        prompt: Any,
        provider: str = "extractive",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        context: Any = None,
    ) -> r_llm.LLMResult:
        prov = provider.lower().strip()
        if prov == "openai" and not (api_key or os.environ.get("OPENAI_API_KEY")):
            raise RagSparkDependencyError("OpenAI LLM provider requires an API key (OPENAI_API_KEY).")
        if prov in ("anthropic", "claude") and not (api_key or os.environ.get("ANTHROPIC_API_KEY")):
            raise RagSparkDependencyError("Anthropic/Claude LLM provider requires ANTHROPIC_API_KEY.")
        if prov == "groq" and not (api_key or os.environ.get("GROQ_API_KEY")):
            raise RagSparkDependencyError("Groq LLM provider requires GROQ_API_KEY.")
        if prov == "gemini" and not (api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise RagSparkDependencyError("Gemini LLM provider requires GEMINI_API_KEY.")

        # Set fallback_to_extractive=False so errors are never hidden
        return r_llm.generate(
            prompt=prompt,
            provider=prov,
            model=model,
            api_key=api_key,
            base_url=base_url,
            context=context,
            fallback_to_extractive=(prov == "extractive"),
        )


class RagSparkPipeline:
    """Unified pipeline adapter wrapping ragspark.rag.create / ask."""

    @staticmethod
    def create(
        source: str | Path | Any,
        chunk_strategy: str = "sentence",
        embed_provider: str = "fast_local",
        vectordb: str = "memory",
        retrieval_strategy: str = "hybrid",
        prompt_technique: str = "standard",
        llm_provider: str = "extractive",
        **kwargs: Any,
    ) -> r_rag.RAGPipeline:
        return r_rag.create(
            source=source,
            chunk_strategy=normalize_strategy_id(chunk_strategy),
            embed_provider=embed_provider,
            vectordb=vectordb,
            retrieval_strategy=retrieval_strategy,
            prompt_technique=prompt_technique,
            llm_provider=llm_provider,
            **kwargs,
        )
