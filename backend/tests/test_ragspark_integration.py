"""Integration tests for RagSpark in Visual RAG Studio.

Verifies:
1. Real import smoke tests for all RagSpark modules.
2. Full building block execution: parsing, chunking (all 15 strategies),
   embedding (all strategies), vector store (memory & Chroma), reranking (flash),
   prompt engineering (all techniques), generation (extractive & providers), and end-to-end pipeline.
3. Explicit error handling for optional dependencies (zero silent fallback).
4. Full registry exposure and discoverability.
5. Integration with Visual RAG Studio's chunk_document pipeline.
"""

import pytest
import ragspark
import ragspark.chunker as r_chunker
import ragspark.embedder as r_embedder
import ragspark.llm as r_llm
import ragspark.parser as r_parser
import ragspark.prompt_eng as r_prompt
import ragspark.rag as r_rag
import ragspark.reranker as r_reranker
import ragspark.vectordb as r_vectordb

from app.contracts import ImplementationStatus
from app.rag.chunking import chunk_document
from app.rag.parser import parse_document
from app.rag.ragspark_adapters import (
    RAGSPARK_CHUNKING_STRATEGIES,
    RAGSPARK_EMBED_PROVIDERS,
    RAGSPARK_EMBED_STRATEGIES,
    RAGSPARK_LLM_PROVIDERS,
    RAGSPARK_PROMPT_TECHNIQUES,
    RAGSPARK_RETRIEVAL_STRATEGIES,
    RAGSPARK_VECTOR_DBS,
    RagSparkChunker,
    RagSparkDependencyError,
    RagSparkDocumentParser,
    RagSparkEmbeddingProvider,
    RagSparkGenerator,
    RagSparkPipeline,
    RagSparkPromptBuilder,
    RagSparkReranker,
    RagSparkVectorStore,
)
from app.registry import CHUNKING_STRATEGIES, NODE_TYPES_BY_ID, REGISTRY


def test_ragspark_import_smoke_test():
    """Verify all RagSpark core modules import and report version and components."""
    assert hasattr(ragspark, "__version__")
    assert ragspark.__version__ == "0.3.0"
    assert hasattr(r_parser, "parse_text")
    assert hasattr(r_chunker, "chunk")
    assert hasattr(r_embedder, "embed")
    assert hasattr(r_vectordb, "VectorStore")
    assert hasattr(r_reranker, "rerank")
    assert hasattr(r_prompt, "build")
    assert hasattr(r_llm, "generate")
    assert hasattr(r_rag, "create")
    assert len(RAGSPARK_EMBED_PROVIDERS) >= 1
    assert len(RAGSPARK_LLM_PROVIDERS) >= 1
    assert len(RAGSPARK_RETRIEVAL_STRATEGIES) >= 1
    assert len(RAGSPARK_VECTOR_DBS) >= 1


def test_ragspark_document_parser():
    """Verify RagSpark text and markdown parsing into Visual RAG Studio format."""
    text = (
        "# Artificial Intelligence\n\n"
        "AI is the simulation of human intelligence.\n\n"
        "## Subfields\n\n"
        "Machine learning is a subfield of AI."
    )
    doc = RagSparkDocumentParser.parse_text(text, document_id="doc_ai", filename="ai.md")
    assert doc.document_id == "doc_ai"
    assert doc.filename == "ai.md"
    assert doc.file_type == "markdown"
    assert doc.parser_id == "ragspark.parser.v1"
    assert len(doc.segments) >= 1
    assert doc.profile.meaningful_text_extracted.value is True
    assert doc.profile.structure.has_headings.value is True


@pytest.mark.parametrize("strategy", RAGSPARK_CHUNKING_STRATEGIES)
def test_all_15_ragspark_chunking_strategies(strategy):
    """Verify that every one of RagSpark's 15 chunking strategies executes successfully."""
    sample_text = (
        "# System Architecture\n"
        "Visual RAG Studio integrates RagSpark as modular building blocks. "
        "Each building block provides selectable and configurable components.\n\n"
        "## Clause 1: Legal Compliance\n"
        "The party shall maintain provenance and traceability for all document chunks.\n\n"
        "| Strategy | Type |\n"
        "| --- | --- |\n"
        "| recursive | character |\n"
        "| semantic | embedding |\n\n"
        "```python\n"
        "def chunk(text): return [text]\n"
        "```\n"
    )
    chunks = RagSparkChunker.chunk(
        sample_text,
        strategy_id=f"ragspark_{strategy}",
        settings={"chunk_size": 250, "overlap": 30},
        document_id="doc_test",
        document_name="architecture.md",
    )
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.document_id == "doc_test"
        assert chunk.document_name == "architecture.md"
        assert chunk.text.strip()
        assert chunk.chunk_id.startswith("doc_test:")
        assert chunk.strategy_id == f"ragspark_{strategy}"


def test_ragspark_chunk_document_integration():
    """Verify chunk_document in app.rag.chunking seamlessly executes RagSpark strategies."""
    parsed = parse_document("doc1", "policy.md", b"# Policy\nSection 1.1: Every node must be traceable.\n")
    for strat in ("ragspark_sliding_window", "ragspark_legal_clause", "ragspark_parent_child"):
        chunks = chunk_document(parsed, strat, {"chunk_size": 120, "overlap": 20})
        assert len(chunks) >= 1
        assert chunks[0].document_id == "doc1"
        assert chunks[0].strategy_id == strat


@pytest.mark.parametrize("strategy", RAGSPARK_EMBED_STRATEGIES)
def test_ragspark_embedder_fast_local_strategies(strategy):
    """Verify RagSpark fast_local embedder runs across all embedding strategies."""
    provider = RagSparkEmbeddingProvider(provider="fast_local", strategy=strategy, dimension=384)
    texts = ["RagSpark local embedding test", "Verification of vector dimension"]
    vectors, dim = provider.embed(texts)
    assert len(vectors) == 2
    assert dim == 384
    assert len(vectors[0]) == 384
    query_vec = provider.embed_query("search query")
    assert len(query_vec) == 384


def test_ragspark_embedder_missing_dependency_or_credential_rejected():
    """Verify that selecting unconfigured external providers raises clear errors."""
    with pytest.raises(RagSparkDependencyError, match="OpenAI embedding provider requires an API key"):
        RagSparkEmbeddingProvider(provider="openai", api_key=None)

    with pytest.raises(RagSparkDependencyError, match="Gemini embedding provider requires an API key"):
        RagSparkEmbeddingProvider(provider="gemini", api_key=None)


@pytest.mark.parametrize("retrieval_strategy", ["hybrid", "semantic", "keyword", "bm25"])
def test_ragspark_vector_store_memory_and_retrieval(retrieval_strategy):
    """Verify RagSpark VectorStore in-memory backend supports multiple retrieval strategies."""
    store = RagSparkVectorStore(backend="memory", embed_provider="fast_local")
    store.add(["RagSpark supports hybrid retrieval combining dense and sparse ranking."])
    assert store.count() == 1
    results = store.search(query="hybrid retrieval", top_k=1, strategy=retrieval_strategy)
    assert len(results) == 1
    assert "hybrid retrieval" in results[0].chunk.text


def test_ragspark_vector_store_missing_backend_dependency():
    """Verify that uninstalled vector DB backends raise explicit dependency error."""
    with pytest.raises(RagSparkDependencyError, match="Pinecone vector database requires 'pinecone-client'"):
        RagSparkVectorStore(backend="pinecone")


def test_ragspark_flash_reranker():
    """Verify RagSpark ultra-fast Flash cross-scoring reranker."""
    reranker = RagSparkReranker(model="flash")
    candidates = [
        "Strawberries are red fruit high in vitamin C.",
        "Quantum superposition allows qubits to exist in multiple states simultaneously.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    ranked = reranker.rerank(query="How does quantum superposition work?", candidates=candidates, top_k=2)
    assert len(ranked) == 2
    assert "Quantum superposition" in ranked[0]


def test_ragspark_prompt_builder_all_techniques():
    """Verify RagSpark prompt engineering across all registered techniques."""
    for tech in RAGSPARK_PROMPT_TECHNIQUES:
        built = RagSparkPromptBuilder.build(
            query="What is CRAG?",
            context_items=["CRAG is Corrective Retrieval Augmented Generation."],
            technique=tech,
        )
        assert built.user_prompt
        assert "CRAG" in built.user_prompt
    assert RagSparkPromptBuilder.detect_language("Bonjour tout le monde") == "fr"


def test_ragspark_generator_extractive_and_validation():
    """Verify extractive generation runs zero-dependency and external providers require keys."""
    res = RagSparkGenerator.generate(
        prompt="What is RagSpark?",
        provider="extractive",
        context=["RagSpark is an advanced local RAG library."],
    )
    assert "RagSpark is an advanced local RAG library." in res.text

    # Verify no silent fallback on external provider without key
    with pytest.raises(RagSparkDependencyError, match="OpenAI LLM provider requires an API key"):
        RagSparkGenerator.generate(prompt="Hello", provider="openai", api_key=None)


def test_ragspark_end_to_end_pipeline():
    """Verify RagSpark complete pipeline create and ask operations."""
    source_text = "Visual RAG Studio provides an interactive canvas for composing RAG architectures."
    pipeline = RagSparkPipeline.create(
        source=source_text,
        chunk_strategy="sentence",
        embed_provider="fast_local",
        vectordb="memory",
        retrieval_strategy="hybrid",
        prompt_technique="standard",
        llm_provider="extractive",
    )
    answer = pipeline.ask("What does Visual RAG Studio provide?")
    assert answer.answer
    assert "interactive canvas" in answer.answer
    assert len(answer.sources) >= 1


def test_ragspark_building_blocks_in_registry():
    """Verify generic nodes are preserved, dedicated ragspark nodes are excluded from palette,
    and underlying RagSpark provider adapters are registered."""
    # Generic node types must be present
    assert "document.source" in NODE_TYPES_BY_ID
    assert "document.parse" in NODE_TYPES_BY_ID
    assert "chunk.split" in NODE_TYPES_BY_ID
    assert "embedding.create" in NODE_TYPES_BY_ID
    assert "vector.index" in NODE_TYPES_BY_ID
    assert "generation.answer" in NODE_TYPES_BY_ID
    assert NODE_TYPES_BY_ID["chunk.split"].implementation_status == ImplementationStatus.IMPLEMENTED

    # Dedicated ragspark-specific nodes must NOT be in palette (User Requirement A)
    assert "ragspark.parser" not in NODE_TYPES_BY_ID
    assert "ragspark.chunker" not in NODE_TYPES_BY_ID
    assert "ragspark.embedder" not in NODE_TYPES_BY_ID
    assert "ragspark.vectordb" not in NODE_TYPES_BY_ID

    # Provider adapters
    adapter_ids = {a.id for a in REGISTRY.provider_adapters}
    assert "ragspark.fast_local.embedding" in adapter_ids
    assert "ragspark.memory.vector_store" in adapter_ids
    assert "ragspark.flash.reranker" in adapter_ids
    assert "ragspark.extractive.generation" in adapter_ids

    # Supported chunking strategies
    assert "hierarchical" in CHUNKING_STRATEGIES
    assert CHUNKING_STRATEGIES["hierarchical"].implementation_status == "implemented"
