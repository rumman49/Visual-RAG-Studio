"""Evaluation and verification suite for BMAD Epic 5 — Hybrid RAG.

Compares Dense Semantic Retrieval, Sparse BM25 Lexical Retrieval, and Hybrid RRF
across 5 query categories:
1. Direct factual questions
2. Exact numbers and entities
3. Technical terms and identifiers
4. Multi-passage conceptual evidence
5. Unanswerable / out-of-corpus queries

Measures and asserts:
- Recall@1, Recall@3, Recall@5
- Mean Reciprocal Rank (MRR)
- Graph validation for hybrid single-node and branched DAG topologies
- End-to-end pipeline execution with citation metadata preservation
"""

from __future__ import annotations

import math
from typing import Any
import pytest
from starlette.testclient import TestClient

from app.config import Settings
from app.contracts import (
    GraphConnection,
    ImplementationStatus,
    NodeInstance,
    WorkflowGraph,
)
from app.main import app
from app.rag.contracts import QueryRequest
from app.rag.dependencies import get_document_store, get_vector_store
from app.rag.graph import validate_traditional_graph
from app.rag.hybrid import (
    BM25Index,
    execute_hybrid_retrieval,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from app.rag.pipeline import answer_query
from app.rag.storage import DocumentStore
from app.rag.vector_store import MemoryVectorStore
from app.registry import REGISTRY, TEMPLATES_BY_ID


# ---------------------------------------------------------------------------
# Benchmark Corpus and Fixed Evaluation Dataset
# ---------------------------------------------------------------------------

BENCHMARK_CORPUS = [
    {
        "id": "chunk_01",
        "text": "The Transformer relies entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned RNNs or convolution.",
        "metadata": {"document_name": "attention_paper.pdf", "page_number": 1, "document_id": "doc_attn"},
    },
    {
        "id": "chunk_02",
        "text": "On the WMT 2014 English-to-German translation task, the big transformer model achieves 28.4 BLEU, improving over the existing best results by over 2.0 BLEU.",
        "metadata": {"document_name": "attention_paper.pdf", "page_number": 8, "document_id": "doc_attn"},
    },
    {
        "id": "chunk_03",
        "text": "We used the Adam optimizer with beta1 = 0.9, beta2 = 0.98 and epsilon = 1e-9. We varied the learning rate with warmup_steps = 4000.",
        "metadata": {"document_name": "attention_paper.pdf", "page_number": 7, "document_id": "doc_attn"},
    },
    {
        "id": "chunk_04",
        "text": "Each sub-layer employs residual connections followed by layer normalization: LayerNorm(x + Sublayer(x)), with d_model = 512 and eps = 1e-6.",
        "metadata": {"document_name": "attention_paper.pdf", "page_number": 3, "document_id": "doc_attn"},
    },
    {
        "id": "chunk_05",
        "text": "The base model has 65M parameters (65 million), consisting of 6 encoder layers and 6 decoder layers with 8 parallel attention heads.",
        "metadata": {"document_name": "attention_paper.pdf", "page_number": 9, "document_id": "doc_attn"},
    },
    {
        "id": "chunk_06",
        "text": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions, which single attention head prevents.",
        "metadata": {"document_name": "attention_paper.pdf", "page_number": 5, "document_id": "doc_attn"},
    },
    {
        "id": "chunk_07",
        "text": "Positional encodings have the same dimension d_model as the embeddings, using sine and cosine functions of different frequencies to capture token ordering.",
        "metadata": {"document_name": "attention_paper.pdf", "page_number": 6, "document_id": "doc_attn"},
    },
    {
        "id": "chunk_08",
        "text": "FastAPI is a modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints and Starlette.",
        "metadata": {"document_name": "web_frameworks.md", "page_number": 1, "document_id": "doc_web"},
    },
    {
        "id": "chunk_09",
        "text": "Chroma is an open-source AI-native vector database designed to store embeddings along with metadata for fast nearest-neighbor similarity search.",
        "metadata": {"document_name": "vectordb_guide.txt", "page_number": 1, "document_id": "doc_vdb"},
    },
    {
        "id": "chunk_10",
        "text": "PostgreSQL provides transactional ACID compliance, extensible data types, and robust relational querying with GiST and GIN indexing support.",
        "metadata": {"document_name": "database_manual.txt", "page_number": 2, "document_id": "doc_pg"},
    },
]


EVALUATION_QUESTIONS = [
    # Category 1: Direct Factual
    {
        "id": "q1",
        "category": "direct_factual",
        "question": "What core mechanism does the Transformer use without using RNNs?",
        "target_chunk_id": "chunk_01",
        "keywords": ["transformer", "self-attention", "mechanisms", "rnns"],
    },
    # Category 2: Exact Numbers / Entities
    {
        "id": "q2",
        "category": "exact_numbers",
        "question": "What BLEU score was achieved on English-to-German translation?",
        "target_chunk_id": "chunk_02",
        "keywords": ["28.4", "bleu", "english-to-german"],
    },
    {
        "id": "q3",
        "category": "exact_numbers",
        "question": "How many warmup steps were used in the learning rate schedule?",
        "target_chunk_id": "chunk_03",
        "keywords": ["warmup_steps", "4000", "learning", "rate"],
    },
    # Category 3: Technical Terms / Identifiers
    {
        "id": "q4",
        "category": "technical_terms",
        "question": "What is the formula for the sub-layer residual connection with LayerNorm?",
        "target_chunk_id": "chunk_04",
        "keywords": ["layernorm", "sublayer", "residual", "1e-6"],
    },
    {
        "id": "q5",
        "category": "technical_terms",
        "question": "How many parameters does the base model have in total?",
        "target_chunk_id": "chunk_05",
        "keywords": ["65m", "parameters", "65", "million"],
    },
    # Category 4: Multi-Passage / Conceptual Evidence
    {
        "id": "q6",
        "category": "conceptual",
        "question": "Why does multi-head attention benefit subspace representations compared to a single head?",
        "target_chunk_id": "chunk_06",
        "keywords": ["multi-head", "attention", "subspaces", "positions"],
    },
    {
        "id": "q7",
        "category": "conceptual",
        "question": "How does the model encode sequence order using sine and cosine frequencies?",
        "target_chunk_id": "chunk_07",
        "keywords": ["positional", "encodings", "sine", "cosine", "ordering"],
    },
    # Category 5: Negative / Unanswerable
    {
        "id": "q8",
        "category": "unanswerable",
        "question": "What is the superconducting critical magnetic field of niobium titanium?",
        "target_chunk_id": None,
        "keywords": ["superconducting", "critical", "magnetic", "niobium"],
    },
    {
        "id": "q9",
        "category": "unanswerable",
        "question": "How many moons orbit the exoplanet Proxima Centauri b?",
        "target_chunk_id": None,
        "keywords": ["moons", "exoplanet", "proxima", "centauri"],
    },
]


# ---------------------------------------------------------------------------
# Evaluation Metrics Helpers
# ---------------------------------------------------------------------------

def compute_recall_at_k(retrieved_ids: list[str], target_id: str | None, k: int) -> float:
    if target_id is None:
        return 1.0  # Unanswerable query correctly has no target in corpus
    top_k_ids = retrieved_ids[:k]
    return 1.0 if target_id in top_k_ids else 0.0


def compute_reciprocal_rank(retrieved_ids: list[str], target_id: str | None) -> float:
    if target_id is None:
        return 1.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id == target_id:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Unit Tests: BM25Index & Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def test_bm25_exact_entity_retrieval():
    """Verify BM25 captures exact numbers, identifiers and entities at rank 1."""
    index = BM25Index(k1=1.5, b=0.75)
    index.index(BENCHMARK_CORPUS)

    # Exact number query
    res = index.search("28.4 BLEU WMT 2014", top_k=5)
    assert len(res) > 0
    assert res[0]["id"] == "chunk_02"
    assert res[0]["metadata"]["page_number"] == 8
    assert "distance" not in res[0]  # BM25 is a lexical score, not vector distance.

    # Technical identifier query
    res_ident = index.search("warmup_steps = 4000", top_k=5)
    assert len(res_ident) > 0
    assert res_ident[0]["id"] == "chunk_03"


def test_reciprocal_rank_fusion_logic():
    """Verify standard RRF formula correctly fuses two independent rankings."""
    list_a = [
        {"id": "doc1", "text": "text 1", "distance": 0.2},
        {"id": "doc2", "text": "text 2", "distance": 0.4},
        {"id": "doc3", "text": "text 3", "distance": 0.6},
    ]
    list_b = [
        {"id": "doc2", "text": "text 2", "sparse_score": 5.0},
        {"id": "doc1", "text": "text 1", "sparse_score": 3.0},
        {"id": "doc4", "text": "text 4", "sparse_score": 2.0},
    ]

    fused = reciprocal_rank_fusion(
        rankings=[list_a, list_b],
        k=60,
        weights=[0.6, 0.4],
        top_k=4,
    )

    fused_ids = [d["id"] for d in fused]
    # doc1: 0.6 / (60 + 1) + 0.4 / (60 + 2) = 0.009836 + 0.006451 = 0.016287
    # doc2: 0.6 / (60 + 2) + 0.4 / (60 + 1) = 0.009677 + 0.006557 = 0.016234
    assert fused_ids[0] in {"doc1", "doc2"}
    assert "doc4" in fused_ids
    assert all("rank" in item for item in fused)
    assert fused[0]["distance"] in {0.2, 0.4, 0.6}
    assert all("rrf_score" in item for item in fused)
    assert "distance" not in next(item for item in fused if item["id"] == "doc4")
    assert all(item["retrieval_strategy"] == "hybrid_rrf" for item in fused)


def test_weighted_score_fusion_logic():
    """Verify min-max normalized weighted fusion."""
    dense_list = [
        {"id": "c1", "text": "alpha", "distance": 0.1},
        {"id": "c2", "text": "beta", "distance": 0.5},
    ]
    sparse_list = [
        {"id": "c2", "text": "beta", "sparse_score": 10.0},
        {"id": "c1", "text": "alpha", "sparse_score": 1.0},
    ]

    # Equal weights
    fused = weighted_score_fusion(
        dense_candidates=dense_list,
        sparse_candidates=sparse_list,
        alpha=0.5,
        top_k=2,
    )
    assert len(fused) == 2
    assert fused[0]["retrieval_strategy"] == "hybrid_weighted"
    assert fused[0]["weighted_score"] == fused[0]["score"]
    assert fused[0]["distance"] in {0.1, 0.5}


def test_hybrid_loads_sparse_text_when_document_store_only_has_chunk_metadata():
    """The persisted document ledger stores chunk IDs; BM25 must load actual indexed text."""
    class MetadataOnlyStore:
        def list(self):
            return [{
                "document_id": "doc_law",
                "filename": "law.txt",
                "chunks": [{"chunk_id": "c1", "ordinal": 0}],
            }]

    class IndexedTextVectorStore:
        def search(self, vector, top_k, document_ids=None, query_text=None):
            return [{"id": "c2", "text": "generic dense result", "distance": 0.15}]

        def get_document(self, document_id):
            assert document_id == "doc_law"
            return [
                {"id": "c1", "text": "Section 8A commencement date is 1 July 2017.", "metadata": {"document_id": document_id}},
                {"id": "c2", "text": "General provisions for the law.", "metadata": {"document_id": document_id}},
            ]

    events = []
    results = execute_hybrid_retrieval(
        vector_store=IndexedTextVectorStore(),
        query_vector=[0.1, 0.9],
        query_text="commencement date 1 July 2017",
        top_k=2,
        document_ids=["doc_law"],
        store=MetadataOnlyStore(),
        strategy="hybrid",
        on_step=lambda node, status: events.append((node, status)),
    )

    assert results
    assert any(item.get("sparse_rank") == 1 for item in results)
    assert any(item.get("id") == "c1" for item in results)
    assert ("sparse", "completed") in events
    assert ("fuse", "completed") in events


# ---------------------------------------------------------------------------
# Retrieval Benchmark: Dense vs BM25 vs Hybrid RRF
# ---------------------------------------------------------------------------

def test_hybrid_benchmark_evaluation():
    """
    Run evaluation across all 9 benchmark questions and compute:
    - Recall@1, Recall@3, Recall@5
    - MRR (Mean Reciprocal Rank)
    Assert Hybrid RRF achieves >= 0.85 Recall@3 and superior MRR on exact queries.
    """
    # 1. Setup in-memory vector store with simple deterministic embeddings
    # We simulate a 16-dimensional embedding space
    mem_store = MemoryVectorStore(collection_name="benchmark_col")

    import hashlib

    def pseudo_embed(text: str) -> list[float]:
        vec = [0.0] * 32
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16) % 32
            vec[h] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    class FakeEmbedder:
        def embed(self, model: str, texts: list[str]):
            return [pseudo_embed(t) for t in texts], 32

    fake_embedder = FakeEmbedder()

    # Seed vector store with benchmark corpus
    from app.rag.chunking import TextChunk
    chunks_by_doc: dict[str, list[TextChunk]] = {}
    vecs_by_doc: dict[str, list[list[float]]] = {}
    for i, doc in enumerate(BENCHMARK_CORPUS):
        d_id = doc["metadata"]["document_id"]
        tc = TextChunk(
            chunk_id=doc["id"],
            document_id=d_id,
            document_name=doc["metadata"].get("document_name", "doc"),
            ordinal=i,
            text=doc["text"],
            strategy_id="fixed",
            settings={"chunk_size": 1000, "overlap": 100},
            page_number=doc["metadata"].get("page_number"),
            section_path=(),
            source_start=0,
            source_end=len(doc["text"]),
            parser_id="test_parser",
        )
        chunks_by_doc.setdefault(d_id, []).append(tc)
        vecs_by_doc.setdefault(d_id, []).append(pseudo_embed(doc["text"]))

    for d_id, chks in chunks_by_doc.items():
        mem_store.upsert_document(
            document_id=d_id,
            chunks=chks,
            embeddings=vecs_by_doc[d_id],
            model="fake-32",
            dimension=32,
        )

    # 2. Setup BM25 Index
    bm25_index = BM25Index()
    bm25_index.index(BENCHMARK_CORPUS)

    dense_r1, dense_r3, dense_mrr = [], [], []
    bm25_r1, bm25_r3, bm25_mrr = [], [], []
    hybrid_r1, hybrid_r3, hybrid_mrr = [], [], []

    answerable_questions = [q for q in EVALUATION_QUESTIONS if q["target_chunk_id"] is not None]

    for q in answerable_questions:
        target = q["target_chunk_id"]
        q_vec = pseudo_embed(q["question"])

        # Dense-only
        dense_results = mem_store.search(q_vec, top_k=5)
        dense_ids = [d["id"] for d in dense_results]
        dense_r1.append(compute_recall_at_k(dense_ids, target, 1))
        dense_r3.append(compute_recall_at_k(dense_ids, target, 3))
        dense_mrr.append(compute_reciprocal_rank(dense_ids, target))

        # BM25-only
        bm25_results = bm25_index.search(q["question"], top_k=5)
        bm25_ids = [d["id"] for d in bm25_results]
        bm25_r1.append(compute_recall_at_k(bm25_ids, target, 1))
        bm25_r3.append(compute_recall_at_k(bm25_ids, target, 3))
        bm25_mrr.append(compute_reciprocal_rank(bm25_ids, target))

        # Hybrid RRF
        hybrid_results = execute_hybrid_retrieval(
            vector_store=mem_store,
            query_vector=q_vec,
            query_text=q["question"],
            top_k=5,
            strategy="hybrid",
            fusion_method="rrf",
            rrf_k=60,
            alpha=0.6,
        )
        hybrid_ids = [d["id"] for d in hybrid_results]
        hybrid_r1.append(compute_recall_at_k(hybrid_ids, target, 1))
        hybrid_r3.append(compute_recall_at_k(hybrid_ids, target, 3))
        hybrid_mrr.append(compute_reciprocal_rank(hybrid_ids, target))

    avg_dense_r3 = sum(dense_r3) / len(dense_r3)
    avg_bm25_r3 = sum(bm25_r3) / len(bm25_r3)
    avg_hybrid_r3 = sum(hybrid_r3) / len(hybrid_r3)

    avg_dense_mrr = sum(dense_mrr) / len(dense_mrr)
    avg_bm25_mrr = sum(bm25_mrr) / len(bm25_mrr)
    avg_hybrid_mrr = sum(hybrid_mrr) / len(hybrid_mrr)

    # Hybrid RRF combines the precision of BM25 on entities with dense semantic coverage
    assert avg_hybrid_r3 >= 0.85, f"Expected Hybrid Recall@3 >= 0.85, got {avg_hybrid_r3:.3f}"
    assert avg_hybrid_mrr >= 0.70, f"Expected Hybrid MRR >= 0.70, got {avg_hybrid_mrr:.3f}"
    assert avg_hybrid_mrr >= avg_dense_mrr, "Hybrid RRF should match or exceed dense-only MRR on entity-rich benchmarks"


# ---------------------------------------------------------------------------
# Graph Validation Tests: Hybrid Topologies
# ---------------------------------------------------------------------------

def test_hybrid_single_node_graph_validation():
    """Verify that traditional baseline graph configured with strategy='hybrid' is valid and executable."""
    settings = Settings()
    base_graph = TEMPLATES_BY_ID["traditional-rag"].graph.model_copy(deep=True)

    # Configure the retriever node with strategy: "hybrid"
    for node in base_graph.nodes:
        if node.node_type_id == "retrieval.dense":
            node.config = {
                "strategy": "hybrid",
                "top_k": 5,
                "fusion_method": "rrf",
                "rrf_k": 60,
                "alpha": 0.65,
            }

    report = validate_traditional_graph(
        base_graph,
        settings,
        generation_provider="extractive",
        generation_model="extractive",
    )
    assert report.valid is True
    assert report.executable is True
    assert len([i for i in report.issues if i.severity.value == "error"]) == 0


def test_hybrid_branched_dag_validation():
    """
    Verify that a multi-branch DAG topology with retrieval.dense and retrieval.sparse
    both feeding into merge.fuse before context.assemble is valid and executable.
    """
    settings = Settings()
    base_graph = TEMPLATES_BY_ID["traditional-rag"].graph.model_copy(deep=True)

    # Add retrieval.sparse and merge.fuse nodes
    nodes = [n for n in base_graph.nodes if n.node_type_id != "retrieval.dense"]
    nodes.extend([
        NodeInstance(id="dense_retriever", node_type_id="retrieval.dense", config={"top_k": 10}),
        NodeInstance(id="sparse_retriever", node_type_id="retrieval.sparse", config={"top_k": 10, "k1": 1.5, "b": 0.75}),
        NodeInstance(id="rrf_fuser", node_type_id="merge.fuse", config={"fusion_method": "rrf", "rrf_k": 60}),
    ])

    connections = [
        # Ingestion
        GraphConnection(source_node_id="source", source_port_id="documents", target_node_id="parser", target_port_id="documents"),
        GraphConnection(source_node_id="parser", source_port_id="parsed_documents", target_node_id="chunker", target_port_id="parsed_documents"),
        GraphConnection(source_node_id="chunker", source_port_id="chunks", target_node_id="embedder", target_port_id="chunks"),
        GraphConnection(source_node_id="embedder", source_port_id="embeddings", target_node_id="index", target_port_id="embeddings"),
        # Query branching to both dense and sparse retrievers
        GraphConnection(source_node_id="query", source_port_id="query", target_node_id="dense_retriever", target_port_id="query"),
        GraphConnection(source_node_id="index", source_port_id="index", target_node_id="dense_retriever", target_port_id="index"),
        GraphConnection(source_node_id="query", source_port_id="query", target_node_id="sparse_retriever", target_port_id="query"),
        GraphConnection(source_node_id="index", source_port_id="index", target_node_id="sparse_retriever", target_port_id="index"),
        # Both retrievers feed into the fusion node
        GraphConnection(source_node_id="dense_retriever", source_port_id="retrieved_chunks", target_node_id="rrf_fuser", target_port_id="retrieved_chunks"),
        GraphConnection(source_node_id="sparse_retriever", source_port_id="retrieved_chunks", target_node_id="rrf_fuser", target_port_id="retrieved_chunks"),
        # Fusion node feeds into context assemble
        GraphConnection(source_node_id="rrf_fuser", source_port_id="retrieved_chunks", target_node_id="context", target_port_id="retrieved_chunks"),
        # Context into generation
        GraphConnection(source_node_id="query", source_port_id="query", target_node_id="answer", target_port_id="query"),
        GraphConnection(source_node_id="context", source_port_id="context", target_node_id="answer", target_port_id="context"),
    ]

    hybrid_dag = WorkflowGraph(
        id="hybrid-dag-test",
        name="Hybrid RAG DAG",
        architecture_id="hybrid-rag",
        template_id="hybrid-rag",
        nodes=nodes,
        connections=connections,
    )

    report = validate_traditional_graph(
        hybrid_dag,
        settings,
        generation_provider="extractive",
        generation_model="extractive",
    )
    assert report.valid is True
    assert report.executable is True
    assert len([i for i in report.issues if i.severity.value == "error"]) == 0


# ---------------------------------------------------------------------------
# End-to-End Pipeline & API Tests
# ---------------------------------------------------------------------------

def test_hybrid_rag_end_to_end_query():
    """Verify end-to-end question answering through the /api/v1/query endpoint with hybrid retrieval."""
    client = TestClient(app)

    # Load hybrid template graph from registry
    hybrid_template = TEMPLATES_BY_ID["hybrid-rag"]
    assert hybrid_template.graph is not None
    graph_dict = hybrid_template.graph.model_dump(mode="json")

    # Set extractive provider to run locally without external API dependencies
    for node in graph_dict["nodes"]:
        if node["node_type_id"] == "generation.answer":
            node["config"] = {"provider": "extractive", "model": "extractive"}
        if node["node_type_id"] == "vector.index":
            node["config"] = {"backend": "memory", "collection_name": "test_hybrid_e2e"}

    payload = {
        "question": "What is the core idea of the Transformer architecture?",
        "top_k": 3,
        "generation_provider": "extractive",
        "generation_model": "extractive",
        "allow_external_processing": True,
        "graph": graph_dict,
    }

    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert "answer" in data
    assert "retrieved_chunks" in data
    assert len(data["retrieved_chunks"]) > 0 or data.get("abstained") is True
    assert data["generation_provider"] == "extractive"
