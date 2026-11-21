"""Real Chroma integration tests verifying retrieval correctness and document scope isolation."""

import pytest

from app.config import get_settings
from app.rag.agent import AgentApprovalPolicy, AgentExecutionMode, AgentOrchestrator
from app.rag.hybrid import execute_hybrid_retrieval
from app.rag.vector_store import ChromaVectorStore


def dummy_embed(text: str) -> list[float]:
    """Deterministically map text to a 4-dimensional normalized vector."""
    t = text.lower()
    v = [
        1.0 if any(k in t for k in ["quantum", "physics", "particle"]) else 0.05,
        1.0 if any(k in t for k in ["plant", "biology", "photosynthesis"]) else 0.05,
        1.0 if any(k in t for k in ["star", "astronomy", "supernova"]) else 0.05,
        1.0 if any(k in t for k in ["law", "tax", "statute", "provisional"]) else 0.05,
    ]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


class MockEmbeddingProvider:
    backend_name = "mock"

    def embed(self, model: str, texts: list[str]) -> tuple[list[list[float]], int]:
        return [dummy_embed(t) for t in texts], 10


class MockGenerationProvider:
    provider_name = "extractive"

    def generate(self, model: str | None, system_prompt: str, prompt: str, **kwargs) -> str:
        return "Grounded response based on retrieved evidence [1]."


@pytest.fixture
def populated_chroma(tmp_path):
    """Create a real Chroma database with 8 distinct, content-isolated documents."""
    vstore = ChromaVectorStore(tmp_path / "chroma_test_isolation", collection_name="test_isolation")
    vstore.collection.modify(metadata={"embedding_dimension": 4})

    documents_data = [
        ("doc-physics", "Quantum entanglement occurs when pairs or groups of particles interact.", "Physics Guide.pdf", 1),
        ("doc-biology", "Photosynthesis in green plants converts solar light energy into chemical energy.", "Biology Basics.pdf", 2),
        ("doc-astronomy", "Supernova explosions occur when massive stars exhaust their nuclear core fuel.", "Astrophysics.pdf", 3),
        ("doc-chemistry", "Covalent chemical bonds involve sharing electron pairs between atoms.", "Chemistry 101.pdf", 4),
        ("doc-geology", "Plate tectonics describes large-scale motions of Earth's lithospheric plates.", "Geology Earth.pdf", 5),
        ("doc-comp-sci", "Turing completeness defines a computational model capable of simulating algorithms.", "CS Theory.pdf", 6),
        ("doc-history", "The Industrial Revolution began in Great Britain during the late eighteenth century.", "World History.pdf", 7),
        ("doc-law", "The Provisional Collection of Taxes Act gives statutory authority for immediate customs duties.", "Tax Law.pdf", 8),
    ]

    for doc_id, text, fname, page in documents_data:
        vec = dummy_embed(text)
        vstore.collection.add(
            ids=[f"{doc_id}-chunk-1"],
            documents=[text],
            metadatas=[{
                "document_id": doc_id,
                "document_name": fname,
                "page_number": page,
                "section_path": "Root / Section",
                "ordinal": 0,
            }],
            embeddings=[vec],
        )
    return vstore


def test_search_all_documents_searches_complete_collection(populated_chroma):
    """Searching all indexed documents (document_ids=None) queries the full collection."""
    q_vec = dummy_embed("quantum particle physics")
    results = populated_chroma.search(q_vec, top_k=5, document_ids=None)
    assert len(results) > 0
    # Top result should be the physics document
    assert results[0]["metadata"]["document_id"] == "doc-physics"


def test_search_single_document_is_strictly_isolated(populated_chroma):
    """Searching one document returns only chunks from that document, ignoring stronger matches elsewhere."""
    # Even if query is about quantum physics, restrict search to doc-biology
    q_vec = dummy_embed("quantum physics particle")
    results = populated_chroma.search(q_vec, top_k=5, document_ids=["doc-biology"])
    assert len(results) == 1
    assert results[0]["metadata"]["document_id"] == "doc-biology"
    assert "doc-physics" not in [r["metadata"]["document_id"] for r in results]


def test_search_selected_subset_less_than_or_equal_five(populated_chroma):
    """Searching 2-5 documents returns only chunks from the selected subset."""
    q_vec = dummy_embed("energy in stars and plants")
    selected = ["doc-biology", "doc-astronomy"]
    results = populated_chroma.search(q_vec, top_k=5, document_ids=selected)
    assert len(results) == 2
    returned_ids = {r["metadata"]["document_id"] for r in results}
    assert returned_ids == set(selected)
    assert not returned_ids.intersection({"doc-physics", "doc-law", "doc-history", "doc-comp-sci", "doc-geology", "doc-chemistry"})


def test_search_more_than_five_documents_does_not_remove_filter(populated_chroma):
    """Selecting > 5 documents MUST NOT remove the document filter or leak unselected documents."""
    # Select 6 documents out of 8; deliberately omit doc-law and doc-history
    selected_six = [
        "doc-physics",
        "doc-biology",
        "doc-astronomy",
        "doc-chemistry",
        "doc-geology",
        "doc-comp-sci",
    ]
    # Query is explicitly about tax law, but doc-law is NOT in selected_six
    q_vec = dummy_embed("provisional collection of taxes statute law")
    results = populated_chroma.search(q_vec, top_k=5, document_ids=selected_six)

    assert len(results) > 0
    for r in results:
        doc_id = r["metadata"]["document_id"]
        assert doc_id in selected_six, f"Unrelated document {doc_id} leaked into results!"
        assert doc_id != "doc-law", "Unselected document 'doc-law' leaked despite query similarity!"
        assert doc_id != "doc-history", "Unselected document 'doc-history' leaked!"


def test_hybrid_retrieval_and_fusion_integrity(populated_chroma):
    """Hybrid retrieval executes dense + sparse and fuses with RRF on selected documents."""
    q_vec = dummy_embed("quantum particles")
    selected = ["doc-physics", "doc-chemistry"]

    fused = execute_hybrid_retrieval(
        vector_store=populated_chroma,
        query_vector=q_vec,
        query_text="quantum particles",
        top_k=2,
        document_ids=selected,
        strategy="hybrid",
        fusion_method="rrf",
    )
    assert len(fused) > 0
    # Must report actual strategy as hybrid_rrf
    assert fused[0]["retrieval_strategy"] == "hybrid_rrf"
    assert "dense_rank" in fused[0]
    assert "sparse_rank" in fused[0]
    assert "rrf_score" in fused[0]
    # Scope isolation preserved
    for item in fused:
        assert item["metadata"]["document_id"] in selected


def test_hybrid_dense_fallback_is_explicitly_identified(populated_chroma):
    """Broad collection queries without on-the-fly sparse index explicitly identify fallback."""
    q_vec = dummy_embed("supernova stars")
    # document_ids=None triggers dense fallback for unindexed broad collection
    results = execute_hybrid_retrieval(
        vector_store=populated_chroma,
        query_vector=q_vec,
        query_text="supernova stars",
        top_k=3,
        document_ids=None,
        strategy="hybrid",
    )
    assert len(results) > 0
    # Must NOT silently label as hybrid
    assert results[0]["retrieval_strategy"] == "dense_fallback"
    assert results[0]["requested_strategy"] == "hybrid"
    assert results[0]["fallback_reason"] is not None
    assert "broad_collection" in results[0]["fallback_reason"]


def test_agent_orchestrator_preserves_scope_isolation(populated_chroma):
    """Agent execution preserves document scope in all retrieved candidates and citations."""
    selected_subset = ["doc-biology", "doc-astronomy"]
    orchestrator = AgentOrchestrator(
        goal="Answer accurately.",
        mode=AgentExecutionMode.BOUNDED_LOOP,
        max_steps=3,
        approval_policy=AgentApprovalPolicy.ON_UNCERTAINTY,
        allowed_tools=["retrieval", "evaluate_evidence", "answer_generation"],
    )

    class MockStore:
        def list(self):
            return [
                {"document_id": "doc-biology", "status": "ready"},
                {"document_id": "doc-astronomy", "status": "ready"},
            ]

    res = orchestrator.run(
        query="Explain photosynthesis in green plants",
        vector_store=populated_chroma,
        provider=MockEmbeddingProvider(),
        settings=get_settings(),
        store=MockStore(),
        generation_provider=MockGenerationProvider(),
        generation_model="mock",
        document_ids=selected_subset,
        initial_strategy="hybrid",
    )

    assert res["status"] == "completed"
    assert len(res["citations"]) > 0
    for cit in res["citations"]:
        assert cit["document_id"] in selected_subset, f"Agent leaked citation from {cit['document_id']}"
        assert cit["document_id"] != "doc-physics"
