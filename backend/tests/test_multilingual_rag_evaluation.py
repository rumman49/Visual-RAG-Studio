"""Comprehensive evaluation test suite for Multilingual RAG in Visual RAG Studio.

Covers:
1. Language detection (Hindi Devanagari, Hinglish Roman, English, European languages).
2. Multilingual abstention detection and localized messages.
3. Quantitative retrieval evaluation: Recall@1, Recall@3, Recall@5, and MRR
   across English, Hindi, Hinglish, and Spanish with cross-lingual queries.
4. Source language and metadata preservation in retrieved context.
5. Grounded answer generation and citation integrity.
6. Localized abstention on unanswerable questions.
7. Validation and execution on both starter template and custom user-created graphs.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.rag.chunking import TextChunk
from app.rag.graph import validate_traditional_graph
from app.rag.multilingual import (
    DEFAULT_ABSTENTION,
    LOCALIZED_ABSTENTIONS,
    SUPPORTED_LANGUAGES,
    build_multilingual_instructions,
    detect_language,
    get_multilingual_abstention,
    is_multilingual_abstention,
)
from app.rag.pipeline import answer_query, assemble_context
from app.rag.providers import SentenceTransformersProvider
from app.rag.storage import DocumentRow, DocumentStore
from app.rag.vector_store import MemoryVectorStore
from app.registry import REGISTRY, TEMPLATES_BY_ID, NodeInstance, WorkflowGraph, GraphConnection


# ---------------------------------------------------------------------------
# 1. Language Detection Tests
# ---------------------------------------------------------------------------

def test_detect_language_hindi_devanagari():
    samples = [
        "सौर ऊर्जा सूर्य से प्राप्त होने वाली प्रकाश और ऊष्मा ऊर्जा है।",
        "कृत्रिम न्यूरल नेटवर्क को कैसे प्रशिक्षित किया जाता है?",
        "भारत में राष्ट्रीय सौर मिशन सौर ऊर्जा को बढ़ावा देता है।",
        "नमस्ते, आप कैसे हैं?",
    ]
    for sample in samples:
        assert detect_language(sample) == "hi", f"Failed to detect Hindi: {sample}"


def test_detect_language_hinglish():
    samples = [
        "Chandrayaan-3 mission me kaunsa rover shamil tha?",
        "Neural network ko train karne ke liye kaunsa algorithm use hota hai?",
        "Mujhe is document ke baare me jankari chahiye, bataiye.",
        "Aapka naam kya hai aur ye kaise kaam karta hai?",
    ]
    for sample in samples:
        assert detect_language(sample) == "hinglish", f"Failed to detect Hinglish: {sample}"


def test_detect_language_english_and_european():
    assert detect_language("What algorithm is used to train artificial neural networks?") == "en"
    assert detect_language("How does solar photovoltaic power generation work?") == "en"
    assert detect_language("¿Qué proceso biológico convierte la luz solar en energía química?") == "es"
    assert detect_language("Comment fonctionne l'apprentissage automatique dans les réseaux de neurones?") == "fr"
    assert detect_language("Wie funktioniert das Training von künstlichen neuronalen Netzen?") == "de"


# ---------------------------------------------------------------------------
# 2. Localized Abstention and System Instructions Tests
# ---------------------------------------------------------------------------

def test_localized_abstention_messages_and_detection():
    for lang, expected_msg in LOCALIZED_ABSTENTIONS.items():
        assert get_multilingual_abstention(lang) == expected_msg
        assert is_multilingual_abstention(expected_msg), f"Failed to recognize {lang} abstention"

    # Variations of abstentions
    assert is_multilingual_abstention("मुझे इस प्रश्न का उत्तर देने के लिए पर्याप्त साक्ष्य नहीं मिले।")
    assert is_multilingual_abstention("Mujhe is prashn ka uttar dene ke liye paryapt evidence nahi mila.")
    assert is_multilingual_abstention("I could not find enough retrieved evidence to answer that question.")
    assert is_multilingual_abstention("The provided documents do not contain information to answer this.")

    # Substantive grounded answers should NEVER be marked as abstentions
    substantive_en = "Backpropagation is used to compute gradients for training artificial neural networks [1]."
    substantive_hi = "सौर ऊर्जा सूर्य से प्राप्त होती है और भारत में राष्ट्रीय सौर मिशन इसे बढ़ावा देता है [1]।"
    substantive_hinglish = "Chandrayaan-3 mission me Vikram lander aur Pragyan rover shamil the [1]."
    assert not is_multilingual_abstention(substantive_en)
    assert not is_multilingual_abstention(substantive_hi)
    assert not is_multilingual_abstention(substantive_hinglish)


def test_build_multilingual_instructions():
    hi_instr, hi_lang = build_multilingual_instructions("hi", "सौर ऊर्जा क्या है?")
    assert hi_lang == "hi"
    assert "Hindi (हिंदी)" in hi_instr
    assert LOCALIZED_ABSTENTIONS["hi"] in hi_instr

    hinglish_instr, hinglish_lang = build_multilingual_instructions("hinglish", "Ye kaise kaam karta hai?")
    assert hinglish_lang == "hinglish"
    assert "Hinglish" in hinglish_instr
    assert LOCALIZED_ABSTENTIONS["hinglish"] in hinglish_instr

    auto_instr, auto_lang = build_multilingual_instructions("auto", "कृत्रिम न्यूरल नेटवर्क क्या है?")
    assert auto_lang == "hi"


# ---------------------------------------------------------------------------
# 3. Repeatable Multilingual Benchmark Dataset & Quantitative MRR Evaluation
# ---------------------------------------------------------------------------

BENCHMARK_DOCUMENTS = [
    {
        "doc_id": "doc_en",
        "filename": "neural_networks_en.txt",
        "chunks": [
            TextChunk(
                chunk_id="chunk_en_1",
                document_id="doc_en",
                document_name="neural_networks_en.txt",
                text=(
                    "Artificial Neural Networks (ANN) are computing systems inspired by biological neural networks. "
                    "Backpropagation is the primary supervised learning algorithm used to train neural networks "
                    "by calculating the gradient of the loss function with respect to the network weights."
                ),
                ordinal=1,
                strategy_id="recursive",
                settings={},
                page_number=1,
                section_path=(),
                source_start=0,
                source_end=200,
                parser_id="text",
            ),
            TextChunk(
                chunk_id="chunk_en_2",
                document_id="doc_en",
                document_name="neural_networks_en.txt",
                text=(
                    "Deep learning uses multiple layers to extract higher-level features from raw input. "
                    "Convolutional Neural Networks (CNN) excel at computer vision, while Transformer models "
                    "dominate modern natural language processing and multimodal systems."
                ),
                ordinal=2,
                strategy_id="recursive",
                settings={},
                page_number=2,
                section_path=(),
                source_start=201,
                source_end=400,
                parser_id="text",
            ),
        ],
    },
    {
        "doc_id": "doc_hi",
        "filename": "solar_energy_hi.txt",
        "chunks": [
            TextChunk(
                chunk_id="chunk_hi_1",
                document_id="doc_hi",
                document_name="solar_energy_hi.txt",
                text=(
                    "सौर ऊर्जा (Solar Energy) सूर्य से प्राप्त होने वाली प्रत्यक्ष प्रकाश और ऊष्मा ऊर्जा है। "
                    "भारत में राष्ट्रीय सौर मिशन (National Solar Mission) देश में सौर ऊर्जा उत्पादन और "
                    "नवीकरणीय ऊर्जा क्षमता को तेजी से बढ़ावा देने के लिए शुरू किया गया एक प्रमुख राष्ट्रीय कार्यक्रम है।"
                ),
                ordinal=1,
                strategy_id="recursive",
                settings={},
                page_number=1,
                section_path=(),
                source_start=0,
                source_end=200,
                parser_id="text",
            ),
            TextChunk(
                chunk_id="chunk_hi_2",
                document_id="doc_hi",
                document_name="solar_energy_hi.txt",
                text=(
                    "फोटोवोल्टिक सेल (Photovoltaic cells) सूर्य के प्रकाश को सीधे विद्युत ऊर्जा में बदलते हैं। "
                    "राजस्थान और गुजरात राज्यों में विशाल सोलर पार्क स्थापित किए गए हैं जो स्वच्छ बिजली प्रदान करते हैं।"
                ),
                ordinal=2,
                strategy_id="recursive",
                settings={},
                page_number=2,
                section_path=(),
                source_start=201,
                source_end=400,
                parser_id="text",
            ),
        ],
    },
    {
        "doc_id": "doc_hinglish",
        "filename": "chandrayaan_hinglish.txt",
        "chunks": [
            TextChunk(
                chunk_id="chunk_hinglish_1",
                document_id="doc_hinglish",
                document_name="chandrayaan_hinglish.txt",
                text=(
                    "Chandrayaan-3 Bharat ka teesra moon exploration mission tha jise ISRO ne launch kiya tha. "
                    "Is mission me Vikram lander aur Pragyan rover shamil the. August 2023 me isne lunar south pole "
                    "ke paas safalta se soft landing ki aur Bharat aisa karne wala pehla desh bana."
                ),
                ordinal=1,
                strategy_id="recursive",
                settings={},
                page_number=1,
                section_path=(),
                source_start=0,
                source_end=200,
                parser_id="text",
            ),
            TextChunk(
                chunk_id="chunk_hinglish_2",
                document_id="doc_hinglish",
                document_name="chandrayaan_hinglish.txt",
                text=(
                    "Pragyan rover ne chandrama ki satah par ghoom kar mineral composition aur elements jaise "
                    "sulphur aur oxygen ki presence confirm ki. Vikram lander ne lunar temperature record kiya."
                ),
                ordinal=2,
                strategy_id="recursive",
                settings={},
                page_number=2,
                section_path=(),
                source_start=201,
                source_end=400,
                parser_id="text",
            ),
        ],
    },
    {
        "doc_id": "doc_es",
        "filename": "fotosintesis_es.txt",
        "chunks": [
            TextChunk(
                chunk_id="chunk_es_1",
                document_id="doc_es",
                document_name="fotosintesis_es.txt",
                text=(
                    "La fotosíntesis es el proceso biológico fundamental mediante el cual las plantas, algas y "
                    "ciertas bacterias convierten la energía solar luminosa en energía química almacenada en azúcares. "
                    "Este proceso libera oxígeno vital a la atmósfera terrestre."
                ),
                ordinal=1,
                strategy_id="recursive",
                settings={},
                page_number=1,
                section_path=(),
                source_start=0,
                source_end=200,
                parser_id="text",
            ),
        ],
    },
]

# Evaluation query test suite covering monolingual and cross-lingual combinations
BENCHMARK_QUERIES = [
    # Monolingual English -> English doc
    {
        "query": "What algorithm is used to train artificial neural networks by computing loss gradients?",
        "target_doc_id": "doc_en",
        "target_chunk_id": "chunk_en_1",
        "category": "Monolingual EN",
    },
    # Monolingual Hindi -> Hindi doc
    {
        "query": "भारत में राष्ट्रीय सौर मिशन किस ऊर्जा को बढ़ावा देने के लिए शुरू किया गया है?",
        "target_doc_id": "doc_hi",
        "target_chunk_id": "chunk_hi_1",
        "category": "Monolingual HI",
    },
    # Monolingual Hinglish -> Hinglish doc
    {
        "query": "Chandrayaan-3 mission me kaunse lander aur rover shamil the?",
        "target_doc_id": "doc_hinglish",
        "target_chunk_id": "chunk_hinglish_1",
        "category": "Monolingual Hinglish",
    },
    # Monolingual Spanish -> Spanish doc
    {
        "query": "¿Qué proceso biológico convierte la energía de la luz solar en energía química en las plantas?",
        "target_doc_id": "doc_es",
        "target_chunk_id": "chunk_es_1",
        "category": "Monolingual ES",
    },
    # Cross-lingual Hindi Query -> English doc
    {
        "query": "कृत्रिम न्यूरल नेटवर्क को प्रशिक्षित करने के लिए मुख्य सुपरवाइज्ड लर्निंग एल्गोरिदम कौन सा है?",
        "target_doc_id": "doc_en",
        "target_chunk_id": "chunk_en_1",
        "category": "Cross-lingual HI -> EN",
    },
    # Cross-lingual English Query -> Hindi doc
    {
        "query": "What national initiative promotes solar energy generation and renewable capacity in India?",
        "target_doc_id": "doc_hi",
        "target_chunk_id": "chunk_hi_1",
        "category": "Cross-lingual EN -> HI",
    },
    # Cross-lingual English Query -> Hinglish doc
    {
        "query": "Which lander and rover successfully achieved a soft landing near the lunar south pole for India?",
        "target_doc_id": "doc_hinglish",
        "target_chunk_id": "chunk_hinglish_1",
        "category": "Cross-lingual EN -> Hinglish",
    },
    # Cross-lingual Hinglish Query -> English doc
    {
        "query": "Neural network ko train karne ke liye kaunsa backpropagation algorithm use hota hai?",
        "target_doc_id": "doc_en",
        "target_chunk_id": "chunk_en_1",
        "category": "Cross-lingual Hinglish -> EN",
    },
    # Cross-lingual Hinglish Query -> Hindi doc
    {
        "query": "Surya ki urja aur solar parks ke liye Bharat ka kaun sa mission chalaya gaya hai?",
        "target_doc_id": "doc_hi",
        "target_chunk_id": "chunk_hi_1",
        "category": "Cross-lingual Hinglish -> HI",
    },
    # Cross-lingual Spanish Query -> Hindi doc
    {
        "query": "¿Qué misión nacional en India promueve la generación de energía solar?",
        "target_doc_id": "doc_hi",
        "target_chunk_id": "chunk_hi_1",
        "category": "Cross-lingual ES -> HI",
    },
]


@pytest.fixture(scope="module")
def multilingual_indexed_store():
    """Index the benchmark documents with the verified multilingual embedding model."""
    provider = SentenceTransformersProvider()
    model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vector_store = MemoryVectorStore()

    for item in BENCHMARK_DOCUMENTS:
        doc_id = item["doc_id"]
        chunks = item["chunks"]
        texts = [c.text for c in chunks]
        vectors, dim = provider.embed(model, texts)
        vector_store.upsert_document(
            document_id=doc_id,
            chunks=chunks,
            embeddings=vectors,
            model=model,
            dimension=dim,
            old_ids=[],
        )

    return vector_store, provider, model


def test_multilingual_quantitative_retrieval_and_mrr(multilingual_indexed_store):
    """Verify Recall@1, Recall@3, Recall@5, and MRR across multilingual benchmark queries."""
    vector_store, provider, model = multilingual_indexed_store

    reciprocal_ranks = []
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    total_queries = len(BENCHMARK_QUERIES)

    for item in BENCHMARK_QUERIES:
        query = item["query"]
        target_doc = item["target_doc_id"]
        target_chunk = item["target_chunk_id"]

        query_vectors, _ = provider.embed(model, [query])
        results = vector_store.search(query_vectors[0], top_k=5)

        rank = None
        for idx, res in enumerate(results, start=1):
            res_doc = res.get("metadata", {}).get("document_id") or res.get("document_id")
            res_chunk = res.get("id")
            if res_doc == target_doc and res_chunk == target_chunk:
                rank = idx
                break
            elif res_doc == target_doc:
                # Same target document
                rank = idx
                break

        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            if rank <= 1:
                hits_at_1 += 1
            if rank <= 3:
                hits_at_3 += 1
            if rank <= 5:
                hits_at_5 += 1
        else:
            reciprocal_ranks.append(0.0)

    recall_1 = hits_at_1 / total_queries
    recall_3 = hits_at_3 / total_queries
    recall_5 = hits_at_5 / total_queries
    mrr = sum(reciprocal_ranks) / total_queries

    print(f"\n[Multilingual Retrieval Benchmark Results]")
    print(f"Total Queries Evaluated: {total_queries}")
    print(f"Recall@1: {recall_1:.4f} ({hits_at_1}/{total_queries})")
    print(f"Recall@3: {recall_3:.4f} ({hits_at_3}/{total_queries})")
    print(f"Recall@5: {recall_5:.4f} ({hits_at_5}/{total_queries})")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

    # Rigorous acceptance criteria
    assert recall_1 >= 0.80, f"Recall@1 too low: {recall_1:.4f}"
    assert recall_3 == 1.00, f"Recall@3 must be 1.00: {recall_3:.4f}"
    assert mrr >= 0.85, f"MRR too low: {mrr:.4f}"


def test_source_language_and_metadata_preservation(multilingual_indexed_store):
    """Verify that chunks preserve their verbatim source language and provenance metadata."""
    vector_store, provider, model = multilingual_indexed_store

    query_vectors, _ = provider.embed(model, ["solar energy"])
    results = vector_store.search(query_vectors[0], top_k=3)

    # Context assembly preserves exact chunk text without corruption
    context_str, supplied = assemble_context("solar energy", results, max_chars=4000, preserve_source_language=True)
    assert len(supplied) > 0
    assert "सौर ऊर्जा" in context_str or "Solar Energy" in context_str
    for item in supplied:
        assert item["metadata"]["document_name"] is not None
        assert item["metadata"]["page_number"] is not None


# ---------------------------------------------------------------------------
# 4. End-to-End Multilingual Generation & Localized Abstention Tests
# ---------------------------------------------------------------------------

def test_multilingual_abstention_on_unanswerable_questions(multilingual_indexed_store):
    """Verify localized abstention on unanswerable questions across languages."""
    vector_store, provider, model = multilingual_indexed_store
    settings = Settings()

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "docs.db"
        doc_store = DocumentStore(f"sqlite:///{db_path.as_posix()}", Path(temp_dir) / "sources")
        try:
            for item in BENCHMARK_DOCUMENTS:
                with doc_store.sessions.begin() as session:
                    session.add(
                        DocumentRow(
                            id=item["doc_id"],
                            filename=item["filename"],
                            file_type="txt",
                            content_type="text/plain",
                            source_path="dummy",
                            status="ready",
                            created_at=datetime.now(UTC),
                            chunks=[],
                            stale_chunks=[],
                        )
                    )

            mock_gen = MagicMock()
            mock_gen.generate.return_value = DEFAULT_ABSTENTION

            # English unanswerable
            res_en = answer_query(
                question="What is the capital city of planet Neptune in the year 3000?",
                top_k=2,
                document_ids=["doc_en"],
                vector_store=vector_store,
                provider=provider,
                settings=settings,
                store=doc_store,
                generation_provider=mock_gen,
                response_language="query",
                embedding_model=model,
            )
            assert res_en["abstained"] is True
            assert res_en["detected_language"] == "en"

            # Hindi unanswerable
            mock_gen.generate.return_value = LOCALIZED_ABSTENTIONS["hi"]
            res_hi = answer_query(
                question="मंगल ग्रह पर पहली मानव बस्ती कब स्थापित की गई थी?",
                top_k=2,
                document_ids=["doc_hi"],
                vector_store=vector_store,
                provider=provider,
                settings=settings,
                store=doc_store,
                generation_provider=mock_gen,
                response_language="query",
                embedding_model=model,
            )
            assert res_hi["abstained"] is True
            assert res_hi["detected_language"] == "hi"
            assert res_hi["response_language"] == "hi"

            # Hinglish unanswerable
            mock_gen.generate.return_value = LOCALIZED_ABSTENTIONS["hinglish"]
            res_hinglish = answer_query(
                question="Kailash parvat ke upar kaunsa international airport bana hai?",
                top_k=2,
                document_ids=["doc_hinglish"],
                vector_store=vector_store,
                provider=provider,
                settings=settings,
                store=doc_store,
                generation_provider=mock_gen,
                response_language="query",
                embedding_model=model,
            )
            assert res_hinglish["abstained"] is True
            assert res_hinglish["detected_language"] == "hinglish"
            assert res_hinglish["response_language"] == "hinglish"
        finally:
            doc_store.close()


def test_multilingual_grounded_answer_with_citations(multilingual_indexed_store):
    """Verify grounded answer generation with citations in target language."""
    vector_store, provider, model = multilingual_indexed_store
    settings = Settings()

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "docs.db"
        doc_store = DocumentStore(f"sqlite:///{db_path.as_posix()}", Path(temp_dir) / "sources")
        try:
            for item in BENCHMARK_DOCUMENTS:
                with doc_store.sessions.begin() as session:
                    session.add(
                        DocumentRow(
                            id=item["doc_id"],
                            filename=item["filename"],
                            file_type="txt",
                            content_type="text/plain",
                            source_path="dummy",
                            status="ready",
                            created_at=datetime.now(UTC),
                            chunks=[],
                            stale_chunks=[],
                        )
                    )

            mock_gen = MagicMock()
            mock_gen.generate.return_value = (
                "भारत में राष्ट्रीय सौर मिशन देश में सौर ऊर्जा उत्पादन को तेजी से बढ़ावा देता है [1]। "
                "इसके तहत विशाल सोलर पार्क स्थापित किए गए हैं [2]।"
            )

            res = answer_query(
                question="भारत में सौर ऊर्जा को कौन सा मिशन बढ़ावा देता है?",
                top_k=2,
                document_ids=["doc_hi"],
                vector_store=vector_store,
                provider=provider,
                settings=settings,
                store=doc_store,
                generation_provider=mock_gen,
                response_language="query",
                embedding_model=model,
            )

            assert res["abstained"] is False
            assert res["status"] == "completed"
            assert len(res["citations"]) > 0
            assert res["citations"][0]["document_name"] == "solar_energy_hi.txt"
            assert res["detected_language"] == "hi"
            assert res["response_language"] == "hi"
        finally:
            doc_store.close()


# ---------------------------------------------------------------------------
# 5. Graph Template & Custom Graph Execution Tests
# ---------------------------------------------------------------------------

def test_multilingual_template_graph_is_valid_and_executable():
    """Verify that the official Multilingual RAG template graph validates and executes."""
    template = TEMPLATES_BY_ID["multilingual-rag"]
    assert template is not None
    assert template.graph is not None

    settings = Settings()
    report = validate_traditional_graph(template.graph, settings, generation_provider="extractive")
    assert report.valid is True
    assert report.executable is True
    assert len(report.issues) == 0


def test_custom_user_created_graph_with_multilingual_configs():
    """Verify that user-created custom graphs with multilingual node configs validate and execute."""
    custom_graph = WorkflowGraph(
        id="custom-multilingual-workflow",
        name="Custom Multilingual Flow",
        architecture_id="multilingual-rag",
        template_id=None,
        nodes=[
            NodeInstance(id="source", node_type_id="document.source", config={}),
            NodeInstance(id="parser", node_type_id="document.parse", config={}),
            NodeInstance(id="chunker", node_type_id="chunk.split", config={"strategy": "recursive", "chunk_size": 800, "overlap": 100}),
            NodeInstance(
                id="embedder",
                node_type_id="embedding.create",
                config={
                    "provider": "sentence_transformers",
                    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "supported_languages": "multilingual",
                },
                provider_instance_id="embedding-provider",
            ),
            NodeInstance(
                id="index",
                node_type_id="vector.index",
                config={"backend": "chroma", "collection_name": "custom_docs"},
                provider_instance_id="chroma-local",
            ),
            NodeInstance(id="query", node_type_id="query.input", config={}),
            NodeInstance(
                id="retriever",
                node_type_id="retrieval.dense",
                config={"strategy": "semantic", "top_k": 4, "query_language_handling": "auto", "cross_lingual": True},
            ),
            NodeInstance(
                id="context",
                node_type_id="context.assemble",
                config={"preserve_source_language": True, "response_language": "hi", "max_chars": 8000},
            ),
            NodeInstance(
                id="answer",
                node_type_id="generation.answer",
                config={"provider": "extractive", "response_language": "hi"},
                provider_instance_id="generation-provider",
            ),
        ],
        connections=[
            GraphConnection(source_node_id="source", source_port_id="documents", target_node_id="parser", target_port_id="documents"),
            GraphConnection(source_node_id="parser", source_port_id="parsed_documents", target_node_id="chunker", target_port_id="parsed_documents"),
            GraphConnection(source_node_id="chunker", source_port_id="chunks", target_node_id="embedder", target_port_id="chunks"),
            GraphConnection(source_node_id="embedder", source_port_id="embeddings", target_node_id="index", target_port_id="embeddings"),
            GraphConnection(source_node_id="query", source_port_id="query", target_node_id="retriever", target_port_id="query"),
            GraphConnection(source_node_id="index", source_port_id="index", target_node_id="retriever", target_port_id="index"),
            GraphConnection(source_node_id="retriever", source_port_id="retrieved_chunks", target_node_id="context", target_port_id="retrieved_chunks"),
            GraphConnection(source_node_id="query", source_port_id="query", target_node_id="answer", target_port_id="query"),
            GraphConnection(source_node_id="context", source_port_id="context", target_node_id="answer", target_port_id="context"),
        ],
    )

    settings = Settings()
    report = validate_traditional_graph(custom_graph, settings, generation_provider="extractive")
    assert report.valid is True
    assert report.executable is True
    assert len(report.issues) == 0
