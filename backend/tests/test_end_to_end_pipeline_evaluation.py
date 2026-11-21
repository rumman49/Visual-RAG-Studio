"""End-to-End Evaluation and Scoring Suite for Visual RAG Studio.

Evaluates and scores:
1. Chunking Strategies:
   - recursive, fixed, sentence, semantic, hierarchical
   - Metrics: chunk count, length statistics, character retention, boundary integrity.
2. Retrieval Architectures:
   - Dense Semantic Retrieval (paraphrase-multilingual-MiniLM-L12-v2)
   - Sparse Lexical Retrieval (Okapi BM25)
   - Hybrid RRF Retrieval (Dense + BM25 Reciprocal Rank Fusion, k=60)
   - Hybrid Weighted Retrieval (alpha=0.6)
   - Reranking RAG (Flash Cross-Encoder Cross-Scorer)
   - Multilingual Cross-Lingual Retrieval (Hindi, Hinglish, Spanish, English)
   - Metrics: Hit Rate, Recall@1, Recall@3, Recall@5, Mean Reciprocal Rank (MRR), Latency.
3. Grounded Generation & Citation Verification:
   - Grounded answering with real Groq LLM (openai/gpt-oss-20b) and Local Extractive.
   - Citation accuracy and provenance integrity.
   - Grounded abstention on out-of-corpus / unanswerable questions.
"""

from __future__ import annotations

import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.rag.chunking import TextChunk, chunk_document
from app.rag.hybrid import (
    BM25Index,
    execute_hybrid_retrieval,
)
from app.rag.multilingual import (
    is_multilingual_abstention,
)
from app.rag.parser import parse_document
from app.rag.pipeline import answer_query
from app.rag.providers import (
    ExtractiveGeneratorProvider,
    GroqProvider,
    SentenceTransformersProvider,
)
from app.rag.ragspark_adapters import RagSparkReranker
from app.rag.storage import DocumentRow, DocumentStore
from app.rag.vector_store import MemoryVectorStore

# ---------------------------------------------------------------------------
# Comprehensive Multi-Domain Benchmark Corpus
# ---------------------------------------------------------------------------

BENCHMARK_ARTICLES = [
    {
        "id": "doc_arch",
        "filename": "rag_architecture.md",
        "text": (
            "# Visual RAG Studio Architecture\n\n"
            "Visual RAG Studio is an enterprise visual pipeline builder and execution engine for Retrieval-Augmented Generation. "
            "It supports deterministic DAG workflows with pluggable node types across ingestion, chunking, embedding, vector indexing, "
            "retrieval, reranking, context assembly, and answer generation.\n\n"
            "## Chunking Modules\n"
            "The system supports recursive character chunking, fixed-size windowing, sentence boundary chunking, "
            "semantic chunking, and hierarchical parent-child chunking. Overlap prevents context loss at chunk boundaries.\n\n"
            "## Vector Database Integration\n"
            "Chroma serves as the default persistent vector store, while MemoryVectorStore provides fast, zero-dependency in-memory testing. "
            "Pinecone and PostgreSQL pgvector are supported as production cloud and relational vector adapters.\n\n"
            "## Retrieval & Reranking\n"
            "Traditional RAG relies on dense vector cosine similarity. Hybrid RAG fuses dense semantic retrieval with Okapi BM25 lexical search "
            "using Reciprocal Rank Fusion with a smoothing constant of k=60. The Flash Reranker cross-scores top candidates to maximize relevance."
        ),
    },
    {
        "id": "doc_transformer",
        "filename": "transformer_attention.txt",
        "text": (
            "The Transformer model replaces recurrent and convolutional layers entirely with self-attention mechanisms. "
            "On the WMT 2014 English-to-German translation task, the big transformer model achieves 28.4 BLEU, improving over "
            "existing best results by over 2.0 BLEU. The base model has 65M parameters (65 million) with 6 encoder layers and "
            "6 decoder layers. Optimization uses the Adam optimizer with beta1 = 0.9, beta2 = 0.98, and epsilon = 1e-9 with 4000 warmup steps. "
            "Each sub-layer employs residual connections followed by LayerNorm with d_model = 512 and eps = 1e-6."
        ),
    },
    {
        "id": "doc_hindi",
        "filename": "solar_mission_india_hi.txt",
        "text": (
            "सौर ऊर्जा (Solar Energy) सूर्य से प्राप्त होने वाली प्रत्यक्ष प्रकाश और ऊष्मा ऊर्जा है। "
            "भारत में राष्ट्रीय सौर मिशन (National Solar Mission) देश में सौर ऊर्जा उत्पादन और नवीकरणीय ऊर्जा क्षमता को "
            "तेजी से बढ़ावा देने के लिए शुरू किया गया एक प्रमुख राष्ट्रीय कार्यक्रम है। "
            "फोटोवोल्टिक सेल (Photovoltaic cells) सूर्य के प्रकाश को सीधे विद्युत ऊर्जा में बदलते हैं। "
            "राजस्थान के भडला सोलर पार्क (Bhadla Solar Park) दुनिया के सबसे बड़े सोलर पार्कों में से एक है।"
        ),
    },
    {
        "id": "doc_hinglish",
        "filename": "chandrayaan_mission_hinglish.txt",
        "text": (
            "Chandrayaan-3 Bharat ka teesra lunar exploration mission tha jise ISRO ne launch kiya tha. "
            "Is mission me Vikram lander aur Pragyan rover shamil the. August 2023 me isne lunar south pole "
            "ke paas safalta se soft landing ki aur Bharat aisa karne wala vishwa ka pehla desh bana. "
            "Pragyan rover ne chandrama ki satah par ghoom kar sulphur aur oxygen ki presence confirm ki."
        ),
    },
]


# ---------------------------------------------------------------------------
# Benchmark Queries for Quantitative Evaluation
# ---------------------------------------------------------------------------

RETRIEVAL_TEST_QUERIES = [
    # Exact entities & numbers
    {
        "id": "q_bleu",
        "query": "What BLEU score was achieved on the WMT 2014 English-to-German translation task?",
        "target_doc_id": "doc_transformer",
        "expected_substring": "28.4 BLEU",
        "category": "Exact Metric",
    },
    {
        "id": "q_params",
        "query": "How many parameters does the base transformer model have in total?",
        "target_doc_id": "doc_transformer",
        "expected_substring": "65M parameters",
        "category": "Exact Metric",
    },
    {
        "id": "q_warmup",
        "query": "How many warmup steps were used with the Adam optimizer?",
        "target_doc_id": "doc_transformer",
        "expected_substring": "4000 warmup steps",
        "category": "Exact Parameter",
    },
    # Architecture concepts
    {
        "id": "q_rrf",
        "query": "What smoothing constant k is used in Reciprocal Rank Fusion for Hybrid RAG?",
        "target_doc_id": "doc_arch",
        "expected_substring": "k=60",
        "category": "Architecture",
    },
    {
        "id": "q_vdb",
        "query": "Which vector store serves as the default persistent vector store in Visual RAG Studio?",
        "target_doc_id": "doc_arch",
        "expected_substring": "Chroma",
        "category": "Architecture",
    },
    # Monolingual Hindi
    {
        "id": "q_hi_mission",
        "query": "भारत में सौर ऊर्जा उत्पादन को बढ़ावा देने के लिए कौन सा राष्ट्रीय कार्यक्रम शुरू किया गया?",
        "target_doc_id": "doc_hindi",
        "expected_substring": "राष्ट्रीय सौर मिशन",
        "category": "Hindi Monolingual",
    },
    {
        "id": "q_hi_park",
        "query": "राजस्थान में दुनिया का कौन सा सबसे बड़ा सोलर पार्क स्थापित है?",
        "target_doc_id": "doc_hindi",
        "expected_substring": "भडला सोलर पार्क",
        "category": "Hindi Monolingual",
    },
    # Monolingual Hinglish
    {
        "id": "q_hinglish_rover",
        "query": "Chandrayaan-3 mission me kaunsa rover aur lander shamil the?",
        "target_doc_id": "doc_hinglish",
        "expected_substring": "Pragyan rover",
        "category": "Hinglish Monolingual",
    },
    {
        "id": "q_hinglish_landing",
        "query": "Chandrayaan-3 ne chandrama ke kis pole ke paas soft landing ki thi?",
        "target_doc_id": "doc_hinglish",
        "expected_substring": "south pole",
        "category": "Hinglish Monolingual",
    },
    # Cross-lingual Hindi -> English
    {
        "id": "q_cross_hi_en",
        "query": "ट्रांसफार्मर मॉडल में रिकरेंट न्यूरल नेटवर्क के स्थान पर किस तंत्र का उपयोग किया जाता है?",
        "target_doc_id": "doc_transformer",
        "expected_substring": "self-attention",
        "category": "Cross-lingual HI -> EN",
    },
    # Cross-lingual English -> Hindi
    {
        "id": "q_cross_en_hi",
        "query": "Which solar park in Rajasthan is recognized as one of the largest in the world?",
        "target_doc_id": "doc_hindi",
        "expected_substring": "भडला सोलर पार्क",
        "category": "Cross-lingual EN -> HI",
    },
    # Cross-lingual English -> Hinglish
    {
        "id": "q_cross_en_hinglish",
        "query": "Which elements did the Pragyan rover confirm on the lunar surface?",
        "target_doc_id": "doc_hinglish",
        "expected_substring": "sulphur",
        "category": "Cross-lingual EN -> Hinglish",
    },
]

UNANSWERABLE_QUERIES = [
    {
        "query": "What is the average surface temperature of the exoplanet Proxima Centauri b?",
        "expected_lang": "en",
    },
    {
        "query": "मंगल ग्रह पर पहली मानव कॉलोनी किस वर्ष में बसाई गई थी?",
        "expected_lang": "hi",
    },
    {
        "query": "Jupiter ke chaand Ganymede par submarine kab launch hui thi?",
        "expected_lang": "hinglish",
    },
]


# ===========================================================================
# 1. Chunking Strategy Evaluation & Scoring
# ===========================================================================

def test_chunking_strategies_comparative_evaluation():
    """Evaluate and score chunking strategies across integrity, length, and bounds."""
    test_text = BENCHMARK_ARTICLES[0]["text"]
    parsed_doc = parse_document("doc_arch", "rag_architecture.md", test_text.encode("utf-8"))

    strategies = ["recursive", "fixed", "sentence", "ragspark.semantic", "hierarchical"]
    chunk_scores: dict[str, dict[str, Any]] = {}

    for strat in strategies:
        start_t = time.perf_counter()
        chunks = chunk_document(parsed_doc, strategy_id=strat, settings={"chunk_size": 400, "overlap": 60})
        elapsed_ms = (time.perf_counter() - start_t) * 1000

        assert len(chunks) > 0, f"Strategy {strat} produced 0 chunks."

        lengths = [len(c.text) for c in chunks]
        min_len = min(lengths)
        max_len = max(lengths)
        avg_len = sum(lengths) / len(chunks)

        # Character retention: all chunks should preserve substantive text
        total_chunk_chars = sum(lengths)
        retention_ratio = min(1.0, total_chunk_chars / len(test_text))

        # Boundary integrity: verify no chunk starts or ends with raw whitespace corruption
        clean_boundaries = all(c.text == c.text.strip() for c in chunks)
        assert clean_boundaries, f"Strategy {strat} produced dirty chunk boundaries."

        chunk_scores[strat] = {
            "chunk_count": len(chunks),
            "min_length": min_len,
            "max_length": max_len,
            "avg_length": round(avg_len, 1),
            "retention_ratio": round(retention_ratio, 3),
            "latency_ms": round(elapsed_ms, 2),
            "clean_boundaries": clean_boundaries,
        }

    # Print Chunking Scorecard
    print("\n" + "=" * 78)
    print("CHUNKING STRATEGIES EVALUATION SCORECARD")
    print("=" * 78)
    print(f"{'Strategy':<15} | {'Chunks':<7} | {'Min Len':<8} | {'Avg Len':<8} | {'Max Len':<8} | {'Latency':<9}")
    print("-" * 78)
    for strat, sc in chunk_scores.items():
        print(
            f"{strat:<15} | {sc['chunk_count']:<7} | {sc['min_length']:<8} | "
            f"{sc['avg_length']:<8} | {sc['max_length']:<8} | {sc['latency_ms']:<6} ms"
        )
    print("=" * 78)

    # Verification assertions
    assert chunk_scores["recursive"]["chunk_count"] >= 3
    assert chunk_scores["fixed"]["chunk_count"] >= 3
    assert chunk_scores["sentence"]["chunk_count"] >= 3
    assert chunk_scores["hierarchical"]["chunk_count"] >= 3


# ===========================================================================
# 2. Retrieval Architectures Evaluation & Scoring
# ===========================================================================

@pytest.fixture(scope="module")
def multi_domain_vector_store():
    """Index all benchmark documents into MemoryVectorStore using SentenceTransformers."""
    provider = SentenceTransformersProvider()
    model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vstore = MemoryVectorStore()

    all_chunks: list[TextChunk] = []
    for art in BENCHMARK_ARTICLES:
        parsed_doc = parse_document(art["id"], art["filename"], art["text"].encode("utf-8"))
        chunks = chunk_document(parsed_doc, strategy_id="recursive", settings={"chunk_size": 350, "overlap": 50})
        all_chunks.extend(chunks)

        texts = [c.text for c in chunks]
        vectors, dim = provider.embed(model, texts)
        vstore.upsert_document(
            document_id=art["id"],
            chunks=chunks,
            embeddings=vectors,
            model=model,
            dimension=dim,
            old_ids=[],
        )

    return vstore, provider, model, all_chunks


def _compute_metrics(ranked_results: list[dict], target_doc: str, target_substring: str) -> tuple[float, float, float, float]:
    """Compute Recall@1, Recall@3, Recall@5, and Reciprocal Rank (RR)."""
    r1 = 0.0
    r3 = 0.0
    r5 = 0.0
    rr = 0.0

    for rank_idx, item in enumerate(ranked_results[:5], start=1):
        doc_id = item.get("metadata", {}).get("document_id") or ""
        text = item.get("text", "")
        # Matches if target document matches and key fact substring is present
        if doc_id == target_doc and (target_substring.lower() in text.lower()):
            if rank_idx == 1:
                r1 = 1.0
            if rank_idx <= 3:
                r3 = 1.0
            if rank_idx <= 5:
                r5 = 1.0
            if rr == 0.0:
                rr = 1.0 / rank_idx
            break

    return r1, r3, r5, rr


def test_retrieval_architectures_comparative_evaluation(multi_domain_vector_store):
    """Compare Traditional Dense, Sparse BM25, Hybrid RRF, Hybrid Weighted, and Reranking."""
    vstore, provider, model, all_chunks = multi_domain_vector_store

    # Build standalone BM25 index on chunk corpus
    bm25 = BM25Index()
    bm25.index([{"id": c.chunk_id, "text": c.text, "metadata": c.chroma_metadata()} for c in all_chunks])

    flash_reranker = RagSparkReranker(model="flash")

    strategies = ["Dense (Traditional)", "Sparse (BM25)", "Hybrid (RRF)", "Hybrid (Weighted)", "Reranking (Flash)"]
    scorecard: dict[str, dict[str, float]] = {s: {"r1": 0.0, "r3": 0.0, "r5": 0.0, "mrr": 0.0, "latency": 0.0} for s in strategies}

    total_queries = len(RETRIEVAL_TEST_QUERIES)

    for q in RETRIEVAL_TEST_QUERIES:
        query_text = q["query"]
        target_doc = q["target_doc_id"]
        target_sub = q["expected_substring"]

        q_vectors, _ = provider.embed(model, [query_text])
        q_vec = q_vectors[0]

        # 1. Dense Semantic
        t0 = time.perf_counter()
        dense_results = vstore.search(q_vec, top_k=5)
        scorecard["Dense (Traditional)"]["latency"] += (time.perf_counter() - t0) * 1000
        r1, r3, r5, rr = _compute_metrics(dense_results, target_doc, target_sub)
        scorecard["Dense (Traditional)"]["r1"] += r1
        scorecard["Dense (Traditional)"]["r3"] += r3
        scorecard["Dense (Traditional)"]["r5"] += r5
        scorecard["Dense (Traditional)"]["mrr"] += rr

        # 2. Sparse BM25
        t0 = time.perf_counter()
        bm25_results = bm25.search(query_text, top_k=5)
        scorecard["Sparse (BM25)"]["latency"] += (time.perf_counter() - t0) * 1000
        r1, r3, r5, rr = _compute_metrics(bm25_results, target_doc, target_sub)
        scorecard["Sparse (BM25)"]["r1"] += r1
        scorecard["Sparse (BM25)"]["r3"] += r3
        scorecard["Sparse (BM25)"]["r5"] += r5
        scorecard["Sparse (BM25)"]["mrr"] += rr

        # 3. Hybrid RRF
        t0 = time.perf_counter()
        hybrid_rrf = execute_hybrid_retrieval(
            vector_store=vstore,
            query_vector=q_vec,
            query_text=query_text,
            top_k=5,
            strategy="hybrid",
            fusion_method="rrf",
            rrf_k=60,
        )
        scorecard["Hybrid (RRF)"]["latency"] += (time.perf_counter() - t0) * 1000
        r1, r3, r5, rr = _compute_metrics(hybrid_rrf, target_doc, target_sub)
        scorecard["Hybrid (RRF)"]["r1"] += r1
        scorecard["Hybrid (RRF)"]["r3"] += r3
        scorecard["Hybrid (RRF)"]["r5"] += r5
        scorecard["Hybrid (RRF)"]["mrr"] += rr

        # 4. Hybrid Weighted
        t0 = time.perf_counter()
        hybrid_weighted = execute_hybrid_retrieval(
            vector_store=vstore,
            query_vector=q_vec,
            query_text=query_text,
            top_k=5,
            strategy="hybrid",
            fusion_method="weighted",
            alpha=0.6,
        )
        scorecard["Hybrid (Weighted)"]["latency"] += (time.perf_counter() - t0) * 1000
        r1, r3, r5, rr = _compute_metrics(hybrid_weighted, target_doc, target_sub)
        scorecard["Hybrid (Weighted)"]["r1"] += r1
        scorecard["Hybrid (Weighted)"]["r3"] += r3
        scorecard["Hybrid (Weighted)"]["r5"] += r5
        scorecard["Hybrid (Weighted)"]["mrr"] += rr

        # 5. Reranking RAG (Flash Cross-Scorer on Hybrid candidate pool)
        t0 = time.perf_counter()
        candidate_pool = execute_hybrid_retrieval(
            vector_store=vstore,
            query_vector=q_vec,
            query_text=query_text,
            top_k=10,
            strategy="hybrid",
            fusion_method="rrf",
        )
        reranked = flash_reranker.rerank(query_text, candidate_pool, top_k=5)
        scorecard["Reranking (Flash)"]["latency"] += (time.perf_counter() - t0) * 1000
        r1, r3, r5, rr = _compute_metrics(reranked, target_doc, target_sub)
        scorecard["Reranking (Flash)"]["r1"] += r1
        scorecard["Reranking (Flash)"]["r3"] += r3
        scorecard["Reranking (Flash)"]["r5"] += r5
        scorecard["Reranking (Flash)"]["mrr"] += rr

    # Normalize metrics across queries
    for s in strategies:
        scorecard[s]["r1"] = round(scorecard[s]["r1"] / total_queries, 4)
        scorecard[s]["r3"] = round(scorecard[s]["r3"] / total_queries, 4)
        scorecard[s]["r5"] = round(scorecard[s]["r5"] / total_queries, 4)
        scorecard[s]["mrr"] = round(scorecard[s]["mrr"] / total_queries, 4)
        scorecard[s]["latency"] = round(scorecard[s]["latency"] / total_queries, 2)

    # Print Retrieval Scorecard Table
    print("\n" + "=" * 82)
    print("RETRIEVAL ARCHITECTURES EVALUATION SCORECARD")
    print("=" * 82)
    print(f"{'Architecture':<22} | {'Recall@1':<9} | {'Recall@3':<9} | {'Recall@5':<9} | {'MRR':<8} | {'Latency':<9}")
    print("-" * 82)
    for s in strategies:
        sc = scorecard[s]
        print(
            f"{s:<22} | {sc['r1'] * 100:>6.1f}%   | {sc['r3'] * 100:>6.1f}%   | "
            f"{sc['r5'] * 100:>6.1f}%   | {sc['mrr']:>6.3f} | {sc['latency']:>6.2f} ms"
        )
    print("=" * 82)

    # Acceptance criteria:
    # Hybrid and Reranking should achieve high Recall@3 and MRR >= 0.80
    assert scorecard["Hybrid (RRF)"]["r3"] >= 0.85
    assert scorecard["Hybrid (RRF)"]["mrr"] >= 0.80
    assert scorecard["Reranking (Flash)"]["r3"] >= 0.85
    assert scorecard["Reranking (Flash)"]["mrr"] >= 0.80


# ===========================================================================
# 3. Grounded Answer Generation, Citations & Abstention Evaluation
# ===========================================================================

def test_end_to_end_answer_generation_and_grounding(multi_domain_vector_store):
    """Test full pipeline execution: answers grounded in retrieved evidence with citations."""
    vstore, provider, model, _ = multi_domain_vector_store
    settings = Settings()

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "docs.db"
        doc_store = DocumentStore(f"sqlite:///{db_path.as_posix()}", Path(temp_dir) / "sources")
        try:
            for art in BENCHMARK_ARTICLES:
                with doc_store.sessions.begin() as session:
                    session.add(
                        DocumentRow(
                            id=art["id"],
                            filename=art["filename"],
                            file_type="txt",
                            content_type="text/plain",
                            source_path="dummy",
                            status="ready",
                            created_at=datetime.now(UTC),
                            chunks=[],
                            stale_chunks=[],
                        )
                    )

            # A. Extractive Generator Evaluation
            extractive_gen = ExtractiveGeneratorProvider()
            res_ext = answer_query(
                question="What BLEU score was achieved on the WMT 2014 English-to-German translation task?",
                top_k=3,
                document_ids=["doc_transformer"],
                vector_store=vstore,
                provider=provider,
                settings=settings,
                store=doc_store,
                generation_provider=extractive_gen,
                embedding_model=model,
                retrieval_strategy="hybrid",
            )

            assert res_ext["status"] == "completed"
            assert not res_ext["abstained"]
            assert len(res_ext["citations"]) > 0
            assert "28.4" in res_ext["answer"]
            # Verify citation structure
            first_cit = res_ext["citations"][0]
            assert "document_name" in first_cit
            assert first_cit["document_name"] == "transformer_attention.txt"

            # B. Real Groq Provider Evaluation (if configured)
            groq_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
            if groq_key:
                groq_gen = GroqProvider(api_key=groq_key)
                groq_model = settings.groq_generation_model or "openai/gpt-oss-20b"

                res_groq = answer_query(
                    question="Which rover and lander were part of Chandrayaan-3?",
                    top_k=3,
                    document_ids=["doc_hinglish"],
                    vector_store=vstore,
                    provider=provider,
                    settings=settings,
                    store=doc_store,
                    generation_provider=groq_gen,
                    generation_model=groq_model,
                    embedding_model=model,
                    retrieval_strategy="hybrid",
                )
                assert res_groq["status"] == "completed"
                assert not res_groq["abstained"]
                assert len(res_groq["citations"]) > 0
                ans_lower = res_groq["answer"].lower()
                assert ("pragyan" in ans_lower or "vikram" in ans_lower)

            # C. Abstention Evaluation on Out-of-Corpus Questions
            for unans in UNANSWERABLE_QUERIES:
                q_text = unans["query"]
                expected_lang = unans["expected_lang"]

                res_unans = answer_query(
                    question=q_text,
                    top_k=2,
                    document_ids=["doc_arch"],
                    vector_store=vstore,
                    provider=provider,
                    settings=settings,
                    store=doc_store,
                    generation_provider=extractive_gen,
                    embedding_model=model,
                    response_language="query",
                )
                assert res_unans["abstained"] is True or is_multilingual_abstention(res_unans["answer"])
                assert res_unans["detected_language"] == expected_lang
        finally:
            doc_store.close()
