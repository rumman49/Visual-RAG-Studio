"""RagSpark Integration Smoke Test for Visual RAG Studio.

Proves that:
1. The ragspark library installs, imports, and runs in the Visual RAG Studio environment.
2. Verified parsing, chunking, embedding, vector database, retrieval, reranking, prompt techniques, and generation operate end-to-end.
3. Building blocks are exposed in the Visual RAG Studio registry.
"""

from __future__ import annotations

import sys
from pathlib import Path

backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

def run_smoke_test() -> bool:
    print("=" * 70)
    print("Visual RAG Studio -- RagSpark Integration Smoke Test")
    print("=" * 70)

    # 1. Import Smoke Test
    print("\n[1/10] Importing RagSpark core modules...")
    try:
        import ragspark
        import ragspark.chunker as r_chunker
        import ragspark.embedder as r_embedder
        import ragspark.llm as r_llm
        import ragspark.parser as r_parser
        import ragspark.prompt_eng as r_prompt
        import ragspark.rag as r_rag
        import ragspark.reranker as r_reranker
        import ragspark.vectordb as r_vectordb

        print(f"  [OK] ragspark v{ragspark.__version__} imported successfully.")
    except Exception as exc:
        print(f"  [FAIL] Failed to import RagSpark: {exc}")
        return False

    # 2. Parsing
    print("\n[2/10] Testing Document Parsing (ragspark.parser)...")
    try:
        sample_text = (
            "# Visual RAG Studio & RagSpark\n\n"
            "Visual RAG Studio integrates RagSpark as modular building blocks.\n\n"
            "## Legal Compliance\n\n"
            "Each block provides transparent provenance and contract verification."
        )
        doc = r_parser.parse_text(sample_text)
        print(f"  [OK] Parsed document: text_len={len(doc.text)}, pages={doc.total_pages}")
    except Exception as exc:
        print(f"  [FAIL] Parsing failed: {exc}")
        return False

    # 3. Chunking across strategies
    print("\n[3/10] Testing Chunking Strategies (ragspark.chunker)...")
    try:
        for strat in ["recursive", "legal_clause", "sentence", "sliding_window"]:
            chunks = r_chunker.chunk(doc.text, strategy=strat, chunk_size=200, chunk_overlap=30)
            print(f"  * Strategy '{strat}': produced {len(chunks)} chunk(s)")
            assert len(chunks) >= 1
        print("  [OK] Chunking strategies executed successfully.")
    except Exception as exc:
        print(f"  [FAIL] Chunking failed: {exc}")
        return False

    # 4. Embeddings
    print("\n[4/10] Testing Embeddings (ragspark.embedder)...")
    try:
        chunk_texts = [c.text for c in chunks]
        vectors = r_embedder.embed(chunk_texts, provider="fast_local", strategy="dense")
        assert len(vectors) == len(chunk_texts)
        dim = len(vectors[0])
        q_vec = r_embedder.embed_query("provenance verification", provider="fast_local")
        assert len(q_vec) == dim
        print(f"  [OK] Embeddings generated: {len(vectors)} vector(s), dimension={dim}")
    except Exception as exc:
        print(f"  [FAIL] Embedding failed: {exc}")
        return False

    # 5. Vector Store & Retrieval Strategies
    print("\n[5/10] Testing Vector Store & Retrieval (ragspark.vectordb)...")
    try:
        store = r_vectordb.VectorStore(provider="memory", backend="memory", embed_provider="fast_local")
        store.add(chunks)
        assert store.count() == len(chunks)
        for ret_strat in ["hybrid", "semantic", "keyword", "bm25"]:
            hits = store.search(query_text="modular building blocks", top_k=1, strategy=ret_strat)
            assert len(hits) >= 1
            print(f"  * Retrieval '{ret_strat}': top hit score={hits[0].score:.4f}")
        print("  [OK] VectorDB memory backend and retrieval strategies validated.")
    except Exception as exc:
        print(f"  [FAIL] VectorDB failed: {exc}")
        return False

    # 6. Reranking
    print("\n[6/10] Testing Reranking (ragspark.reranker)...")
    try:
        candidates = [
            "Strawberries and blueberries are fresh fruits.",
            "Visual RAG Studio integrates RagSpark modular building blocks.",
            "Weather forecasts predict sunny skies tomorrow.",
        ]
        ranked = r_reranker.rerank("modular building blocks", candidates, top_k=2, model="flash")
        assert len(ranked) == 2
        assert "Visual RAG Studio" in ranked[0]
        print(f"  [OK] Flash reranker selected top candidate: '{ranked[0][:50]}...'")
    except Exception as exc:
        print(f"  [FAIL] Reranking failed: {exc}")
        return False

    # 7. Prompt Engineering
    print("\n[7/10] Testing Prompt Engineering (ragspark.prompt_eng)...")
    try:
        for tech in ["standard", "cot", "citation_strict"]:
            prompt = r_prompt.build("What does Visual RAG Studio provide?", [ranked[0]], technique=tech)
            assert prompt.user_prompt
        lang = r_prompt.detect_language("Bonjour tout le monde")
        print(f"  [OK] Prompt building verified across techniques. Language detected: {lang}")
    except Exception as exc:
        print(f"  [FAIL] Prompt engineering failed: {exc}")
        return False

    # 8. Generation
    print("\n[8/10] Testing LLM Generation (ragspark.llm)...")
    try:
        answer = r_llm.generate(
            prompt="What does Visual RAG Studio integrate?",
            provider="extractive",
            context=[ranked[0]],
        )
        assert answer.text
        print(f"  [OK] Extractive generation produced answer: {answer.text[:80]}...")
    except Exception as exc:
        print(f"  [FAIL] LLM generation failed: {exc}")
        return False

    # 9. End-to-End Pipeline
    print("\n[9/10] Testing End-to-End Pipeline (ragspark.rag)...")
    try:
        pipeline = r_rag.create(
            sample_text,
            chunk_strategy="sentence",
            embed_provider="fast_local",
            vectordb="memory",
            retrieval_strategy="hybrid",
            prompt_technique="standard",
            llm_provider="extractive",
        )
        res = pipeline.ask("What does Visual RAG Studio integrate?")
        assert res.answer
        print(f"  [OK] Pipeline answer: {res.answer[:80]}...")
    except Exception as exc:
        print(f"  [FAIL] End-to-end pipeline failed: {exc}")
        return False

    # 10. Studio Registry Verification
    print("\n[10/10] Verifying Visual RAG Studio Registry Exposure...")
    try:
        from app.rag.ragspark_adapters import RAGSPARK_CHUNKING_STRATEGIES
        from app.registry import CHUNKING_STRATEGIES, NODE_TYPES_BY_ID, REGISTRY

        ragspark_nodes = [nid for nid in NODE_TYPES_BY_ID if "ragspark" in nid]
        print(f"  * Registered RagSpark node types: {len(ragspark_nodes)} ({', '.join(ragspark_nodes)})")
        assert len(ragspark_nodes) == 8

        ragspark_strats = [s for s in CHUNKING_STRATEGIES if s.startswith("ragspark_")]
        print(f"  * Registered RagSpark chunking strategies: {len(ragspark_strats)}")
        assert len(ragspark_strats) == len(RAGSPARK_CHUNKING_STRATEGIES)

        adapters = [a.id for a in REGISTRY.provider_adapters if "ragspark" in a.id]
        print(f"  * Registered RagSpark provider adapters: {len(adapters)}")
        assert len(adapters) == 7

        print("  [OK] Registry exposure and discoverability verified.")
    except Exception as exc:
        print(f"  [FAIL] Registry verification failed: {exc}")
        return False

    print("\n" + "=" * 70)
    print("ALL 10 SMOKE TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
