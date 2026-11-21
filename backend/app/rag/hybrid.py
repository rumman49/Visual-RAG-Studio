"""Hybrid retrieval engine combining Dense Vector Search and Okapi BM25 Lexical Search via RRF."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from app.rag.vector_store import VectorStoreError


class BM25Index:
    """Fast, zero-dependency Okapi BM25 index with Robertson-Spärck Jones IDF."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: list[int] = []
        self.doc_freqs: dict[str, int] = {}
        self.term_freqs: list[dict[str, int]] = []
        self.documents: list[dict[str, Any]] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        if not text:
            return []
        return [w.lower() for w in re.findall(r"\w+", text) if len(w) >= 1]

    def index(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.corpus_size = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}
        self.term_freqs = []

        total_tokens = 0
        for doc in documents:
            text = doc.get("text", "")
            tokens = self.tokenize(text)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_tokens += doc_len

            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self.term_freqs.append(tf)

            for tok in tf:
                self.doc_freqs[tok] = self.doc_freqs.get(tok, 0) + 1

        self.avg_doc_len = (total_tokens / self.corpus_size) if self.corpus_size > 0 else 0.0

    def idf(self, term: str) -> float:
        n_q = self.doc_freqs.get(term, 0)
        # Robertson-Spärck Jones formula with +1.0 floor to guarantee non-negative IDF
        return math.log((self.corpus_size - n_q + 0.5) / (n_q + 0.5) + 1.0)

    def search(
        self,
        query: str,
        top_k: int = 10,
        k1: float | None = None,
        b: float | None = None,
    ) -> list[dict[str, Any]]:
        if not self.documents or not query:
            return []

        k1_val = k1 if k1 is not None else self.k1
        b_val = b if b is not None else self.b

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        q_lower = query.lower().strip()
        scored: list[tuple[float, int]] = []

        for idx in range(self.corpus_size):
            tf_map = self.term_freqs[idx]
            doc_len = self.doc_lengths[idx]
            score = 0.0

            for term in q_tokens:
                tf = tf_map.get(term, 0)
                if tf > 0:
                    idf_val = self.idf(term)
                    num = tf * (k1_val + 1.0)
                    denom = tf + k1_val * (1.0 - b_val + b_val * (doc_len / max(1.0, self.avg_doc_len)))
                    score += idf_val * (num / denom)

            # Boost exact phrase matches (crucial for exact numbers, named entities, identifiers)
            doc_text = self.documents[idx].get("text", "").lower()
            if q_lower in doc_text:
                score += 2.0

            if score > 0:
                scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(scored[:top_k], start=1):
            doc_copy = dict(self.documents[idx])
            # Ensure metadata dict is present
            if "metadata" not in doc_copy or not isinstance(doc_copy["metadata"], dict):
                doc_copy["metadata"] = {}
            doc_copy["score"] = float(score)
            doc_copy["sparse_score"] = float(score)
            doc_copy["rank"] = rank
            doc_copy["retrieval_strategy"] = "bm25"
            results.append(doc_copy)

        return results


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]],
    k: int = 60,
    weights: list[float] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Standard Reciprocal Rank Fusion (RRF).
    RRF(d) = sum_m (weight_m / (k + rank_m(d)))
    """
    if not rankings:
        return []

    if weights is None:
        weights = [1.0] * len(rankings)

    scores: dict[str, float] = {}
    doc_map: dict[str, dict[str, Any]] = {}
    dense_scores: dict[str, float] = {}
    sparse_scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}
    dense_distances: dict[str, float | None] = {}

    for system_idx, ranking in enumerate(rankings):
        weight = weights[system_idx] if system_idx < len(weights) else 1.0
        for rank, item in enumerate(ranking, start=1):
            doc_id = item.get("id") or item.get("chunk_id")
            if not doc_id:
                continue

            if doc_id not in doc_map:
                doc_map[doc_id] = dict(item)
            else:
                # Merge metadata fields if missing
                existing_meta = doc_map[doc_id].setdefault("metadata", {})
                new_meta = item.get("metadata") or {}
                for mk, mv in new_meta.items():
                    if mk not in existing_meta:
                        existing_meta[mk] = mv

            if system_idx == 0:
                dense_ranks[doc_id] = rank
                dist = item.get("distance")
                dense_distances[doc_id] = float(dist) if dist is not None else None
                dense_scores[doc_id] = float(max(0.0, 1.0 - dist)) if dist is not None else 0.0
            else:
                sparse_ranks[doc_id] = rank
                sparse_scores[doc_id] = float(item.get("sparse_score") or item.get("score") or 0.0)

            rrf_contrib = weight / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_contrib

    if not scores:
        return []

    # Sort descending by RRF score
    sorted_doc_ids = sorted(scores.keys(), key=lambda d_id: scores[d_id], reverse=True)
    max_possible = sum(w / (k + 1) for w in weights) or 1.0

    results: list[dict[str, Any]] = []
    for rank_idx, doc_id in enumerate(sorted_doc_ids[:top_k], start=1):
        item = dict(doc_map[doc_id])
        rrf_val = scores[doc_id]
        normalized = min(1.0, rrf_val / max_possible)
        item["score"] = rrf_val
        item["rrf_score"] = rrf_val
        item["normalized_score"] = normalized
        # Preserve the real dense cosine distance; fusion rank/score is separate.
        dense_distance = dense_distances.get(doc_id)
        if dense_distance is None:
            item.pop("distance", None)
        else:
            item["distance"] = dense_distance
        item["rank"] = rank_idx
        item["dense_rank"] = dense_ranks.get(doc_id)
        item["sparse_rank"] = sparse_ranks.get(doc_id)
        item["dense_score"] = dense_scores.get(doc_id, 0.0)
        item["sparse_score"] = sparse_scores.get(doc_id, 0.0)
        item["retrieval_strategy"] = "hybrid_rrf"
        results.append(item)

    return results


def weighted_score_fusion(
    dense_candidates: list[dict[str, Any]],
    sparse_candidates: list[dict[str, Any]],
    alpha: float = 0.65,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Min-Max Normalized Weighted Score Fusion.
    Score = alpha * DenseScore + (1 - alpha) * SparseScore
    """
    doc_map: dict[str, dict[str, Any]] = {}
    dense_raw: dict[str, float] = {}
    sparse_raw: dict[str, float] = {}

    for item in dense_candidates:
        doc_id = item.get("id") or item.get("chunk_id")
        if not doc_id:
            continue
        doc_map[doc_id] = dict(item)
        dist = item.get("distance")
        dense_raw[doc_id] = max(0.0, 1.0 - dist) if dist is not None else 0.0

    for item in sparse_candidates:
        doc_id = item.get("id") or item.get("chunk_id")
        if not doc_id:
            continue
        if doc_id not in doc_map:
            doc_map[doc_id] = dict(item)
        sparse_raw[doc_id] = float(item.get("sparse_score") or item.get("score") or 0.0)

    if not doc_map:
        return []

    # Min-Max normalize dense scores
    d_vals = list(dense_raw.values())
    d_min, d_max = (min(d_vals), max(d_vals)) if d_vals else (0.0, 1.0)
    d_range = (d_max - d_min) or 1.0

    # Min-Max normalize sparse scores
    s_vals = list(sparse_raw.values())
    s_min, s_max = (min(s_vals), max(s_vals)) if s_vals else (0.0, 1.0)
    s_range = (s_max - s_min) or 1.0

    combined: list[tuple[float, str]] = []
    for doc_id in doc_map:
        d_norm = (dense_raw.get(doc_id, d_min) - d_min) / d_range if doc_id in dense_raw else 0.0
        s_norm = (sparse_raw.get(doc_id, s_min) - s_min) / s_range if doc_id in sparse_raw else 0.0
        final_score = (alpha * d_norm) + ((1.0 - alpha) * s_norm)
        combined.append((final_score, doc_id))

    combined.sort(key=lambda x: x[0], reverse=True)

    results: list[dict[str, Any]] = []
    for rank_idx, (score, doc_id) in enumerate(combined[:top_k], start=1):
        item = dict(doc_map[doc_id])
        item["score"] = score
        item["weighted_score"] = score
        item["rank"] = rank_idx
        item["retrieval_strategy"] = "hybrid_weighted"
        results.append(item)

    return results


def _tag_dense_fallback(
    dense_candidates: list[dict[str, Any]],
    requested_strategy: str,
    reason: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Explicitly identify dense-only fallback; never silently pretend hybrid executed."""
    results = []
    for rank_idx, item in enumerate(dense_candidates[:top_k], start=1):
        doc_copy = dict(item)
        if "metadata" not in doc_copy or not isinstance(doc_copy["metadata"], dict):
            doc_copy["metadata"] = {}
        doc_copy["retrieval_strategy"] = "dense_fallback"
        doc_copy["requested_strategy"] = requested_strategy
        doc_copy["fallback_reason"] = reason
        doc_copy["rank"] = rank_idx
        results.append(doc_copy)
    return results


def execute_hybrid_retrieval(
    vector_store: Any,
    query_vector: list[float],
    query_text: str,
    top_k: int = 5,
    document_ids: list[str] | None = None,
    strategy: str = "hybrid",
    fusion_method: str = "rrf",
    rrf_k: int = 60,
    alpha: float = 0.65,
    dense_top_k: int = 10,
    sparse_top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
    store: Any = None,
    on_step: Callable[[str, str], None] | None = None,
) -> list[dict[str, Any]]:
    """
    End-to-end hybrid retrieval coordinating Dense Semantic Search and Sparse BM25 Search.
    """
    strat = strategy.lower().strip() if strategy else "semantic"

    # Step 1: Run Dense Retrieval if needed
    dense_results: list[dict[str, Any]] = []
    if strat in ("semantic", "hybrid", "rrf"):
        fetch_n = max(dense_top_k, top_k)
        try:
            dense_results = vector_store.search(
                vector=query_vector,
                top_k=fetch_n,
                document_ids=document_ids,
                query_text=None,
            )
        except TypeError as exc:
            # Keep compatibility with third-party adapters that predate the
            # optional query_text parameter, without masking other failures.
            if "query_text" not in str(exc):
                raise
            dense_results = vector_store.search(
                vector=query_vector, top_k=fetch_n, document_ids=document_ids
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Hybrid dense retrieval failed ({type(exc).__name__}: {str(exc).splitlines()[0][:160]})."
            ) from exc
        if on_step:
            on_step("retriever", "completed")

    if strat == "semantic":
        return dense_results[:top_k]

    # Step 2: Extract Chunks for Sparse BM25 Search
    corpus_docs: list[dict[str, Any]] = []
    clean_doc_ids = [str(d) for d in document_ids if d and str(d).strip()] if document_ids else []
    eligible_set = set(clean_doc_ids) if clean_doc_ids else None

    # First attempt: DocumentStore preserves parsed/chunked records with full metadata
    if store is not None:
        try:
            doc_rows = store.list()
            for row in doc_rows:
                if eligible_set and row.get("document_id") not in eligible_set:
                    continue
                doc_name = row.get("filename", "")
                doc_id = row.get("document_id", "")
                for chk in row.get("chunks", []):
                    c_id = chk.get("chunk_id") or chk.get("id")
                    c_meta = dict(chk.get("metadata") or {})
                    c_meta.setdefault("document_id", doc_id)
                    c_meta.setdefault("document_name", doc_name)
                    if "page_number" in chk and "page_number" not in c_meta:
                        c_meta["page_number"] = chk["page_number"]
                    if "section_path" in chk and "section_path" not in c_meta:
                        c_meta["section_path"] = chk["section_path"]
                    text = chk.get("text")
                    if c_id and isinstance(text, str) and text.strip():
                        corpus_docs.append({
                            "id": c_id,
                            "text": text,
                            "metadata": c_meta,
                        })
        except Exception as exc:
            raise VectorStoreError(
                f"Hybrid sparse retrieval could not read document metadata ({type(exc).__name__}: {str(exc).splitlines()[0][:160]})."
            ) from exc

    # Second attempt: Vector store retrieval for selected documents
    if not corpus_docs and eligible_set:
        try:
            where = {"document_id": next(iter(eligible_set))} if len(eligible_set) == 1 else {"document_id": {"$in": list(eligible_set)}}
            if hasattr(vector_store, "collection"):
                coll_data = vector_store.collection.get(where=where, include=["documents", "metadatas"])
                ids = coll_data.get("ids") or []
                docs = coll_data.get("documents") or []
                metas = coll_data.get("metadatas") or []
                for i, c_id in enumerate(ids):
                    text = docs[i] if i < len(docs) else ""
                    if text and text.strip():
                        corpus_docs.append({
                            "id": c_id,
                            "text": text,
                            "metadata": metas[i] if i < len(metas) else {},
                        })
            elif hasattr(vector_store, "get_document"):
                for doc_id in sorted(eligible_set):
                    corpus_docs.extend(vector_store.get_document(doc_id))
        except Exception as exc:
            if dense_results:
                return _tag_dense_fallback(dense_results, strat, f"sparse_chunk_load_failed: {type(exc).__name__}", top_k)
            raise VectorStoreError(
                f"Hybrid sparse retrieval could not load indexed chunks ({type(exc).__name__}: {str(exc).splitlines()[0][:160]})."
            ) from exc

    # If MemoryVectorStore:
    if not corpus_docs and hasattr(vector_store, "_collections"):
        try:
            col = vector_store._collections.get(getattr(vector_store, "collection_name", ""), {})
            for item in col.values():
                if eligible_set and item.get("metadata", {}).get("document_id") not in eligible_set:
                    continue
                corpus_docs.append({
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": dict(item.get("metadata", {})),
                })
        except Exception as exc:
            if dense_results:
                return _tag_dense_fallback(dense_results, strat, f"memory_sparse_load_failed: {type(exc).__name__}", top_k)
            raise VectorStoreError(
                f"Hybrid sparse retrieval could not load indexed chunks ({type(exc).__name__}: {str(exc).splitlines()[0][:160]})."
            ) from exc

    # If corpus_docs is empty (e.g. broad search without pre-inverted index or empty store)
    if not corpus_docs or not any(str(doc.get("text", "")).strip() for doc in corpus_docs):
        if dense_results:
            fallback_reason = "broad_collection_sparse_unindexed" if not eligible_set else "selected_documents_contain_no_extractable_text"
            if on_step:
                on_step("sparse", "skipped")
            return _tag_dense_fallback(dense_results, strat, fallback_reason, top_k)
        return []

    # If corpus_docs is excessively large for on-the-fly BM25 (> 15,000 chunks)
    if len(corpus_docs) > 15000 and dense_results:
        if on_step:
            on_step("sparse", "skipped")
        return _tag_dense_fallback(dense_results, strat, f"corpus_size_{len(corpus_docs)}_exceeds_realtime_sparse_capacity", top_k)

    # Step 3: Run Sparse BM25 Search
    if on_step:
        on_step("sparse", "running")
    bm25 = BM25Index(k1=k1, b=b)
    bm25.index(corpus_docs)
    fetch_sparse_n = max(sparse_top_k, top_k)
    sparse_results = bm25.search(query=query_text, top_k=fetch_sparse_n, k1=k1, b=b)
    if on_step:
        on_step("sparse", "completed")

    if strat in ("bm25", "keyword"):
        return sparse_results[:top_k]

    # Step 4: Hybrid Fusion (RRF or Weighted)
    if on_step:
        on_step("fuse", "running")
    if fusion_method == "weighted":
        fused = weighted_score_fusion(
            dense_candidates=dense_results,
            sparse_candidates=sparse_results,
            alpha=alpha,
            top_k=top_k,
        )
    else:
        # Default to RRF
        fused = reciprocal_rank_fusion(
            rankings=[dense_results, sparse_results],
            k=rrf_k,
            weights=[alpha, 1.0 - alpha],
            top_k=top_k,
        )
    if on_step:
        on_step("fuse", "completed")
    return fused
