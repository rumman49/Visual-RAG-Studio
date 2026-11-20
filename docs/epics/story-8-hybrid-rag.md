# Story 8 — BMAD Epic 5: Hybrid RAG Implementation and Verification

**Status:** Implemented, benchmarked, and verified across backend, frontend, and visual canvas interaction.
**Date:** 2026-10-26

## Objective and Scope

Implement and verify Hybrid RAG as the next working architecture in Visual RAG Studio without rebuilding Traditional RAG or replacing the visual builder. Ensure users can build custom pipelines by dragging, configuring, connecting, and disconnecting generic nodes, while retaining grounded answers with full provenance and citations.

---

## Delivered Components

### 1. Hybrid Retrieval Engine (`backend/app/rag/hybrid.py`)
- **Zero Framework Bloat:** Standalone, zero-external-dependency implementation of Okapi BM25 and Reciprocal Rank Fusion (RRF).
- **BM25Index:**
  - Tokenization and inverted term-frequency mapping.
  - Robertson-Spärck Jones IDF with a +1.0 floor to guarantee non-negative weight.
  - Standard term saturation ($k_1$) and document length normalization ($b$).
  - Exact phrase/number boost for numerical benchmarks (e.g. "28.4 BLEU", "4000 warmup steps", "1e-6").
- **Reciprocal Rank Fusion (RRF):**
  - Standard formula: $\text{RRF}(d) = \sum_{m} \frac{w_m}{k + \text{rank}_m(d)}$ with smoothing constant $k$ (default 60).
  - Normalization, metadata preservation (`document_name`, `page_number`, `document_id`, `section_path`), and distance conversion ($1.0 - \text{score}$).
- **Weighted Score Fusion:**
  - Min-max normalized fusion: $\text{Score} = \alpha \cdot \text{DenseScore} + (1 - \alpha) \cdot \text{SparseScore}$.
- **Dual Hybrid Topologies Supported:**
  1. *Composite single-node topology:* `retrieval.dense` configured with `strategy: "hybrid"` or `"rrf"`.
  2. *Branched DAG topology:* `query.input` branching to both `retrieval.dense` and `retrieval.sparse`, both feeding into `merge.fuse` with cardinality `MANY` before `context.assemble`.

### 2. Capability Registry and Generic Node Schemas (`backend/app/registry.py`)
- Added generic node definitions:
  - `retrieval.sparse`: "Sparse BM25 Retrieval" with parameters `top_k`, `k1`, `b`.
  - `merge.fuse`: "Fusion & Ranking (RRF)" with parameters `fusion_method`, `rrf_k`, `dense_weight`, `sparse_weight`, and `cardinality=PortCardinality.MANY`.
- Updated `retrieval.dense` schema with `strategy` (`semantic`, `hybrid`, `bm25`, `rrf`), `fusion_method`, `rrf_k`, `alpha`, `dense_top_k`, `sparse_top_k`, `k1`, `b`.
- Updated priority template:
  - `hybrid-rag` template graph registered and marked `ImplementationStatus.IMPLEMENTED`.

### 3. Pipeline Orchestration & API Routing (`backend/app/rag/pipeline.py` & `routes.py`)
- Extended `answer_query` and `/api/v1/query` to dynamically parse retrieval strategy and fusion settings from the active request graph.
- Executes dense search, BM25 corpus search, rank fusion, optional reranking, context assembly, and citation extraction without modifying underlying storage.

### 4. Canvas Ergonomics and Visual Port Feedback (`frontend/src/VisualBuilder.tsx` & `builder.css`)
- **Connection Feedback:**
  - Added `ConnectionDragContext` tracking active wire drag in React Flow.
  - Valid target ports dynamically glow and pulse bright green (`.port-valid-target`), while incompatible ports dim away (`.port-incompatible`).
  - Active drag handle is highlighted (`.port-active-source`).
- **Disconnection Ergonomics:**
  - Full support for pulling a wire away into empty space (`onReconnectEnd`).
  - Double-click to remove an edge (`onEdgeDoubleClick`).
  - Right-click context menu to remove an edge (`onEdgeContextMenu`).
  - Select + Delete/Backspace key removal.

---

## Benchmark Evaluation Results (`test_hybrid_rag_evaluation.py`)

Evaluation was run across a 10-chunk benchmark corpus and 9 evaluation queries covering 5 categories:
1. Direct factual queries
2. Exact numbers and entities ("28.4 BLEU", "warmup_steps = 4000")
3. Technical terms and identifiers ("LayerNorm(x + Sublayer(x))", "65M parameters")
4. Conceptual multi-passage evidence ("subspace representations in multi-head attention")
5. Unanswerable negative questions (qubit coherence, exoplanet moons)

### Quantitative Metrics

| Retrieval Strategy | Recall@1 | Recall@3 | Recall@5 | Mean Reciprocal Rank (MRR) |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Only** | 0.571 | 0.857 | 1.000 | 0.726 |
| **BM25 Only** | 0.857 | 1.000 | 1.000 | 0.929 |
| **Hybrid RRF** | **0.857** | **1.000** | **1.000** | **0.929** |

**Findings:**
- Hybrid RRF captures exact numbers and named entities that dense search alone misses or places low in rank, while preserving dense semantic relevance.
- On unanswerable queries, no false positive hallucinations were produced, properly falling back to abstention.

---

## Verification Evidence

1. **Backend Tests:**
   - Command: `backend\.venv\Scripts\pytest backend/tests/`
   - Output: **107 passed, 1 warning in 55.38s (100% pass rate)**.
   - Includes full contract checks, health tests, RagSpark adapter safety, Traditional RAG tests, and Hybrid RAG evaluation suite.

2. **Frontend Tests:**
   - Command: `npm --prefix frontend test -- --run`
   - Output: **27 passed (27) across 2 test files (100% pass rate)**.

3. **Production Build:**
   - Command: `npm --prefix frontend run build`
   - Output: `tsc -b && vite build` **succeeded in 8.83s** with zero errors.
