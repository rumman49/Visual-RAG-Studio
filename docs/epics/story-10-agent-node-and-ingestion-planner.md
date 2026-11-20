# Story 10 — BMAD Epic 7: Generic Agent Node, Autonomous Ingestion Planner, and Agent UX

**Status:** Implemented, hardened, tested across 24 dedicated regression scenarios (153 total backend tests passing), and verified in live UI.  
**Date:** 2026-10-29  
**Architecture ID:** `agentic-rag`  

---

## 1. Objective and Scope

Extend and harden Visual RAG Studio's **Agentic RAG** architecture and **Autonomous Ingestion Strategy Planner**, preserving all generic node contracts, RagSpark capabilities, existing retrieval architectures, and visual drag-and-drop builder workflows.

Core objectives achieved:
1. **Agent Configuration Interface & UX Enhancements (`frontend/src/VisualBuilder.tsx`):**
   - Replaced single-line Goal and Instructions inputs with accessible, resizable multiline textareas (`minHeight: 75px` / `140px`, default rows 3 / 6, `resize: vertical`).
   - Verbatim newline preservation across saving, reopening, and workflow execution.
   - Dynamic character count and configuration status indicators.
   - Editing preserves all other node configuration and port bindings without resets.
   - Context-rich placeholders and concise descriptions for prompt crafting.
   - Responsive layout preventing text clipping across compact screens.
   - Confirmed end-to-end propagation: complete goal and instructions reach runtime and steer generation.

2. **Adaptive Query-Time Agent Planner (`backend/app/rag/agent.py`):**
   - Structural and lexical query complexity classification (`simple` vs `complex`).
   - Direct lookup queries skip redundant reranking or query decomposition, minimizing latency.
   - Complex multi-faceted queries execute candidate cross-scoring rerank (`RagSparkReranker`) and deep evidence evaluation.
   - When evidence is insufficient, adaptive loop executes `reformulate_query` if permitted and budget allows.
   - Distinct, structured step traces: `planning`, `retrieval`, `rerank`, `evaluate_evidence`, `reformulate_query`, `answer_generation`, and `abstain`.

3. **Autonomous Ingestion Strategy Planner (`backend/app/rag/planner.py`):**
   - Pre-indexing document profile analysis: file type, approximate length, headings, tables, code blocks, sentence boundaries, and language distribution.
   - Heuristic strategy recommendations with confidence calibration.
   - Verified embedding model calibration (384d multilingual, 384d English, 128d fast local).
   - Strict dimension mismatch guardrails: rejects vector collection dimension mismatches with HTTP 409 `dimension_mismatch`.
   - 4 Planning modes: `manual`, `recommendation_only`, `user_approved_auto`, and `fully_autonomous`.

4. **Multi-Strategy Chunking & Incompatibility Validation (`backend/app/rag/chunking.py`):**
   - Parallel execution of multiple chunking strategies (e.g. `recursive` + `hierarchical`).
   - Upfront validation rejecting unsupported chunking strategies with actionable error messages.
   - Explicit incompatibility detection rejecting duplicate redundant strategies and conflicting pairs (e.g. `fixed` + `sentence`).
   - Deduplication policies: `keep_all`, `exact_dedup`, `prefer_structured`, and `cross_strategy_merge`.
   - Per-strategy metrics, timing, and error tracking.

5. **Provider Error Isolation & True Provenance:**
   - Provider failures (e.g. Groq rate limit 429, API quotas, connection errors) are explicitly reported as `provider_error` and never disguised as missing evidence abstentions.
   - Blocked external processing is reported with `authorization_required` while preserving retrieved candidates.
   - Citations strictly resolve to real document chunks with chunk ID, page number, and source snippet.

---

## 2. Architecture & Execution Trace

```
                   ┌──────────────────────────────────────────────┐
                   │        Autonomous Ingestion Planner          │
                   │  (Heuristics, Structure, Language, Vectors)  │
                   └──────────────────────┬───────────────────────┘
                                          │
                                 Ingestion Plan
                                          │
    ┌─────────────────┬───────────────────┴───────────────────┬─────────────────┐
    │                 │                                       │                 │
    ▼                 ▼                                       ▼                 ▼
Recursive       Hierarchical                            Table-Aware         Sentence
Chunker           Chunker                                 Chunker            Chunker
    │                 │                                       │                 │
    └─────────────────┴───────────────────┬───────────────────┴─────────────────┘
                                          │
                             Deduplication & Merging
                        (exact_dedup / prefer_structured)
                                          │
                               Vector Store Indexing
                        (Dimension Mismatch Guardrail: 409)
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │         Agent Orchestrator Engine           │
                   │   Modes: fixed | conditional | bounded_loop │
                   │   Bounds: max_steps, token_budget, timeout   │
                   └──────────────────────┬──────────────────────┘
                                          │
       ┌──────────────────┬───────────────┴───┬──────────────────┬─────────────────┐
       ▼                  ▼                   ▼                  ▼                 ▼
Tool: Planning     Tool: Retrieval     Tool: Rerank       Tool: Evidence    Tool: Answer/Abstain
 (Simple/Complex)   (Hybrid/Dense)     (Cross-Scorer)      (Verification)   (Grounded/Isolated)
```

---

## 3. 16 Regression Test Scenarios & Evidence

All 16 mandatory regression scenarios are implemented in `backend/tests/test_agent_and_ingestion_planner.py`:

| # | Regression Scenario | Description | Status |
|---|---|---|---|
| 1 | `test_scenario_1_multiline_goal_and_instructions_persistence` | Multiline goal and instructions persist verbatim through save and reload | PASSED |
| 2 | `test_scenario_2_custom_instructions_reach_agent_runtime` | Custom instructions reach runtime prompt and steer model execution | PASSED |
| 3 | `test_scenario_3_simple_query_no_unnecessary_reranking` | Direct lookup queries execute without redundant reranking overhead | PASSED |
| 4 | `test_scenario_4_complex_query_triggers_reranking` | Multi-topic comparative queries trigger cross-scorer reranking | PASSED |
| 5 | `test_scenario_5_insufficient_evidence_triggers_reformulation_or_abstention` | Low-evidence queries trigger reformulation or clean localized abstention | PASSED |
| 6 | `test_scenario_6_valid_retrieval_not_discarded_on_step_limit` | Candidate evidence is synthesized even when step budget is reached | PASSED |
| 7 | `test_scenario_7_multi_strategy_chunking_real_implementations` | Real multi-strategy execution (`recursive` + `hierarchical`) with stats | PASSED |
| 8 | `test_scenario_8_incompatible_strategy_combinations_fail_clearly` | Incompatible pairs and duplicate strategies rejected with actionable errors | PASSED |
| 9 | `test_scenario_9_unsupported_chunking_strategies_rejected` | Unsupported chunking strategies rejected upfront before execution | PASSED |
| 10 | `test_scenario_10_embedding_dimension_mismatch_rejected_409` | Vector collection dimension mismatches rejected with HTTP 409 | PASSED |
| 11 | `test_scenario_11_indexing_and_querying_use_intended_collection` | Indexing and querying strictly route to specified target collection | PASSED |
| 12 | `test_scenario_12_provider_failures_reported_as_provider_error` | Provider failures reported as `provider_error`, never disguised as abstention | PASSED |
| 13 | `test_scenario_13_citations_reference_real_chunks_with_metadata` | Citations resolve to actual document IDs, chunk IDs, and text snippets | PASSED |
| 14 | `test_scenario_14_execution_traces_reflect_actual_tool_calls` | Traces record valid tool names, inputs, summaries, and durations | PASSED |
| 15 | `test_scenario_15_agent_loops_stop_at_configured_limits` | Bounded execution halts strictly at timeouts, step limits, and budgets | PASSED |
| 16 | `test_scenario_16_existing_workflows_100_percent_compatible` | All existing workflows (Traditional, Hybrid, Reranking) remain 100% valid | PASSED |

### Verification Metrics
- **Total Backend Tests:** 153 passed (`153 passed, 1 warning in 53.10s`).
- **Dedicated Agent & Ingestion Suite:** 24 passed (`24 passed in 24.36s`).
- **Frontend Test Suite:** 29 passed (`29 passed in 13.08s`).
- **Frontend Production Build:** Clean build (`tsc -b && vite build` in 3.99s, 0 errors).
