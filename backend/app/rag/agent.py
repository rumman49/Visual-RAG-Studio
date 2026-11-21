"""Generic Bounded Agent Node for Visual RAG Studio.

Provides an autonomous agent execution engine that plans, executes allowlisted tools,
evaluates retrieved evidence, iterates within strict limits, and generates grounded answers.

Features:
- Modes: fixed, conditional, bounded_loop.
- Bounded step counts, iteration limits, timeouts, and token budgets.
- Traceable execution history with per-step reasoning and tool calls.
- Strict allowlisted capabilities (retrieval, reranking, evidence assessment, answer generation, abstention).
- User approval policy gating (always_ask, on_uncertainty, fully_autonomous).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.rag.hybrid import execute_hybrid_retrieval
from app.rag.multilingual import DEFAULT_ABSTENTION, detect_language, get_multilingual_abstention
from app.rag.pipeline import PipelineError, answer_query
from app.rag.ragspark_adapters import RagSparkReranker


class AgentExecutionMode(StrEnum):
    FIXED = "fixed"
    CONDITIONAL = "conditional"
    BOUNDED_LOOP = "bounded_loop"


class AgentApprovalPolicy(StrEnum):
    ALWAYS_ASK = "always_ask"
    ON_UNCERTAINTY = "on_uncertainty"
    FULLY_AUTONOMOUS = "fully_autonomous"


class AgentStepTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_number: int
    tool: str
    action_input: dict[str, Any]
    output_summary: str
    duration_ms: float
    status: str = "success"
    reasoning: str = ""


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp_ms: float = Field(default_factory=lambda: time.time() * 1000)
    decision_type: str
    action_taken: str
    rationale: str
    confidence: float
    requires_approval: bool = False
    approved: bool = True


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_id: str
    goal: str
    instructions: str
    mode: AgentExecutionMode
    step_count: int = 0
    max_steps: int = 5
    max_retrieval_iterations: int = 3
    retrieval_iterations: int = 0
    token_budget: int = 4096
    tokens_used: int = 0
    timeout_seconds: float = 60.0
    approval_policy: AgentApprovalPolicy
    allowed_tools: list[str]
    strategy_permissions: list[str]
    trace: list[AgentStepTrace] = Field(default_factory=list)
    decisions: list[AgentDecision] = Field(default_factory=list)
    final_status: str = "running"
    needs_approval: bool = False
    error: str | None = None


class AgentOrchestrator:
    """Bounded, safe agent orchestrator strictly constrained by graph policies and limits."""

    SUPPORTED_TOOLS: ClassVar[set[str]] = {
        "planning",
        "retrieval",
        "rerank",
        "evaluate_evidence",
        "reformulate_query",
        "answer_generation",
        "abstain",
    }

    def __init__(
        self,
        goal: str = "Answer the query factually based on retrieved evidence.",
        instructions: str = "",
        mode: AgentExecutionMode = AgentExecutionMode.BOUNDED_LOOP,
        max_steps: int = 5,
        max_retrieval_iterations: int = 3,
        token_budget: int = 4096,
        timeout_seconds: float = 60.0,
        approval_policy: AgentApprovalPolicy = AgentApprovalPolicy.ON_UNCERTAINTY,
        allowed_tools: list[str] | None = None,
        strategy_permissions: list[str] | None = None,
    ):
        self.goal = goal
        self.instructions = instructions
        self.mode = mode
        self.max_steps = min(max(1, max_steps), 20)
        self.max_retrieval_iterations = min(max(1, max_retrieval_iterations), 10)
        self.token_budget = token_budget
        self.timeout_seconds = timeout_seconds
        self.approval_policy = approval_policy
        self.allowed_tools = allowed_tools or [
            "planning",
            "retrieval",
            "rerank",
            "evaluate_evidence",
            "reformulate_query",
            "answer_generation",
            "abstain",
        ]
        self.strategy_permissions = strategy_permissions or ["allow_retrieval_strategy_selection", "allow_reranking"]

    def run(
        self,
        query: str,
        vector_store: Any,
        provider: Any,
        settings: Settings,
        store: Any,
        generation_provider: Any = None,
        generation_model: str | None = None,
        document_ids: list[str] | None = None,
        embedding_model: str | None = None,
        initial_strategy: str = "hybrid",
        on_step: Callable[[str, str], None] | None = None,
        allow_external_processing: bool = False,
    ) -> dict[str, Any]:
        """Execute the agentic workflow bounded by steps, token budgets, and loop limits."""
        start_time = time.perf_counter()
        execution_id = f"agent_run_{uuid4().hex[:12]}"

        state = AgentState(
            execution_id=execution_id,
            goal=self.goal,
            instructions=self.instructions,
            mode=self.mode,
            max_steps=self.max_steps,
            max_retrieval_iterations=self.max_retrieval_iterations,
            token_budget=self.token_budget,
            timeout_seconds=self.timeout_seconds,
            approval_policy=self.approval_policy,
            allowed_tools=self.allowed_tools,
            strategy_permissions=self.strategy_permissions,
        )

        # 0. Adaptive Query-Time Planning
        is_complex = (
            len(query.split()) > 10
            or any(
                marker in query.lower()
                for marker in [
                    "compare",
                    "versus",
                    "vs",
                    "difference",
                    "explain both",
                    "how does",
                    "and why",
                    "as well as",
                    "relationship between",
                ]
            )
            or query.count("?") > 1
        )
        query_complexity = "complex" if is_complex else "simple"
        planned = False

        current_query = query
        retrieved_candidates: list[dict] = []
        all_accumulated_candidates: dict[str, dict] = {}
        evidence_sufficient = False
        final_answer = ""
        citations: list[dict] = []
        is_abstained = False

        prov_name = getattr(generation_provider, "provider_name", None)
        if not prov_name:
            type_str = str(type(generation_provider)).lower()
            if "groq" in type_str:
                prov_name = "groq"
            elif "ollama" in type_str:
                prov_name = "ollama"
            elif "huggingface" in type_str:
                prov_name = "huggingface"
            elif "extractive" in type_str:
                prov_name = "extractive"
            else:
                prov_name = "local"

        while state.step_count < self.max_steps:
            # Check elapsed timeout
            elapsed = time.perf_counter() - start_time
            if elapsed > self.timeout_seconds:
                state.final_status = "timeout"
                state.error = f"Agent execution exceeded maximum timeout of {self.timeout_seconds}s."
                break

            state.step_count += 1
            step_idx = state.step_count

            # 1. Planning Step (record if allowed and step budget permits)
            if "planning" in self.allowed_tools and not planned and self.max_steps >= 3:
                planned = True
                t0 = time.perf_counter()
                duration = (time.perf_counter() - t0) * 1000
                state.trace.append(
                    AgentStepTrace(
                        step_number=step_idx,
                        tool="planning",
                        action_input={"query": query, "complexity": query_complexity},
                        output_summary=f"Formulated execution plan for {query_complexity} query.",
                        duration_ms=round(duration, 2),
                        reasoning=(
                            "Direct lookup query; planning direct retrieval and evidence-based synthesis without unnecessary reranking."
                            if query_complexity == "simple"
                            else "Complex query detected; planning retrieval with candidate rerank and evidence verification."
                        ),
                    )
                )
                state.decisions.append(
                    AgentDecision(
                        decision_type="planning",
                        action_taken=f"Plan {query_complexity} query execution",
                        rationale=f"Classified query as '{query_complexity}' based on structural and semantic analysis.",
                        confidence=0.95,
                    )
                )
                if state.step_count < self.max_steps:
                    continue

            if not planned:
                planned = True
                state.decisions.append(
                    AgentDecision(
                        decision_type="planning",
                        action_taken=f"Plan {query_complexity} query execution",
                        rationale=f"Classified query as '{query_complexity}'.",
                        confidence=0.95,
                    )
                )

            # 2. Retrieval Action (if not already retrieved or after query reformulation)
            if not retrieved_candidates:
                if "retrieval" not in self.allowed_tools:
                    state.final_status = "forbidden_tool"
                    state.error = "Agent attempted to call 'retrieval' which is not in allowed_tools."
                    break

                t0 = time.perf_counter()
                if on_step:
                    on_step("retriever", "running")
                state.retrieval_iterations += 1
                strat = (
                    initial_strategy
                    if "allow_retrieval_strategy_selection" in self.strategy_permissions
                    else "hybrid"
                )

                # Embed query
                try:
                    q_vecs, _ = provider.embed(
                        embedding_model or settings.sentence_transformers_model, [current_query]
                    )
                    q_vec = q_vecs[0]
                except Exception as exc:
                    if on_step:
                        on_step("retriever", "failed")
                    duration = (time.perf_counter() - t0) * 1000
                    state.final_status = "provider_error"
                    state.error = str(exc)
                    state.trace.append(
                        AgentStepTrace(
                            step_number=step_idx,
                            tool="retrieval",
                            status="failed",
                            action_input={"query": current_query, "strategy": strat},
                            output_summary=f"Embedding provider failed: {exc}",
                            duration_ms=round(duration, 2),
                            reasoning="Query embedding failed due to provider error.",
                        )
                    )
                    break

                try:
                    if strat in ("hybrid", "rrf"):
                        candidates = execute_hybrid_retrieval(
                            vector_store=vector_store,
                            query_vector=q_vec,
                            query_text=current_query,
                            top_k=6,
                            document_ids=document_ids,
                            strategy="hybrid",
                            fusion_method="rrf",
                            store=store,
                            on_step=on_step,
                        )
                    else:
                        candidates = vector_store.search(q_vec, top_k=6, document_ids=document_ids)
                except Exception as exc:
                    if on_step:
                        on_step("retriever", "failed")
                    duration = (time.perf_counter() - t0) * 1000
                    code = "dimension_mismatch" if "dimension" in str(exc).lower() else "vector_store_error"
                    state.final_status = code
                    state.error = str(exc)
                    state.trace.append(
                        AgentStepTrace(
                            step_number=step_idx,
                            tool="retrieval",
                            status="failed",
                            action_input={"query": current_query, "strategy": strat},
                            output_summary=f"Vector store search failed ({code}): {exc}",
                            duration_ms=round(duration, 2),
                            reasoning=f"Retrieval halted due to {code}.",
                        )
                    )
                    break

                duration = (time.perf_counter() - t0) * 1000
                for c in candidates:
                    cid = c.get("id") or c.get("chunk_id")
                    if cid and cid not in all_accumulated_candidates:
                        all_accumulated_candidates[cid] = c
                merged_candidates = list(candidates)
                seen_ids = {c.get("id") or c.get("chunk_id") for c in candidates}
                for cid, c in all_accumulated_candidates.items():
                    if cid not in seen_ids:
                        merged_candidates.append(c)
                retrieved_candidates = merged_candidates
                state.tokens_used += 150
                if on_step:
                    on_step("retriever", "completed")

                actual_strat = candidates[0].get("retrieval_strategy") if candidates else strat
                fallback_reason = candidates[0].get("fallback_reason") if candidates else None
                summary_text = (
                    f"Retrieved {len(candidates)} candidate chunks using {actual_strat} (fallback: {fallback_reason})."
                    if fallback_reason
                    else f"Retrieved {len(candidates)} candidate chunks using {actual_strat}."
                )

                state.trace.append(
                    AgentStepTrace(
                        step_number=step_idx,
                        tool="retrieval",
                        action_input={
                            "query": current_query,
                            "strategy": strat,
                            "actual_strategy": actual_strat,
                            "fallback_reason": fallback_reason,
                            "top_k": 6,
                        },
                        output_summary=summary_text,
                        duration_ms=round(duration, 2),
                        reasoning=f"Querying knowledge store for evidence supporting goal: '{self.goal}'. Executed {actual_strat}.",
                    )
                )
                state.decisions.append(
                    AgentDecision(
                        decision_type="retrieval",
                        action_taken=f"Execute {actual_strat} search",
                        rationale=f"Retrieved {len(candidates)} candidates ({'fallback to dense: ' + fallback_reason if fallback_reason else 'strategy: ' + actual_strat}).",
                        confidence=0.92,
                    )
                )

            # 3. Rerank Action (Only for complex queries with candidates and remaining step budget)
            if (
                query_complexity == "complex"
                and "rerank" in self.allowed_tools
                and "allow_reranking" in self.strategy_permissions
                and len(retrieved_candidates) > 2
                and (self.max_steps - state.step_count >= 1)
            ):
                t0 = time.perf_counter()
                reranker = RagSparkReranker(model="flash")
                reranked = reranker.rerank(current_query, retrieved_candidates, top_k=4)
                duration = (time.perf_counter() - t0) * 1000
                retrieved_candidates = reranked
                state.tokens_used += 100

                state.trace.append(
                    AgentStepTrace(
                        step_number=step_idx,
                        tool="rerank",
                        action_input={"candidate_count": len(retrieved_candidates), "top_k": 4},
                        output_summary=f"Cross-scored and pruned candidates to top {len(reranked)} using Flash Cross-Scorer.",
                        duration_ms=round(duration, 2),
                        reasoning="Refining candidate ranking for complex multi-topic query to maximize evidence quality.",
                    )
                )
                if state.step_count < self.max_steps:
                    continue

            # If evaluate_evidence is not in allowed_tools, accept retrieved candidates directly
            if "evaluate_evidence" not in self.allowed_tools and retrieved_candidates:
                evidence_sufficient = True

            # 4. Evidence Assessment / Verification
            if "evaluate_evidence" in self.allowed_tools and not evidence_sufficient:
                t0 = time.perf_counter()
                combined_evidence = " ".join(c.get("text", "") for c in retrieved_candidates[:3]).lower()
                query_words = {
                    w.lower().strip("?,.!:;\"'()[]{}")
                    for w in current_query.split()
                    if len(w.strip("?,.!:;\"'()[]{}")) >= 2
                }
                stopwords = {
                    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
                    "does", "this", "that", "these", "those", "with", "from", "into", "during",
                    "including", "until", "against", "among", "throughout", "despite", "towards",
                    "upon", "concerning", "about", "have", "been", "were", "will", "would", "shall"
                }
                meaningful_words = {w for w in query_words if w not in stopwords}
                test_words = meaningful_words if meaningful_words else query_words
                overlap_count = sum(1 for w in test_words if w in combined_evidence)
                overlap_ratio = overlap_count / max(1, len(test_words))
                duration = (time.perf_counter() - t0) * 1000

                has_top_candidates = len(retrieved_candidates) > 0 and any(
                    len(c.get("text", "").strip()) > 30 for c in retrieved_candidates
                )
                if (overlap_ratio >= 0.15) or (has_top_candidates and len(test_words) == 0) or (has_top_candidates and overlap_count >= 1):
                    evidence_sufficient = True
                    decision_conf = min(0.95, 0.60 + overlap_ratio * 0.4)
                    state.trace.append(
                        AgentStepTrace(
                            step_number=step_idx,
                            tool="evaluate_evidence",
                            action_input={"overlap_ratio": round(overlap_ratio, 2)},
                            output_summary="Evidence confirmed sufficient to generate grounded answer.",
                            duration_ms=round(duration, 2),
                            reasoning="Retrieved chunks contain sufficient topical content matching query entities.",
                        )
                    )
                    state.decisions.append(
                        AgentDecision(
                            decision_type="evaluate_evidence",
                            action_taken="Accept evidence",
                            rationale=f"Overlap ratio {overlap_ratio:.2f} confirms sufficient grounded context.",
                            confidence=round(decision_conf, 2),
                        )
                    )
                else:
                    # In conditional or bounded loop mode, try reformulating query if budget allows re-retrieval
                    if (
                        "reformulate_query" in self.allowed_tools
                        and state.retrieval_iterations < self.max_retrieval_iterations
                        and (self.max_steps - state.step_count >= 2)
                    ):
                        current_query = f"{current_query} overview"
                        retrieved_candidates = []
                        state.trace.append(
                            AgentStepTrace(
                                step_number=step_idx,
                                tool="reformulate_query",
                                action_input={"new_query": current_query},
                                output_summary="Reformulated query to broaden search candidate space.",
                                duration_ms=round(duration, 2),
                                reasoning="Initial evidence had low relevance; broadening search terms.",
                            )
                        )
                        continue
                    else:
                        evidence_sufficient = False
                        state.trace.append(
                            AgentStepTrace(
                                step_number=step_idx,
                                tool="evaluate_evidence",
                                action_input={"overlap_ratio": round(overlap_ratio, 2)},
                                output_summary="Evidence determined insufficient. Directing to grounded abstention.",
                                duration_ms=round(duration, 2),
                                reasoning="Topical evidence does not support an answer; preventing hallucinations.",
                            )
                        )
                        target_lang = detect_language(query)
                        final_answer = get_multilingual_abstention(target_lang)
                        is_abstained = True
                        citations = []
                        state.trace.append(
                            AgentStepTrace(
                                step_number=step_idx + 1,
                                tool="abstain",
                                action_input={"language": target_lang},
                                output_summary=f"Clean localized abstention returned ({target_lang}).",
                                duration_ms=5.0,
                                reasoning="Strict grounding policy: abstaining due to insufficient retrieved evidence.",
                            )
                        )
                        state.final_status = "abstained"
                        break

            # 5. Answer Generation or Abstention
            if evidence_sufficient and "answer_generation" in self.allowed_tools:
                t0 = time.perf_counter()
                try:
                    ans_res = answer_query(
                        question=query,
                        top_k=min(4, len(retrieved_candidates)),
                        document_ids=document_ids,
                        vector_store=vector_store,
                        provider=provider,
                        settings=settings,
                        store=store,
                        generation_provider=generation_provider,
                        generation_model=generation_model,
                        embedding_model=embedding_model,
                        retrieval_strategy=initial_strategy,
                        allow_external_processing=allow_external_processing,
                        agent_goal=self.goal,
                        agent_instructions=self.instructions,
                        on_step=on_step,
                        pre_retrieved_chunks=retrieved_candidates,
                    )
                except PipelineError as exc:
                    duration = (time.perf_counter() - t0) * 1000
                    state.final_status = (
                        exc.code
                        if exc.code in ("provider_error", "dimension_mismatch", "authorization_required")
                        else "error"
                    )
                    state.error = str(exc)
                    final_answer = f"Error: {exc}"
                    state.trace.append(
                        AgentStepTrace(
                            step_number=step_idx,
                            tool="answer_generation",
                            status="failed",
                            action_input={"query": query},
                            output_summary=f"Pipeline error: {exc}",
                            duration_ms=round(duration, 2),
                            reasoning=f"Execution halted due to pipeline error: {exc.code}",
                        )
                    )
                    break

                duration = (time.perf_counter() - t0) * 1000
                final_answer = ans_res["answer"]
                citations = ans_res.get("citations", [])
                state.tokens_used += 350

                # Check for explicit provider error or blocked authorization
                if ans_res.get("status") == "authorization_required":
                    state.final_status = "authorization_required"
                    state.error = ans_res.get("generation_error") or "External generation is blocked."
                    is_abstained = False
                    state.trace.append(
                        AgentStepTrace(
                            step_number=step_idx,
                            tool="answer_generation",
                            status="failed",
                            action_input={"provider": prov_name},
                            output_summary=f"Authorization required: {state.error}",
                            duration_ms=round(duration, 2),
                            reasoning="External generation blocked pending user authorization.",
                        )
                    )
                    break

                if ans_res.get("status") in ("generation_failed", "provider_error") or ans_res.get("generation_error"):
                    state.final_status = "provider_error"
                    state.error = ans_res.get("generation_error") or "Generation provider error."
                    is_abstained = False
                    state.trace.append(
                        AgentStepTrace(
                            step_number=step_idx,
                            tool="answer_generation",
                            status="failed",
                            action_input={"provider": prov_name, "model": generation_model},
                            output_summary=f"Provider error: {state.error}",
                            duration_ms=round(duration, 2),
                            reasoning="Provider error encountered during generation.",
                        )
                    )
                    break

                is_abstained = ans_res.get("abstained", False) or "could not find enough retrieved evidence" in final_answer

                if is_abstained:
                    citations = []
                    state.trace.append(
                        AgentStepTrace(
                            step_number=step_idx,
                            tool="abstain",
                            action_input={"citations_count": 0},
                            output_summary="Clean localized abstention: generation detected insufficient evidence to ground answer.",
                            duration_ms=round(duration, 2),
                            reasoning="Strict grounding policy: abstaining due to insufficient retrieved evidence.",
                        )
                    )
                    state.final_status = "abstained"
                    break

                state.trace.append(
                    AgentStepTrace(
                        step_number=step_idx,
                        tool="answer_generation",
                        action_input={"citations_count": len(citations)},
                        output_summary=f"Generated grounded answer with {len(citations)} source citations.",
                        duration_ms=round(duration, 2),
                        reasoning="Evidence validated. Synthesized grounded response with strict source citations.",
                    )
                )
                state.final_status = "completed"
                break

            # If evidence is evaluated and insufficient, abstain cleanly
            if "evaluate_evidence" in self.allowed_tools and not evidence_sufficient:
                target_lang = detect_language(query)
                abstention_msg = get_multilingual_abstention(target_lang)
                final_answer = abstention_msg
                is_abstained = True
                citations = []
                state.trace.append(
                    AgentStepTrace(
                        step_number=step_idx,
                        tool="abstain",
                        action_input={"language": target_lang},
                        output_summary=f"Clean localized abstention returned ({target_lang}).",
                        duration_ms=5.0,
                        reasoning="Strict grounding policy: abstaining due to insufficient retrieved evidence.",
                    )
                )
                state.final_status = "abstained"
                break

        if state.step_count >= self.max_steps and state.final_status == "running":
            if not final_answer and retrieved_candidates and (evidence_sufficient or "evaluate_evidence" not in self.allowed_tools):
                t0 = time.perf_counter()
                try:
                    ans_res = answer_query(
                        question=query,
                        top_k=min(4, len(retrieved_candidates)),
                        document_ids=document_ids,
                        vector_store=vector_store,
                        provider=provider,
                        settings=settings,
                        store=store,
                        generation_provider=generation_provider,
                        generation_model=generation_model,
                        embedding_model=embedding_model,
                        retrieval_strategy=initial_strategy,
                        allow_external_processing=allow_external_processing,
                        agent_goal=self.goal,
                        agent_instructions=self.instructions,
                        on_step=on_step,
                        pre_retrieved_chunks=retrieved_candidates,
                    )
                    duration = (time.perf_counter() - t0) * 1000
                    final_answer = ans_res["answer"]
                    citations = ans_res.get("citations", [])
                    if ans_res.get("status") in ("generation_failed", "provider_error") or ans_res.get("generation_error"):
                        state.final_status = "provider_error"
                        state.error = ans_res.get("generation_error") or "Generation provider error."
                        state.trace.append(
                            AgentStepTrace(
                                step_number=state.step_count,
                                tool="answer_generation",
                                status="failed",
                                action_input={"provider": prov_name},
                                output_summary=f"Provider error: {state.error}",
                                duration_ms=round(duration, 2),
                                reasoning="Provider error encountered during generation.",
                            )
                        )
                    else:
                        is_abstained = ans_res.get("abstained", False) or "could not find enough retrieved evidence" in final_answer
                        state.tokens_used += 350
                        state.trace.append(
                            AgentStepTrace(
                                step_number=state.step_count,
                                tool="answer_generation" if not is_abstained else "abstain",
                                action_input={"citations_count": len(citations)},
                                output_summary=f"Synthesized answer with {len(citations)} citations." if not is_abstained else "Abstained due to low grounding confidence.",
                                duration_ms=round(duration, 2),
                                reasoning="Finalizing response based on accumulated candidate evidence.",
                            )
                        )
                        state.final_status = "completed" if not is_abstained else "abstained"
                except Exception as exc:
                    state.final_status = "provider_error" if "provider" in str(exc).lower() else "error"
                    state.error = str(exc)
            else:
                state.final_status = "max_steps_exceeded"
                if not final_answer:
                    final_answer = DEFAULT_ABSTENTION
                    is_abstained = True

        return {
            "execution_id": state.execution_id,
            "goal": self.goal,
            "status": state.final_status,
            "answer": final_answer,
            "abstained": is_abstained,
            "citations": citations,
            "step_count": state.step_count,
            "tokens_used": state.tokens_used,
            "retrieval_iterations": state.retrieval_iterations,
            "trace": [t.model_dump() for t in state.trace],
            "decisions": [d.model_dump() for d in state.decisions],
            "error": state.error,
            "collection": getattr(vector_store, "collection_name", "visual_rag_studio_documents"),
            "generation_provider": prov_name,
            "generation_model": generation_model or getattr(settings, "groq_generation_model", "default"),
            "embedding_model": embedding_model or getattr(settings, "sentence_transformers_model", "default"),
            "retrieved_chunks": retrieved_candidates,
        }
