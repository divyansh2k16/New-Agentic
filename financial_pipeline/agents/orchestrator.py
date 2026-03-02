"""
Orchestrator — The LangGraph Graph Definition

CONCEPT: This is the brain of the multi-agent system. It defines:
1. All nodes (agents)
2. Conditional edges (routing logic)
3. Entry point and terminal states

WHY LANGGRAPH vs plain function calls?
- Automatic state persistence (resume from any step)
- Built-in human-in-the-loop checkpoints
- Parallel node execution (fan-out/fan-in)
- Visual graph rendering for debugging
- Easy to add/remove agents without rewiring everything

GRAPH FLOW:
  START
    │
    ▼
[GUARDRAIL CHECK] ── REJECT ──► END
    │
    ▼
[CLASSIFIER] ─── all docs classified
    │
    ▼
[EXTRACTOR] ─── all docs extracted
    │
    ▼
[COMPARATOR] ─── if >1 doc ──► compare ──► [SUMMARIZER]
               ─── if 1 doc ─────────────► [SUMMARIZER]
    │
    ▼
[SUMMARIZER]
    │
    ▼
[QUERY AGENT] ─── if query task
    │
    ▼
   END
"""
import uuid
from datetime import datetime
from typing import Literal
from loguru import logger

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from agents.state import FinancialPipelineState
from agents.classifier import classifier_node
from agents.extractor import extractor_node
from agents.comparator import comparator_node
from agents.summarizer import summarizer_node
from agents.query_agent import query_agent_node
from tools.guardrails import guardrail_check_node


def route_after_guardrail(state: FinancialPipelineState) -> Literal["classifier", "END"]:
    """Router: if guardrail passes → classify; else → END."""
    if "guardrail_rejected" in state.get("completed_steps", []):
        return "END"
    return "classifier"


def route_after_extraction(state: FinancialPipelineState) -> Literal["comparator", "summarizer"]:
    """Router: if multiple docs → compare; single doc → summarize."""
    if len(state.get("extractions", [])) > 1:
        return "comparator"
    return "summarizer"


def route_after_summarizer(state: FinancialPipelineState) -> Literal["query_agent", "END"]:
    """Router: if a query was provided → RAG; else → END."""
    if state.get("query") and state.get("task") in ["query", "full_pipeline"]:
        return "query_agent"
    return "END"


def build_graph() -> StateGraph:
    """
    Constructs the LangGraph StateGraph.
    Call compile() on the result to get a runnable graph.
    """
    graph = StateGraph(FinancialPipelineState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    graph.add_node("guardrail", guardrail_check_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("comparator", comparator_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("query_agent", query_agent_node)

    # ── Define edges ──────────────────────────────────────────────────────────
    graph.add_edge(START, "guardrail")

    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"classifier": "classifier", "END": END},
    )

    graph.add_edge("classifier", "extractor")

    graph.add_conditional_edges(
        "extractor",
        route_after_extraction,
        {"comparator": "comparator", "summarizer": "summarizer"},
    )

    graph.add_edge("comparator", "summarizer")

    graph.add_conditional_edges(
        "summarizer",
        route_after_summarizer,
        {"query_agent": "query_agent", "END": END},
    )

    graph.add_edge("query_agent", END)

    return graph


def create_pipeline(use_checkpointing: bool = True):
    """
    Returns a compiled LangGraph pipeline.

    use_checkpointing=True enables MemorySaver — stores state in memory so
    you can resume interrupted pipelines (use SQLite/Redis in production).
    """
    graph = build_graph()

    if use_checkpointing:
        memory = MemorySaver()
        compiled = graph.compile(checkpointer=memory)
    else:
        compiled = graph.compile()

    logger.info("[ORCHESTRATOR] Pipeline compiled successfully")
    return compiled


def make_initial_state(
    document_paths: list,
    task: str = "full_pipeline",
    query: str = None,
    user_id: str = "anonymous",
    session_id: str = None,
) -> FinancialPipelineState:
    """
    Factory function to create a fresh pipeline state.
    Always use this instead of constructing the dict manually.
    """
    return FinancialPipelineState(
        task=task,
        user_id=user_id,
        session_id=session_id or str(uuid.uuid4()),
        query=query,
        document_paths=document_paths,
        current_doc_index=0,
        documents_metadata=[],
        classifications=[],
        extractions=[],
        comparison=None,
        summary=None,
        query_response=None,
        retrieved_chunks=[],
        messages=[],
        next_agent="guardrail",
        errors=[],
        completed_steps=[],
        total_input_tokens=0,
        total_output_tokens=0,
        final_report=None,
    )


# ── Module-level singleton (lazy initialisation) ──────────────────────────────
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = create_pipeline()
    return _pipeline


def run_pipeline(
    document_paths: list,
    task: str = "full_pipeline",
    query: str = None,
    user_id: str = "anonymous",
    session_id: str = None,
) -> FinancialPipelineState:
    """
    High-level entry point. Run the full multi-agent pipeline.

    Args:
        document_paths: List of local file paths to PDF documents
        task: 'full_pipeline' | 'classify' | 'extract' | 'query'
        query: Natural language question for RAG
        user_id: Authenticated user id
        session_id: Session uuid for checkpointing/memory

    Returns:
        Final FinancialPipelineState with all agent outputs populated
    """
    pipeline = get_pipeline()
    state = make_initial_state(document_paths, task, query, user_id, session_id)

    config = {"configurable": {"thread_id": state["session_id"]}}

    logger.info(
        f"[ORCHESTRATOR] Starting pipeline | Task: {task} | "
        f"Docs: {len(document_paths)} | Session: {state['session_id']}"
    )

    final_state = pipeline.invoke(state, config=config)

    logger.info(
        f"[ORCHESTRATOR] Pipeline complete | Steps: {final_state.get('completed_steps')} | "
        f"Tokens in: {final_state.get('total_input_tokens')} | "
        f"Tokens out: {final_state.get('total_output_tokens')}"
    )

    return final_state
