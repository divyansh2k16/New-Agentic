"""
Query Agent (RAG-based)

CONCEPT: Retrieval-Augmented Generation for answering natural language questions
about the financial documents.

Key patterns:
- Hybrid retrieval: semantic + keyword search (BM25 + vector)
- Context window management: smart truncation to fit in context
- Hallucination prevention: only answer from retrieved context
- Conversational memory: multi-turn Q&A within a session
- Source citation: every answer cites which document/page
"""
import json
from loguru import logger
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agents.state import FinancialPipelineState
from config.settings import get_settings
from rag.retriever import FinancialRetriever

settings = get_settings()


RAG_SYSTEM = """You are a financial document assistant with access to specific company documents.

STRICT RULES:
1. Answer ONLY from the provided context — never from general knowledge
2. If the answer is not in the context, say "I cannot find this information in the provided documents"
3. Always cite your source: [Document: {filename}, Page: {page}]
4. For numerical answers, quote the exact figure from the document
5. If multiple documents have conflicting data, mention both and explain the difference

CONVERSATION MEMORY:
You have access to the previous Q&A history for this session. Use it for context.
"""


def query_agent_node(state: FinancialPipelineState) -> FinancialPipelineState:
    """
    LangGraph node: Answers natural language questions using RAG.
    This node is only activated when task == 'query'.
    """
    query = state.get("query")
    if not query:
        return {
            **state,
            "query_response": "No query provided.",
            "next_agent": "END",
        }

    logger.info(f"[QUERY AGENT] Processing: '{query}'")

    try:
        retriever = FinancialRetriever()
        retrieved = retriever.retrieve(
            query=query,
            session_id=state["session_id"],
            k=6  # Top-6 chunks
        )
    except Exception as e:
        logger.error(f"[QUERY AGENT] Retrieval failed: {e}")
        retrieved = []

    if not retrieved:
        return {
            **state,
            "query_response": "No relevant context found in the uploaded documents for this query.",
            "retrieved_chunks": [],
            "next_agent": "END",
        }

    # ── Build context block (manage token budget) ────────────────────────────
    context_parts = []
    total_chars = 0
    MAX_CONTEXT_CHARS = 6000  # ~1500 tokens — leave room for response

    for chunk in retrieved:
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        score = chunk.get("score", 0)

        chunk_text = f"[Source: {source}, Page: {page}, Relevance: {score:.2f}]\n{text}\n"
        if total_chars + len(chunk_text) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    context = "\n---\n".join(context_parts)

    # ── Build conversation history (last 3 turns to save tokens) ─────────────
    history_messages = []
    recent_messages = state.get("messages", [])[-6:]  # last 3 Q&A pairs
    for msg in recent_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            history_messages.append(AIMessage(content=content))

    # ── LLM call ─────────────────────────────────────────────────────────────
    llm = ChatAnthropic(
        model=settings.primary_llm_model,
        api_key=settings.anthropic_api_key,
        max_tokens=800,
    )

    messages = [
        SystemMessage(content=RAG_SYSTEM),
        *history_messages,
        HumanMessage(content=f"""CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {query}

Provide a precise, cited answer based strictly on the context above."""),
    ]

    try:
        response = llm.invoke(messages)
        answer = response.content
        input_tokens = response.usage_metadata.get("input_tokens", 0)
        output_tokens = response.usage_metadata.get("output_tokens", 0)
    except Exception as e:
        logger.error(f"[QUERY AGENT] LLM call failed: {e}")
        answer = f"Query processing failed: {str(e)}"
        input_tokens = 0
        output_tokens = 0

    logger.info(f"[QUERY AGENT] Answer generated ({len(answer)} chars)")

    # Append to message history
    new_messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]

    return {
        **state,
        "query_response": answer,
        "retrieved_chunks": retrieved,
        "messages": new_messages,  # operator.add will append these
        "total_input_tokens": state["total_input_tokens"] + input_tokens,
        "total_output_tokens": state["total_output_tokens"] + output_tokens,
        "completed_steps": state["completed_steps"] + ["query"],
        "next_agent": "END",
    }
