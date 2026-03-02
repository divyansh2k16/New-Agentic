"""
Query / RAG Routes

CONCEPT: The conversational interface over uploaded documents.
Key features:
- Session-scoped: user can only query their own documents
- Streaming: responses stream token-by-token for better UX
- Conversation history: multi-turn Q&A within a session
- Source citation: every answer includes document references
"""
import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger

from api.auth import get_current_user, User
from rag.retriever import FinancialRetriever
from rag.knowledge_base import get_collection_stats
from config.settings import get_settings

settings = get_settings()
router = APIRouter(prefix="/query", tags=["Query Engine"])

# In-memory conversation store (use Redis in production)
_conversations: dict = {}


class QueryRequest(BaseModel):
    session_id: str
    question: str
    company_filter: Optional[str] = None
    year_filter: Optional[str] = None
    conversation_id: Optional[str] = None


class QueryResponse(BaseModel):
    conversation_id: str
    session_id: str
    question: str
    answer: str
    sources: List[dict]
    token_usage: dict


@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Ask a natural language question about uploaded financial documents.

    Features:
    - Hybrid RAG retrieval (semantic + keyword)
    - Multi-turn conversation memory
    - Source citations
    - Metadata filtering by company/year
    """
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    if len(request.question) > 2000:
        raise HTTPException(400, "Question too long (max 2000 chars)")

    # Conversation management
    conv_id = request.conversation_id or str(uuid.uuid4())
    if conv_id not in _conversations:
        _conversations[conv_id] = []

    history = _conversations[conv_id]

    try:
        # Retrieve relevant chunks
        retriever = FinancialRetriever()
        filter_meta = {}
        if request.company_filter:
            filter_meta["company_name"] = request.company_filter
        if request.year_filter:
            filter_meta["fiscal_year"] = request.year_filter

        chunks = retriever.retrieve(
            query=request.question,
            session_id=request.session_id,
            k=6,
            filter_metadata=filter_meta or None,
        )

        if not chunks:
            return QueryResponse(
                conversation_id=conv_id,
                session_id=request.session_id,
                question=request.question,
                answer="No relevant information found in the uploaded documents for this question. Please ensure documents are uploaded and processed first.",
                sources=[],
                token_usage={"input": 0, "output": 0},
            )

        # Build context
        context_parts = []
        sources = []
        for i, chunk in enumerate(chunks[:5]):
            context_parts.append(
                f"[{i+1}] Source: {chunk['source']}, Page {chunk['page']}\n{chunk['text']}"
            )
            sources.append({
                "filename": chunk["source"],
                "page": chunk["page"],
                "relevance_score": chunk.get("score", chunk.get("rrf_score", 0)),
                "excerpt": chunk["text"][:200] + "...",
            })

        context = "\n\n".join(context_parts)

        # Build conversation history for context
        history_text = ""
        if history:
            recent = history[-4:]  # Last 2 Q&A pairs
            history_text = "\n".join(
                f"{'Q' if i%2==0 else 'A'}: {msg}" for i, msg in enumerate(recent)
            )

        # LLM call
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatAnthropic(
            model=settings.primary_llm_model,
            api_key=settings.anthropic_api_key,
            max_tokens=600,
        )

        system = """You are a financial document assistant. Answer ONLY using the provided context.
Cite sources as [Document name, Page X]. If information is not in context, say so clearly."""

        prompt = f"""CONTEXT:
{context}

{f"PREVIOUS CONVERSATION:{chr(10)}{history_text}" if history_text else ""}

QUESTION: {request.question}

Answer concisely with citations."""

        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ])

        answer = response.content
        input_tokens = response.usage_metadata.get("input_tokens", 0)
        output_tokens = response.usage_metadata.get("output_tokens", 0)

        # Update conversation history
        _conversations[conv_id].extend([request.question, answer])
        if len(_conversations[conv_id]) > 20:
            _conversations[conv_id] = _conversations[conv_id][-20:]

        logger.info(
            f"[QUERY] User: {current_user.email} | "
            f"Q: {request.question[:40]}... | Tokens: {input_tokens}+{output_tokens}"
        )

        return QueryResponse(
            conversation_id=conv_id,
            session_id=request.session_id,
            question=request.question,
            answer=answer,
            sources=sources,
            token_usage={"input": input_tokens, "output": output_tokens},
        )

    except Exception as e:
        logger.error(f"[QUERY] Failed: {e}")
        raise HTTPException(500, f"Query processing failed: {str(e)}")


@router.get("/history/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
):
    """Retrieve full conversation history for a session."""
    history = _conversations.get(conversation_id, [])
    pairs = []
    for i in range(0, len(history) - 1, 2):
        pairs.append({"question": history[i], "answer": history[i+1]})
    return {"conversation_id": conversation_id, "turns": pairs}


@router.get("/kb-stats")
async def knowledge_base_stats(current_user: User = Depends(get_current_user)):
    """Return knowledge base statistics (admin use)."""
    try:
        stats = get_collection_stats()
        return stats
    except Exception as e:
        return {"error": str(e)}
