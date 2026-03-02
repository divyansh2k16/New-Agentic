"""
Knowledge Base — Vector Store Management

CONCEPT: The knowledge base is the persistent memory of the RAG system.
Documents are ingested once, chunked, embedded, and stored in ChromaDB.
At query time, we search this store rather than re-processing the PDFs.

ChromaDB is perfect for local development. In production (AWS):
→ Amazon OpenSearch with KNN, or Pinecone, or pgvector on RDS

Key design decisions:
- Session-scoped collections: each user session has isolated document context
- Metadata filtering: filter by company, year, doc_type at retrieval time
- Deduplication via file hash: re-uploading same doc doesn't re-index
"""
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    logger.error("chromadb not installed. Run: pip install chromadb")

from config.settings import get_settings
from rag.embeddings import embed_texts, embed_query
from tools.pdf_loader import extract_text_by_page, chunk_text, get_pdf_metadata

settings = get_settings()

_client = None  # ChromaDB client singleton


def get_chroma_client():
    global _client
    if _client is None:
        if not HAS_CHROMA:
            raise ImportError("Install chromadb: pip install chromadb")
        _client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"[KB] ChromaDB connected at {settings.chroma_persist_dir}")
    return _client


def get_collection(collection_name: str = None):
    """Get or create a ChromaDB collection."""
    client = get_chroma_client()
    name = collection_name or settings.chroma_collection_name
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # Cosine similarity
    )
    return collection


def ingest_document(
    file_path: str,
    session_id: str,
    classification: Dict = None,
    collection_name: str = None,
) -> Dict:
    """
    Ingest a single PDF into the knowledge base.

    Steps:
    1. Extract text page-by-page
    2. Chunk text into overlapping segments
    3. Generate embeddings for each chunk
    4. Store in ChromaDB with rich metadata

    Returns:
        {"chunks_added": int, "doc_id": str, "skipped": bool}
    """
    collection = get_collection(collection_name)
    meta = get_pdf_metadata(file_path)
    file_hash = meta["file_hash"]
    doc_id = f"{session_id}_{file_hash}"

    # Deduplication check
    existing = collection.get(where={"file_hash": file_hash, "session_id": session_id})
    if existing and existing.get("ids"):
        logger.info(f"[KB] Document already indexed: {meta['filename']} (hash: {file_hash})")
        return {"chunks_added": 0, "doc_id": doc_id, "skipped": True}

    # Extract text by page
    pages = extract_text_by_page(file_path)
    if not pages:
        logger.warning(f"[KB] No text extracted from {file_path}")
        return {"chunks_added": 0, "doc_id": doc_id, "skipped": False}

    # Chunk all pages
    all_chunks = []
    for page_data in pages:
        page_chunks = chunk_text(
            text=page_data["text"],
            chunk_size=500,
            overlap=50,
            source=meta["filename"],
            page=page_data["page"],
        )
        all_chunks.extend(page_chunks)

    if not all_chunks:
        logger.warning(f"[KB] No chunks created for {file_path}")
        return {"chunks_added": 0, "doc_id": doc_id, "skipped": False}

    # Generate embeddings (batched)
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    # Build ChromaDB records
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(all_chunks))]
    documents = texts
    metadatas = []
    for i, chunk in enumerate(all_chunks):
        m = {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "session_id": session_id,
            "doc_id": doc_id,
            "file_hash": file_hash,
            "filename": meta["filename"],
        }
        # Add classification metadata if available
        if classification:
            m["company_name"] = classification.get("company_name", "")
            m["fiscal_year"] = classification.get("fiscal_year", "")
            m["doc_type"] = classification.get("doc_type", "")
            m["is_dual_use"] = str(classification.get("is_dual_use_material", False))
        metadatas.append(m)

    # Upsert in batches of 100 (ChromaDB limit)
    BATCH_SIZE = 100
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    logger.info(f"[KB] Ingested {len(all_chunks)} chunks from {meta['filename']}")
    return {"chunks_added": len(all_chunks), "doc_id": doc_id, "skipped": False}


def ingest_documents_batch(
    file_paths: List[str],
    session_id: str,
    classifications: List[Dict] = None,
) -> List[Dict]:
    """Ingest multiple documents for a session."""
    results = []
    for i, path in enumerate(file_paths):
        clf = classifications[i] if classifications and i < len(classifications) else None
        result = ingest_document(path, session_id, clf)
        results.append(result)
    return results


def search(
    query: str,
    session_id: str,
    k: int = 6,
    filter_metadata: Dict = None,
    collection_name: str = None,
) -> List[Dict]:
    """
    Semantic search against the knowledge base.

    Args:
        query: Natural language search query
        session_id: Scope results to this session's documents only
        k: Number of results to return
        filter_metadata: Optional metadata filters (e.g., {"fiscal_year": "2023"})

    Returns:
        List of {"text", "source", "page", "score", "metadata"} dicts
    """
    collection = get_collection(collection_name)
    query_embedding = embed_query(query)

    where_clause = {"session_id": session_id}
    if filter_metadata:
        where_clause.update(filter_metadata)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count()),
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"[KB] Search failed: {e}")
        return []

    if not results["ids"][0]:
        return []

    chunks = []
    for i, doc_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        # Convert cosine distance to similarity score (1 - distance for cosine)
        score = round(1 - distance, 4)
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("filename", ""),
            "page": results["metadatas"][0][i].get("page", 0),
            "score": score,
            "metadata": results["metadatas"][0][i],
        })

    # Sort by score descending
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


def delete_session_documents(session_id: str, collection_name: str = None):
    """Remove all documents for a session (cleanup)."""
    collection = get_collection(collection_name)
    try:
        collection.delete(where={"session_id": session_id})
        logger.info(f"[KB] Deleted all docs for session {session_id}")
    except Exception as e:
        logger.error(f"[KB] Delete failed: {e}")


def get_collection_stats(collection_name: str = None) -> Dict:
    """Return stats about the knowledge base."""
    collection = get_collection(collection_name)
    return {
        "collection_name": collection.name,
        "total_chunks": collection.count(),
        "persist_dir": settings.chroma_persist_dir,
    }
