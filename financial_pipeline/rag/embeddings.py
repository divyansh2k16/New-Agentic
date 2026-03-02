"""
Embeddings Module

CONCEPT: Embeddings convert text → dense vectors (1536 dimensions typically).
Similar meaning = vectors close together in vector space.
This is what makes semantic search work.

Why sentence-transformers (local) vs OpenAI embeddings:
- Local: no API cost, faster, no data leaving your machine (GDPR-friendly)
- OpenAI: slightly better quality, but costs money and sends data externally

At a bank like Citi: local/private embeddings are MANDATORY for sensitive docs.
"""
from typing import List
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")

from config.settings import get_settings

settings = get_settings()

_model = None  # Module-level cache — load once, use many times


def get_embedding_model() -> "SentenceTransformer":
    """
    Lazy-loads the embedding model.
    Singleton pattern: expensive to load, cheap to reuse.
    """
    global _model
    if _model is None:
        if not HAS_ST:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        logger.info(f"[EMBEDDINGS] Loading model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        logger.info(f"[EMBEDDINGS] Model loaded | Dim: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (each is a list of floats)
    """
    if not texts:
        return []

    model = get_embedding_model()

    # Batch processing — much faster than one-by-one
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalise for cosine similarity
    )

    logger.debug(f"[EMBEDDINGS] Generated {len(embeddings)} embeddings")
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single query for retrieval.
    Slightly different from document embedding for asymmetric retrieval.
    """
    model = get_embedding_model()
    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.tolist()
