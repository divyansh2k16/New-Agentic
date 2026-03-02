"""
Hybrid Retriever

CONCEPT: Production RAG uses hybrid retrieval:
- Vector search: finds semantically similar text (handles synonyms, paraphrases)
- BM25/keyword search: finds exact term matches (good for specific numbers, names)
- Reranking: re-scores results using a cross-encoder for better accuracy

This two-stage approach (retrieve many → rerank to few) is the industry standard.
At Citi scale: also adds metadata filtering (by document type, fiscal year, company).

Interview insight: "Why hybrid?" → Vector alone misses exact numbers like '34.5 billion'.
Keyword alone misses 'What is the company's profitability?' (needs semantic understanding).
"""
from typing import List, Dict, Optional
from loguru import logger

from rag.knowledge_base import search as vector_search
from config.settings import get_settings

settings = get_settings()


class BM25Retriever:
    """
    Simple in-memory BM25 for keyword matching.
    In production: use Elasticsearch or OpenSearch with BM25 built in.
    """
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self._build_index()

    def _build_index(self):
        """Build inverted index for BM25."""
        self.doc_term_freqs = []
        self.idf = {}
        self.avg_doc_len = 0

        all_terms = []
        for chunk in self.chunks:
            terms = chunk["text"].lower().split()
            self.doc_term_freqs.append({})
            for term in terms:
                self.doc_term_freqs[-1][term] = self.doc_term_freqs[-1].get(term, 0) + 1
            all_terms.extend(set(terms))

        # IDF
        from collections import Counter
        term_doc_freq = Counter(all_terms)
        n = len(self.chunks)
        import math
        for term, df in term_doc_freq.items():
            self.idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1)

        self.avg_doc_len = sum(
            sum(tf.values()) for tf in self.doc_term_freqs
        ) / max(len(self.chunks), 1)

    def search(self, query: str, k: int = 6) -> List[Dict]:
        """BM25 scoring and ranking."""
        import math
        query_terms = query.lower().split()
        k1, b = 1.5, 0.75
        scores = []

        for idx, tf_dict in enumerate(self.doc_term_freqs):
            doc_len = sum(tf_dict.values())
            score = 0.0
            for term in query_terms:
                if term in tf_dict:
                    tf = tf_dict[term]
                    idf = self.idf.get(term, 0)
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len))
                    score += idf * (numerator / denominator)
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        max_score = scores[0][1] if scores else 1
        for idx, score in scores[:k]:
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = round(score / max_score, 4)
            results.append(chunk)
        return results


def reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60,
) -> List[Dict]:
    """
    Reciprocal Rank Fusion (RRF) — combines two ranked lists.
    Score = Σ 1 / (k + rank_i) for each result set.

    This is the standard way to merge vector + keyword results.
    Reference: Cormack et al., 2009.
    """
    scores = {}
    source_map = {}

    for rank, result in enumerate(vector_results):
        key = result.get("text", "")[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        source_map[key] = result

    for rank, result in enumerate(bm25_results):
        key = result.get("text", "")[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        if key not in source_map:
            source_map[key] = result

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused = []
    for key in sorted_keys:
        result = dict(source_map[key])
        result["rrf_score"] = round(scores[key], 6)
        fused.append(result)

    return fused


class FinancialRetriever:
    """
    Main retriever class used by the Query Agent.

    Usage:
        retriever = FinancialRetriever()
        results = retriever.retrieve("What was the net income in 2023?", session_id="abc")
    """

    def retrieve(
        self,
        query: str,
        session_id: str,
        k: int = 6,
        use_hybrid: bool = True,
        filter_metadata: Dict = None,
    ) -> List[Dict]:
        """
        Retrieve relevant chunks using hybrid search + RRF.

        Args:
            query: Natural language question
            session_id: Scope to user's uploaded documents
            k: Final number of results to return
            use_hybrid: If True, combine vector + BM25; if False, vector only
            filter_metadata: Optional ChromaDB metadata filters

        Returns:
            Top-k chunks sorted by relevance
        """
        logger.info(f"[RETRIEVER] Query: '{query[:60]}...' | Session: {session_id}")

        # ── Vector retrieval ─────────────────────────────────────────────────
        vector_k = k * 3  # Over-retrieve, then fuse + rerank
        vector_results = vector_search(
            query=query,
            session_id=session_id,
            k=vector_k,
            filter_metadata=filter_metadata,
        )

        if not vector_results:
            logger.warning("[RETRIEVER] No vector results found")
            return []

        if not use_hybrid or len(vector_results) < 3:
            return vector_results[:k]

        # ── BM25 retrieval ───────────────────────────────────────────────────
        bm25_retriever = BM25Retriever(vector_results)
        bm25_results = bm25_retriever.search(query, k=vector_k)

        # ── Fuse with RRF ────────────────────────────────────────────────────
        fused = reciprocal_rank_fusion(vector_results, bm25_results)

        logger.info(f"[RETRIEVER] Returning {min(k, len(fused))} chunks after hybrid retrieval")
        return fused[:k]

    def retrieve_by_company_year(
        self,
        query: str,
        session_id: str,
        company_name: str = None,
        fiscal_year: str = None,
        k: int = 6,
    ) -> List[Dict]:
        """
        Filtered retrieval: only chunks from a specific company/year.
        Useful when user asks "What was Apple's revenue in 2022?"
        """
        filter_meta = {}
        if company_name:
            filter_meta["company_name"] = company_name
        if fiscal_year:
            filter_meta["fiscal_year"] = fiscal_year

        return self.retrieve(query, session_id, k=k, filter_metadata=filter_meta or None)
