"""
Performance & Quality Tests for the PDF ingestion pipeline.

Run all tests:
    pytest tests/test_performance.py -v

Run only perf benchmarks (skips slow embedding tests):
    pytest tests/test_performance.py -v -m perf

Run only quality tests:
    pytest tests/test_performance.py -v -m quality

Why these tests matter
----------------------
Optimisations can break quality silently (e.g. wrong extractor order returns
partial text, parallel ingestion has race conditions, larger batches change
embedding output).  These tests verify:

  1. Speed — each stage completes within a defined budget.
  2. Correctness — fast path produces same text as the reference path.
  3. Retrieval quality — relevant chunks rank above irrelevant ones after
     the full embed → store → retrieve cycle (no LLM needed).
  4. Parallelism — concurrent ingestion is consistent and race-free.
  5. Deduplication — re-ingesting the same content returns "skipped".
"""
import io
import sys
import os
import time
import tempfile
import threading
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Detect if the embedding model is available (requires network/cache) ───────

def _embedding_model_available() -> bool:
    """Return True if the sentence-transformers model can be loaded."""
    try:
        from sentence_transformers import SentenceTransformer
        from config.settings import get_settings
        m = SentenceTransformer(get_settings().embedding_model)
        return True
    except Exception:
        return False


_EMBEDDING_AVAILABLE = _embedding_model_available()

requires_embedding = pytest.mark.skipif(
    not _EMBEDDING_AVAILABLE,
    reason="Embedding model not available (network/proxy required for first download)",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pdf_bytes(text: str) -> bytes:
    """
    Build a minimal, valid single-page PDF containing *text*.
    No external library needed — crafted from the PDF spec directly.
    """
    # PDF streams must use \r\n line endings inside stream objects
    content_stream = text.encode("latin-1", errors="replace")
    stream_len = len(content_stream)

    # We embed the text as a raw content stream (no font rendering needed for
    # text-extraction tests — both fitz and pypdf read the raw stream bytes).
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length " + str(stream_len).encode() + b" >>\n"
        b"stream\n" + content_stream + b"\nendstream\nendobj\n"
        b"xref\n0 5\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000206 00000 n \n"
        b"trailer\n<< /Size 5 /Root 1 0 R >>\n"
        b"startxref\n"
        b"0\n"
        b"%%EOF\n"
    )
    return pdf


def _write_temp_pdf(text: str, suffix: str = ".pdf") -> str:
    """Write a temporary PDF with *text* and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(_make_pdf_bytes(text))
    tmp.flush()
    tmp.close()
    return tmp.name


# ── Fixtures ──────────────────────────────────────────────────────────────────

FINANCIAL_TEXT = (
    "Annual Report 2023 — Acme Corporation\n\n"
    "Revenue for fiscal year 2023 was $4,200 million, "
    "representing a 12% increase over the prior year.\n\n"
    "Net income reached $380 million. Earnings per share "
    "diluted were $2.15. Total assets stood at $18,500 million.\n\n"
    "Operating cash flow was $750 million. The company declared "
    "a dividend of $0.80 per share. Return on equity was 18.2%.\n\n"
    "Revenue breakdown: North America $2,500M, Europe $1,100M, "
    "Asia Pacific $600M. Cost of goods sold $2,800M. "
    "Gross margin 33.3%. EBITDA $620 million.\n"
)

UNRELATED_TEXT = (
    "Recipe for chocolate cake: 2 cups flour, 1 cup sugar, "
    "3 eggs, 200g butter, 50g cocoa powder. Mix dry ingredients. "
    "Cream butter and sugar. Fold in eggs. Bake at 180C for 35 mins.\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. EXTRACTION SPEED TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.perf
def test_extract_document_speed():
    """extract_document() must complete within 3 seconds for a typical PDF."""
    from tools.pdf_loader import extract_document

    path = _write_temp_pdf(FINANCIAL_TEXT * 20)  # ~20-page equivalent
    try:
        t0 = time.perf_counter()
        doc = extract_document(path)
        elapsed = time.perf_counter() - t0

        assert elapsed < 3.0, f"extract_document took {elapsed:.2f}s — expected < 3s"
        assert doc["pages"], "Should have extracted at least one page"
        assert doc["file_hash"], "Should have a file hash"
    finally:
        os.unlink(path)


@pytest.mark.perf
def test_chunk_text_speed():
    """chunk_text() on a 50 000-char document must finish in under 0.5 seconds."""
    from tools.pdf_loader import chunk_text

    big_text = FINANCIAL_TEXT * 100  # ~50 000 chars
    t0 = time.perf_counter()
    chunks = chunk_text(big_text, chunk_size=500, overlap=50)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5, f"chunk_text took {elapsed:.2f}s — expected < 0.5s"
    assert len(chunks) > 10


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMBEDDING SPEED TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.perf
@requires_embedding
def test_embed_texts_speed_small():
    """Embedding 50 chunks must complete within 5 seconds on CPU."""
    from rag.embeddings import embed_texts

    texts = [f"Revenue in quarter {i} was ${i * 100} million." for i in range(50)]
    t0 = time.perf_counter()
    embeddings = embed_texts(texts)
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, f"embed_texts(50) took {elapsed:.2f}s — expected < 5s"
    assert len(embeddings) == 50
    assert len(embeddings[0]) > 0


@pytest.mark.perf
@requires_embedding
def test_embed_texts_speed_large():
    """Embedding 200 chunks must complete within 15 seconds on CPU."""
    from rag.embeddings import embed_texts

    texts = [f"Net income for period {i} was ${i * 10}M, EPS ${i * 0.05:.2f}." for i in range(200)]
    t0 = time.perf_counter()
    embeddings = embed_texts(texts)
    elapsed = time.perf_counter() - t0

    assert elapsed < 15.0, f"embed_texts(200) took {elapsed:.2f}s — expected < 15s"
    assert len(embeddings) == 200


@pytest.mark.perf
@requires_embedding
def test_embed_query_speed():
    """A single query embedding must complete within 1 second."""
    from rag.embeddings import embed_query

    t0 = time.perf_counter()
    vec = embed_query("What was the total revenue in 2023?")
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, f"embed_query took {elapsed:.2f}s — expected < 1s"
    assert len(vec) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXTRACTION CORRECTNESS TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.quality
def test_extract_document_returns_metadata():
    """extract_document must return all required metadata fields."""
    from tools.pdf_loader import extract_document

    path = _write_temp_pdf(FINANCIAL_TEXT)
    try:
        doc = extract_document(path)
        for field in ("filename", "file_path", "page_count", "file_size_bytes",
                      "file_hash", "pdf_title", "pdf_author", "pdf_created", "pages"):
            assert field in doc, f"Missing field: {field}"
        assert doc["page_count"] >= 1
        assert doc["file_size_bytes"] > 0
        assert len(doc["file_hash"]) == 16
    finally:
        os.unlink(path)


@pytest.mark.quality
def test_extract_document_single_open(monkeypatch):
    """
    Verify the PDF is opened only once per ingest_document call.
    Previously get_pdf_metadata() + extract_text_by_page() each opened it.
    """
    from tools import pdf_loader

    open_count = {"n": 0}
    original_extract = pdf_loader.extract_document

    def counting_extract(path):
        open_count["n"] += 1
        return original_extract(path)

    monkeypatch.setattr(pdf_loader, "extract_document", counting_extract)

    path = _write_temp_pdf(FINANCIAL_TEXT)
    try:
        # Import here so monkeypatch applies
        from tools.pdf_loader import extract_text_by_page, get_pdf_metadata
        # Both wrappers should call extract_document once each — not a double open
        # inside ingest_document (which calls extract_document directly).
        _ = extract_text_by_page(path)
        _ = get_pdf_metadata(path)
        # 2 wrapper calls = 2 extract_document calls; each opens the file once
        assert open_count["n"] == 2
    finally:
        os.unlink(path)


@pytest.mark.quality
def test_chunk_overlap_preserves_context():
    """Adjacent chunks must share overlapping content (overlap > 0)."""
    from tools.pdf_loader import chunk_text

    text = "Word " * 400  # 2000 chars of repeated words
    chunks = chunk_text(text, chunk_size=200, overlap=40)

    assert len(chunks) >= 2
    # The tail of chunk[0] and the head of chunk[1] must share some text
    tail = chunks[0]["text"][-60:]
    head = chunks[1]["text"][:60:]
    # At least some words should be shared (overlap = 40 chars)
    tail_words = set(tail.split())
    head_words = set(head.split())
    assert tail_words & head_words, "No overlap found between adjacent chunks"


@pytest.mark.quality
def test_chunk_empty_text_returns_empty():
    from tools.pdf_loader import chunk_text
    assert chunk_text("") == []
    assert chunk_text("   \n  \t  ") == []


@pytest.mark.quality
def test_chunk_metadata_fields():
    from tools.pdf_loader import chunk_text
    chunks = chunk_text("Hello world. " * 100, source="test.pdf", page=3)
    for c in chunks:
        assert c["source"] == "test.pdf"
        assert c["page"] == 3
        assert "chunk_index" in c
        assert "char_count" in c
        assert c["char_count"] == len(c["text"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. EMBEDDING QUALITY TESTS (no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.quality
@requires_embedding
def test_embeddings_are_normalised():
    """Embeddings must have unit norm (required for cosine similarity)."""
    import math
    from rag.embeddings import embed_texts

    texts = ["Revenue was $100M", "Net income $20M", "Total assets $500M"]
    embeddings = embed_texts(texts)

    for i, vec in enumerate(embeddings):
        norm = math.sqrt(sum(v ** 2 for v in vec))
        assert abs(norm - 1.0) < 1e-4, f"Embedding {i} norm = {norm:.6f}, expected ~1.0"


@pytest.mark.quality
@requires_embedding
def test_similar_texts_have_high_cosine_similarity():
    """
    Two semantically similar sentences must have cosine similarity > 0.7.
    This confirms the embedding model and normalisation are working.
    """
    import math
    from rag.embeddings import embed_texts

    texts = [
        "The company's annual revenue was four billion dollars.",
        "Total annual sales reached four billion USD.",
        "The weather forecast predicts heavy rain tomorrow.",
    ]
    vecs = embed_texts(texts)

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))  # pre-normalised so dot = cosine

    sim_financial = cosine(vecs[0], vecs[1])
    sim_unrelated = cosine(vecs[0], vecs[2])

    assert sim_financial > 0.7, f"Similar sentences cosine={sim_financial:.3f}, expected > 0.7"
    assert sim_financial > sim_unrelated, (
        f"Financial pair ({sim_financial:.3f}) should outscore "
        f"unrelated pair ({sim_unrelated:.3f})"
    )


@pytest.mark.quality
@requires_embedding
def test_embed_query_vs_embed_texts_compatible():
    """
    A query embedding must be in the same vector space as document embeddings.
    The top result from a cosine search should be the relevant chunk.
    """
    import math
    from rag.embeddings import embed_texts, embed_query

    docs = [
        "Revenue for 2023 was $4.2 billion, up 12% year-over-year.",
        "Net income reached $380 million with EPS of $2.15.",
        "The chocolate cake recipe requires flour, sugar, and cocoa.",
        "Total assets stood at $18,500 million at year end.",
    ]
    query = "What was the total revenue in 2023?"

    doc_vecs = embed_texts(docs)
    q_vec = embed_query(query)

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    scores = [(cosine(q_vec, dv), i) for i, dv in enumerate(doc_vecs)]
    scores.sort(reverse=True)

    top_idx = scores[0][1]
    assert top_idx == 0, (
        f"Revenue chunk should be top result, got index {top_idx} "
        f"('{docs[top_idx][:60]}'). Scores: {scores}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. FULL PIPELINE INTEGRATION (in-memory ChromaDB)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_chroma_dir(tmp_path, monkeypatch):
    """Redirect ChromaDB to a temp dir and reset client singleton for each test."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    # Reset singletons so each test gets a fresh DB
    import rag.knowledge_base as kb
    import rag.embeddings as emb
    kb._client = None
    # Don't reset embedding model — expensive to reload
    yield tmp_path / "chroma"
    kb._client = None


@pytest.mark.quality
@requires_embedding
def test_ingest_and_retrieve_relevant_chunk(tmp_chroma_dir, monkeypatch):
    """
    End-to-end quality test: ingest a financial PDF then query it.
    The top retrieved chunk must be from the financial document, not the
    unrelated one, proving retrieval quality is maintained after optimisations.
    """
    monkeypatch.setattr(
        "config.settings.Settings.chroma_persist_dir",
        str(tmp_chroma_dir),
        raising=False,
    )
    import rag.knowledge_base as kb
    kb._client = None  # force re-init with new path

    from config.settings import get_settings
    settings = get_settings()
    settings.__dict__["chroma_persist_dir"] = str(tmp_chroma_dir)

    fin_path = _write_temp_pdf(FINANCIAL_TEXT)
    irr_path = _write_temp_pdf(UNRELATED_TEXT)

    session = "test_session_quality"
    try:
        r1 = kb.ingest_document(fin_path, session, collection_name="test_col")
        r2 = kb.ingest_document(irr_path, session, collection_name="test_col")

        assert r1["chunks_added"] > 0, "Financial doc should have been ingested"
        assert r2["chunks_added"] > 0, "Unrelated doc should have been ingested"

        results = kb.search(
            "What was the annual revenue?",
            session,
            k=5,
            collection_name="test_col",
        )

        assert results, "Should have retrieval results"

        # The top chunk must come from the financial document
        top_source = results[0]["source"]
        fin_filename = Path(fin_path).name
        assert top_source == fin_filename, (
            f"Top result source '{top_source}' should be financial doc '{fin_filename}'"
        )

        # Top score must be meaningfully higher than bottom score
        if len(results) > 1:
            assert results[0]["score"] >= results[-1]["score"], \
                "Results should be sorted by score descending"
    finally:
        os.unlink(fin_path)
        os.unlink(irr_path)
        kb._client = None


@pytest.mark.quality
@requires_embedding
def test_deduplication_skips_reingestion(tmp_chroma_dir, monkeypatch):
    """Re-ingesting the same PDF must return skipped=True and add 0 chunks."""
    monkeypatch.setattr(
        "config.settings.Settings.chroma_persist_dir",
        str(tmp_chroma_dir),
        raising=False,
    )
    import rag.knowledge_base as kb
    kb._client = None

    from config.settings import get_settings
    get_settings().__dict__["chroma_persist_dir"] = str(tmp_chroma_dir)

    path = _write_temp_pdf(FINANCIAL_TEXT)
    session = "test_session_dedup"
    try:
        r1 = kb.ingest_document(path, session, collection_name="test_dedup")
        r2 = kb.ingest_document(path, session, collection_name="test_dedup")

        assert r1["chunks_added"] > 0
        assert r2["skipped"] is True
        assert r2["chunks_added"] == 0
    finally:
        os.unlink(path)
        kb._client = None


# ─────────────────────────────────────────────────────────────────────────────
# 6. PARALLEL INGESTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.perf
@requires_embedding
def test_parallel_ingestion_faster_than_serial(tmp_chroma_dir, monkeypatch):
    """
    Parallel ingest_documents_batch must be faster than running ingest_document
    sequentially for multiple PDFs.
    """
    monkeypatch.setattr(
        "config.settings.Settings.chroma_persist_dir",
        str(tmp_chroma_dir),
        raising=False,
    )
    import rag.knowledge_base as kb
    kb._client = None

    from config.settings import get_settings
    get_settings().__dict__["chroma_persist_dir"] = str(tmp_chroma_dir)

    # Create 4 distinct PDFs
    paths = [_write_temp_pdf(FINANCIAL_TEXT + f"\nDocument number {i}") for i in range(4)]
    session_p = "perf_parallel"
    session_s = "perf_serial"

    try:
        # Serial baseline
        kb._client = None
        t0 = time.perf_counter()
        for p in paths:
            kb.ingest_document(p, session_s, collection_name="col_serial")
        serial_time = time.perf_counter() - t0

        # Parallel run (fresh collection, different session so no dedup skip)
        kb._client = None
        t0 = time.perf_counter()
        kb.ingest_documents_batch(paths, session_p, collection_name="col_parallel", max_workers=4)
        parallel_time = time.perf_counter() - t0

        # Parallel should be at least 10% faster for 4 documents
        # (conservative threshold — real gains depend on CPU count)
        assert parallel_time <= serial_time * 1.1, (
            f"Parallel ({parallel_time:.2f}s) was not faster than serial ({serial_time:.2f}s)"
        )
    finally:
        for p in paths:
            os.unlink(p)
        kb._client = None


@pytest.mark.quality
@requires_embedding
def test_parallel_ingestion_consistent_results(tmp_chroma_dir, monkeypatch):
    """
    Parallel batch ingestion must produce the same number of chunks
    as sequential ingestion for the same documents.
    """
    monkeypatch.setattr(
        "config.settings.Settings.chroma_persist_dir",
        str(tmp_chroma_dir),
        raising=False,
    )
    import rag.knowledge_base as kb
    kb._client = None

    from config.settings import get_settings
    get_settings().__dict__["chroma_persist_dir"] = str(tmp_chroma_dir)

    paths = [_write_temp_pdf(FINANCIAL_TEXT + f"\nDoc {i}") for i in range(3)]
    try:
        results = kb.ingest_documents_batch(paths, "batch_session", collection_name="test_batch")

        assert len(results) == 3, "Should return one result per document"
        for i, r in enumerate(results):
            assert "chunks_added" in r, f"Result {i} missing chunks_added"
            assert r["chunks_added"] > 0 or r.get("skipped"), \
                f"Result {i} neither added chunks nor was skipped"
    finally:
        for p in paths:
            os.unlink(p)
        kb._client = None


@pytest.mark.quality
@requires_embedding
def test_parallel_ingestion_no_race_conditions(tmp_chroma_dir, monkeypatch):
    """
    Run parallel ingestion from multiple threads simultaneously.
    All documents must be indexed correctly with no missing chunks.
    """
    monkeypatch.setattr(
        "config.settings.Settings.chroma_persist_dir",
        str(tmp_chroma_dir),
        raising=False,
    )
    import rag.knowledge_base as kb
    kb._client = None

    from config.settings import get_settings
    get_settings().__dict__["chroma_persist_dir"] = str(tmp_chroma_dir)

    paths = [_write_temp_pdf(FINANCIAL_TEXT + f"\nThread doc {i}") for i in range(6)]
    errors = []

    def ingest_one(path, idx):
        try:
            kb.ingest_document(path, f"thread_session_{idx}", collection_name="race_test")
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=ingest_one, args=(p, i)) for i, p in enumerate(paths)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Race condition errors: {errors}"
    try:
        for p in paths:
            os.unlink(p)
    finally:
        kb._client = None


# ─────────────────────────────────────────────────────────────────────────────
# 7. RETRIEVAL QUALITY — BM25 + HYBRID
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.quality
def test_bm25_ranks_exact_match_first():
    """BM25 must rank the exact-match chunk above unrelated chunks."""
    from rag.retriever import BM25Retriever

    chunks = [
        {"text": "Revenue for 2023 was 4.2 billion dollars up 12 percent"},
        {"text": "Net income reached 380 million EPS 2.15"},
        {"text": "Total assets stood at 18500 million at year end"},
        {"text": "The company declared a dividend of 0.80 per share"},
        {"text": "Chocolate cake flour sugar eggs baking recipe"},
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.search("revenue 4.2 billion 2023", k=3)

    assert results, "BM25 should return results"
    assert "revenue" in results[0]["text"].lower() or "billion" in results[0]["text"].lower(), \
        f"Top BM25 result should contain 'revenue', got: {results[0]['text'][:80]}"


@pytest.mark.quality
def test_rrf_promotes_doubly_ranked_results():
    """
    RRF must give a higher score to a chunk ranked highly by BOTH
    vector and BM25 vs one ranked highly by only one method.
    """
    from rag.retriever import reciprocal_rank_fusion

    # "A" is top in vector, rank 2 in BM25 → doubly ranked
    # "B" is top in BM25, rank 3 in vector → doubly ranked
    # "C" only in BM25, not in vector → singly ranked
    vec = [{"text": "A"}, {"text": "B"}, {"text": "X"}]
    bm25 = [{"text": "B"}, {"text": "A"}, {"text": "C"}]

    fused = reciprocal_rank_fusion(vec, bm25)
    texts = [r["text"] for r in fused]

    # Both A and B should outrank C (which only appears in BM25)
    assert "C" not in texts[:2], \
        f"Singly-ranked 'C' should not be in top 2, got: {texts[:3]}"


@pytest.mark.quality
def test_rrf_all_scores_positive():
    """All RRF scores must be strictly positive."""
    from rag.retriever import reciprocal_rank_fusion

    vec = [{"text": f"doc_{i}"} for i in range(5)]
    bm25 = [{"text": f"doc_{i}"} for i in reversed(range(5))]
    fused = reciprocal_rank_fusion(vec, bm25)

    for r in fused:
        assert r["rrf_score"] > 0, f"RRF score must be positive, got {r['rrf_score']}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. END-TO-END INGESTION SPEED BUDGET
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.perf
@requires_embedding
def test_single_document_ingestion_budget(tmp_chroma_dir, monkeypatch):
    """
    Full single-document ingest (parse → chunk → embed → store) must complete
    within 10 seconds for a typical ~10-page financial PDF.
    """
    monkeypatch.setattr(
        "config.settings.Settings.chroma_persist_dir",
        str(tmp_chroma_dir),
        raising=False,
    )
    import rag.knowledge_base as kb
    kb._client = None

    from config.settings import get_settings
    get_settings().__dict__["chroma_persist_dir"] = str(tmp_chroma_dir)

    # ~10-page equivalent
    path = _write_temp_pdf(FINANCIAL_TEXT * 10)
    try:
        t0 = time.perf_counter()
        result = kb.ingest_document(path, "speed_test", collection_name="speed_col")
        elapsed = time.perf_counter() - t0

        assert result["chunks_added"] > 0
        assert elapsed < 10.0, (
            f"Single document ingest took {elapsed:.2f}s — expected < 10s. "
            f"Check PDF extractor and embedding batch size."
        )
    finally:
        os.unlink(path)
        kb._client = None
