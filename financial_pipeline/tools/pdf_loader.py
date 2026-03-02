"""
PDF Loading & Parsing Tools

CONCEPT: The raw data ingestion layer. Quality here determines quality downstream.
Extraction priority (fastest → most accurate for text):
1. pymupdf (fitz)   — fastest (~3-10x over pdfplumber), excellent text quality
2. PyPDF            — fast, reliable fallback for straightforward PDFs
3. pdfplumber       — slowest, reserved for table extraction where it excels

Key perf change: pdfplumber was previously primary; pymupdf is now primary.
PDF file is opened ONCE via extract_document() which returns both metadata and
page text, eliminating the previous double-open pattern.

In production (Citi): would also integrate with OCR (AWS Textract) for scanned docs.
"""
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

try:
    import fitz  # pymupdf — fastest text extractor
    HAS_FITZ = True
except BaseException:
    HAS_FITZ = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except BaseException:
    # pypdf → pdfminer → cryptography may raise pyo3_runtime.PanicException
    # (not a subclass of Exception) on systems with broken Rust crypto bindings.
    HAS_PYPDF = False

try:
    import pdfplumber  # kept for table extraction only
    HAS_PDFPLUMBER = True
except BaseException:
    HAS_PDFPLUMBER = False


# ---------------------------------------------------------------------------
# Primary API: single-pass extraction (open the file once, get everything)
# ---------------------------------------------------------------------------

def extract_document(file_path: str) -> Dict:
    """
    Open the PDF ONCE and return both metadata and per-page text.

    Previously ingest_document() called get_pdf_metadata() then
    extract_text_by_page() separately — two full file opens with pdfplumber.
    This function eliminates that duplication.

    Returns dict with keys:
      filename, file_path, page_count, file_size_bytes, file_hash,
      pdf_title, pdf_author, pdf_created,
      pages: [{"page": int, "text": str, "char_count": int}, ...]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    stat = path.stat()
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    pages: List[Dict] = []
    meta: Dict = {}
    page_count = 0

    # Strategy 1: PyMuPDF (fitz) — 3-10x faster than pdfplumber for text
    if HAS_FITZ:
        try:
            doc = fitz.open(file_path)
            page_count = doc.page_count
            info = doc.metadata or {}
            meta = {
                "pdf_title": info.get("title", ""),
                "pdf_author": info.get("author", ""),
                "pdf_created": info.get("creationDate", ""),
            }
            for i, page in enumerate(doc):
                text = page.get_text("text")  # "text" mode is fastest
                pages.append({"page": i + 1, "text": text or "", "char_count": len(text or "")})
            doc.close()
        except Exception as e:
            logger.warning(f"[PDF] fitz failed on {file_path}: {e}, trying fallback")
            pages = []

    # Strategy 2: PyPDF fallback
    if not pages and HAS_PYPDF:
        try:
            reader = PdfReader(file_path)
            page_count = len(reader.pages)
            if reader.metadata:
                raw = reader.metadata
                meta = {
                    "pdf_title": str(raw.get("/Title", "")),
                    "pdf_author": str(raw.get("/Author", "")),
                    "pdf_created": str(raw.get("/CreationDate", "")),
                }
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append({"page": i + 1, "text": text, "char_count": len(text)})
        except Exception as e:
            logger.warning(f"[PDF] pypdf failed on {file_path}: {e}")
            pages = []

    # Strategy 3: pdfplumber last resort for text
    if not pages and HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                raw = pdf.metadata or {}
                meta = {
                    "pdf_title": raw.get("/Title", ""),
                    "pdf_author": raw.get("/Author", ""),
                    "pdf_created": raw.get("/CreationDate", ""),
                }
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    pages.append({"page": i + 1, "text": text, "char_count": len(text)})
        except Exception as e:
            logger.error(f"[PDF] All extractors failed for {file_path}: {e}")

    if not pages:
        logger.error(f"[PDF] No text extracted from {file_path}")

    logger.debug(f"[PDF] Extracted {len(pages)} pages from {path.name}")

    return {
        "filename": path.name,
        "file_path": str(path.absolute()),
        "page_count": page_count or len(pages),
        "file_size_bytes": stat.st_size,
        "file_hash": file_hash,
        "pdf_title": meta.get("pdf_title", ""),
        "pdf_author": meta.get("pdf_author", ""),
        "pdf_created": meta.get("pdf_created", ""),
        "pages": pages,
    }


# ---------------------------------------------------------------------------
# Legacy helpers — now thin wrappers over extract_document()
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str, max_chars: int = None) -> str:
    """
    Extract full text from a PDF as a single string.
    Uses extract_document() so the file is opened only once.
    """
    doc = extract_document(file_path)
    text = "\n\n".join(p["text"] for p in doc["pages"] if p["text"])
    if max_chars:
        text = text[:max_chars]
    logger.debug(f"[PDF] Extracted {len(text)} chars from {Path(file_path).name}")
    return text


def extract_text_by_page(file_path: str) -> List[Dict]:
    """
    Extract text page by page with page-number metadata.
    Uses extract_document() — no extra file open.
    """
    return extract_document(file_path)["pages"]


def get_pdf_metadata(file_path: str) -> Dict:
    """
    Get PDF file metadata (title, author, hash, page count).
    Uses extract_document() — no extra file open.
    """
    doc = extract_document(file_path)
    return {
        "filename": doc["filename"],
        "file_path": doc["file_path"],
        "page_count": doc["page_count"],
        "file_size_bytes": doc["file_size_bytes"],
        "file_hash": doc["file_hash"],
        "pdf_title": doc["pdf_title"],
        "pdf_author": doc["pdf_author"],
        "pdf_created": doc["pdf_created"],
    }


def extract_tables_from_pdf(file_path: str) -> List[str]:
    """
    Extract tables from PDF as formatted strings.
    pdfplumber is intentionally kept here — it genuinely excels at table
    structure detection compared to fitz or pypdf.
    """
    tables = []

    if not HAS_PDFPLUMBER:
        logger.warning("[PDF] pdfplumber not available — table extraction skipped")
        return tables

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for table in (page_tables or []):
                    if not table:
                        continue
                    rows = []
                    for row in table:
                        cleaned = [str(cell).strip() if cell else "" for cell in row]
                        rows.append(" | ".join(cleaned))
                    if rows:
                        tables.append("\n".join(rows))
    except Exception as e:
        logger.error(f"[PDF] Table extraction failed for {file_path}: {e}")

    logger.debug(f"[PDF] Extracted {len(tables)} tables from {Path(file_path).name}")
    return tables


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    source: str = "",
    page: int = 0,
) -> List[Dict]:
    """
    Split text into overlapping chunks for vector store ingestion.

    CONCEPT: Chunking strategy matters enormously for RAG quality:
    - Too small: loses context
    - Too large: wastes tokens, dilutes relevance
    - Overlap: prevents answers being split across chunk boundaries
    - 500 chars is a good default for financial documents

    Returns:
        List of {"text": str, "source": str, "page": int, "chunk_index": int}
    """
    if not text.strip():
        return []

    chunks = []
    start = 0
    chunk_index = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break at a sentence boundary near the target end
        if end < text_len:
            mid = start + chunk_size // 2
            for boundary in (". ", "\n\n", "\n", " "):
                pos = text.rfind(boundary, mid, end)
                if pos != -1:
                    end = pos + len(boundary)
                    break

        chunk_val = text[start:end].strip()
        if chunk_val:
            chunks.append({
                "text": chunk_val,
                "source": source,
                "page": page,
                "chunk_index": chunk_index,
                "char_count": len(chunk_val),
            })
            chunk_index += 1

        start = max(start + 1, end - overlap)

    return chunks
