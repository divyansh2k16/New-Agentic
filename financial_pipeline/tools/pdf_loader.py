"""
PDF Loading & Parsing Tools

CONCEPT: The raw data ingestion layer. Quality here determines quality downstream.
Three strategies based on document complexity:
1. pdfplumber — best for text + table-heavy documents
2. PyPDF — fastest, good fallback
3. pymupdf (fitz) — best for scanned/image PDFs

In production (Citi): would also integrate with OCR (AWS Textract) for scanned docs.
"""
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import fitz  # pymupdf
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


def extract_text_from_pdf(file_path: str, max_chars: int = None) -> str:
    """
    Extract text from a PDF using the best available library.
    Falls back gracefully if libraries are missing.

    Args:
        file_path: Absolute path to the PDF
        max_chars: If set, truncate at this many characters (cost optimisation)

    Returns:
        Extracted text as a single string
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    text = ""

    # Strategy 1: pdfplumber (best quality)
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(file_path) as pdf:
                pages_text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)
                text = "\n\n".join(pages_text)
        except Exception as e:
            logger.warning(f"pdfplumber failed on {file_path}: {e}, trying fallback")

    # Strategy 2: PyPDF fallback
    if not text and HAS_PYPDF:
        try:
            reader = PdfReader(file_path)
            pages_text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
            text = "\n\n".join(pages_text)
        except Exception as e:
            logger.warning(f"pypdf failed on {file_path}: {e}")

    # Strategy 3: PyMuPDF fallback
    if not text and HAS_FITZ:
        try:
            doc = fitz.open(file_path)
            pages_text = [page.get_text() for page in doc]
            text = "\n\n".join(pages_text)
            doc.close()
        except Exception as e:
            logger.warning(f"pymupdf failed on {file_path}: {e}")

    if not text:
        logger.error(f"All PDF extractors failed for {file_path}")
        return ""

    if max_chars:
        text = text[:max_chars]

    logger.debug(f"Extracted {len(text)} chars from {path.name}")
    return text


def extract_text_by_page(file_path: str) -> List[Dict]:
    """
    Extract text page by page, returning metadata per page.
    Used for RAG chunking — preserves page number for citations.

    Returns:
        List of {"page": int, "text": str, "char_count": int}
    """
    pages = []

    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    pages.append({
                        "page": i + 1,
                        "text": text,
                        "char_count": len(text),
                    })
            return pages
        except Exception as e:
            logger.warning(f"Page extraction failed: {e}")

    if HAS_PYPDF:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text, "char_count": len(text)})

    return pages


def extract_tables_from_pdf(file_path: str) -> List[str]:
    """
    Extract tables from PDF and return as formatted strings.
    Tables contain the structured financial data (income statement rows, etc.)

    Returns:
        List of table strings (TSV-like format)
    """
    tables = []

    if not HAS_PDFPLUMBER:
        logger.warning("pdfplumber not available — table extraction skipped")
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
                        # Clean cells: replace None with empty string
                        cleaned = [str(cell).strip() if cell else "" for cell in row]
                        rows.append(" | ".join(cleaned))
                    if rows:
                        tables.append("\n".join(rows))
    except Exception as e:
        logger.error(f"Table extraction failed for {file_path}: {e}")

    logger.debug(f"Extracted {len(tables)} tables from {Path(file_path).name}")
    return tables


def get_pdf_metadata(file_path: str) -> Dict:
    """
    Extract PDF metadata (title, author, page count, file hash).
    Used for document deduplication and provenance tracking.
    """
    path = Path(file_path)
    stat = path.stat()

    # Compute content hash for deduplication
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    page_count = 0
    metadata = {}

    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                metadata = dict(pdf.metadata or {})
        except Exception:
            pass
    elif HAS_PYPDF:
        try:
            reader = PdfReader(file_path)
            page_count = len(reader.pages)
            if reader.metadata:
                metadata = {k: str(v) for k, v in reader.metadata.items()}
        except Exception:
            pass

    return {
        "filename": path.name,
        "file_path": str(path.absolute()),
        "page_count": page_count,
        "file_size_bytes": stat.st_size,
        "file_hash": file_hash,
        "pdf_title": metadata.get("/Title", ""),
        "pdf_author": metadata.get("/Author", ""),
        "pdf_created": metadata.get("/CreationDate", ""),
    }


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
    - 500 tokens (~2000 chars) is a good default for financial documents

    Args:
        text: Text to chunk
        chunk_size: Target chars per chunk
        overlap: Overlap chars between consecutive chunks
        source: Source filename for metadata
        page: Page number for citation

    Returns:
        List of {"text": str, "source": str, "page": int, "chunk_index": int}
    """
    if not text.strip():
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary
        if end < len(text):
            # Look for '. ' or '\n' near the end
            for boundary in [". ", "\n\n", "\n", " "]:
                pos = text.rfind(boundary, start + chunk_size // 2, end)
                if pos != -1:
                    end = pos + len(boundary)
                    break

        chunk_text_val = text[start:end].strip()
        if chunk_text_val:
            chunks.append({
                "text": chunk_text_val,
                "source": source,
                "page": page,
                "chunk_index": chunk_index,
                "char_count": len(chunk_text_val),
            })
            chunk_index += 1

        start = max(start + 1, end - overlap)

    return chunks
