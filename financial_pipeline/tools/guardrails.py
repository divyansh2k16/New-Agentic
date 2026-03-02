"""
Guardrails — Input/Output Safety Layer

CONCEPT: Guardrails are non-negotiable in financial AI systems.
At Citi, this would include:
1. Input validation: file type, size, content safety
2. Output validation: no PII leakage, no hallucinated numbers
3. Compliance checks: GDPR, FINRA, MiFID II requirements
4. Rate limiting: prevent abuse
5. Audit trail: every check is logged

This is a FIRST-CLASS concern in any bank's AI deployment.
"""
import re
from pathlib import Path
from typing import Tuple, List
from loguru import logger

from agents.state import FinancialPipelineState
from config.settings import get_settings

settings = get_settings()


# ── Input guardrails ──────────────────────────────────────────────────────────

PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+1\s?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b"),
}

BANNED_CONTENT_PATTERNS = [
    re.compile(r"\b(password|passwd|secret|api.?key|token)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\b(hack|exploit|bypass|jailbreak)\b", re.IGNORECASE),
]

MAX_FILE_SIZE_BYTES = settings.max_file_size_mb * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf"}


def validate_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate a file before processing.

    Returns:
        (is_valid, reason) — reason is empty string if valid
    """
    path = Path(file_path)

    if not path.exists():
        return False, f"File not found: {file_path}"

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type: {path.suffix}. Only PDF allowed."

    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        return False, (
            f"File too large: {file_size / 1024 / 1024:.1f}MB. "
            f"Max: {settings.max_file_size_mb}MB"
        )

    if file_size == 0:
        return False, "File is empty"

    # Check PDF magic bytes
    with open(file_path, "rb") as f:
        header = f.read(4)
    if header != b"%PDF":
        return False, "File is not a valid PDF (bad magic bytes)"

    return True, ""


def detect_pii(text: str) -> List[dict]:
    """
    Scan text for PII patterns.

    Returns:
        List of {"type": str, "match": str} found PII instances
    """
    found = []
    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        for match in matches:
            found.append({"type": pii_type, "match": match[:20] + "..."})
    return found


def validate_query(query: str) -> Tuple[bool, str]:
    """
    Validate a natural language query before sending to LLM.
    Prevents prompt injection and banned content.
    """
    if not query or not query.strip():
        return False, "Empty query"

    if len(query) > 2000:
        return False, "Query too long (max 2000 chars)"

    for pattern in BANNED_CONTENT_PATTERNS:
        if pattern.search(query):
            return False, "Query contains disallowed content"

    return True, ""


def validate_llm_output(output: str) -> Tuple[bool, str, str]:
    """
    Post-process LLM output to catch hallucinations and PII.

    Returns:
        (is_safe, sanitised_output, warning_message)
    """
    if not output:
        return True, output, ""

    warnings = []

    # Check for PII in output
    pii_found = detect_pii(output)
    if pii_found:
        pii_types = list(set(p["type"] for p in pii_found))
        warnings.append(f"PII detected in output: {pii_types}. Review required.")
        # In production: redact PII automatically
        logger.warning(f"[GUARDRAIL] PII in LLM output: {pii_types}")

    # Check for suspicious fabricated numbers (e.g., impossible percentages)
    extreme_pct = re.findall(r"(\d{4,}%)", output)
    if extreme_pct:
        warnings.append(f"Suspiciously extreme percentages: {extreme_pct}")

    sanitised = output  # In production: apply redaction
    warning_msg = " | ".join(warnings)

    return len(warnings) == 0, sanitised, warning_msg


# ── LangGraph node ────────────────────────────────────────────────────────────

def guardrail_check_node(state: FinancialPipelineState) -> FinancialPipelineState:
    """
    LangGraph node: First node in the graph.
    Validates all input files before any LLM calls are made.
    Fails fast — no wasted API credits on bad inputs.
    """
    logger.info(f"[GUARDRAIL] Checking {len(state['document_paths'])} documents")
    errors = []
    valid_paths = []

    for file_path in state["document_paths"]:
        is_valid, reason = validate_file(file_path)
        if is_valid:
            valid_paths.append(file_path)
            logger.info(f"[GUARDRAIL] ✓ {Path(file_path).name}")
        else:
            logger.warning(f"[GUARDRAIL] ✗ {Path(file_path).name}: {reason}")
            errors.append(f"File validation failed [{file_path}]: {reason}")

    # Validate query if present
    query = state.get("query")
    if query:
        is_valid, reason = validate_query(query)
        if not is_valid:
            errors.append(f"Query validation failed: {reason}")
            logger.warning(f"[GUARDRAIL] Query rejected: {reason}")
            return {
                **state,
                "errors": errors,
                "completed_steps": state["completed_steps"] + ["guardrail_rejected"],
                "next_agent": "END",
            }

    if not valid_paths:
        logger.error("[GUARDRAIL] No valid documents — rejecting pipeline")
        return {
            **state,
            "document_paths": [],
            "errors": errors,
            "completed_steps": state["completed_steps"] + ["guardrail_rejected"],
            "next_agent": "END",
        }

    logger.info(f"[GUARDRAIL] Passed: {len(valid_paths)}/{len(state['document_paths'])} docs")

    return {
        **state,
        "document_paths": valid_paths,  # Only pass valid files downstream
        "errors": errors,               # Accumulate any warnings
        "completed_steps": state["completed_steps"] + ["guardrail_passed"],
        "next_agent": "classifier",
    }
