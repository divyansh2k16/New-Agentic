"""
Basic integration tests for the pipeline.
Run: pytest tests/ -v
"""
import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Unit tests ────────────────────────────────────────────────────────────────

def test_chunk_text():
    from tools.pdf_loader import chunk_text
    text = "This is a test sentence. " * 50  # 1250 chars
    chunks = chunk_text(text, chunk_size=200, overlap=20, source="test.pdf", page=1)
    assert len(chunks) > 1
    assert all("text" in c for c in chunks)
    assert all("source" in c for c in chunks)
    assert all(len(c["text"]) <= 220 for c in chunks)  # Allow slight overshoot


def test_validate_file_nonexistent():
    from tools.guardrails import validate_file
    is_valid, reason = validate_file("/nonexistent/path/file.pdf")
    assert not is_valid
    assert "not found" in reason.lower()


def test_validate_query_empty():
    from tools.guardrails import validate_query
    is_valid, reason = validate_query("")
    assert not is_valid


def test_validate_query_too_long():
    from tools.guardrails import validate_query
    is_valid, reason = validate_query("x" * 3000)
    assert not is_valid
    assert "long" in reason.lower()


def test_validate_query_valid():
    from tools.guardrails import validate_query
    is_valid, reason = validate_query("What was the net income in 2023?")
    assert is_valid
    assert reason == ""


def test_pct_change():
    from agents.comparator import _pct_change, _trend
    assert _pct_change(100, 110) == 10.0
    assert _pct_change(100, 90) == -10.0
    assert _pct_change(0, 100) is None  # Division by zero safe
    assert _trend(15) == "strong_growth"
    assert _trend(5) == "growth"
    assert _trend(-5) == "decline"
    assert _trend(-15) == "strong_decline"


def test_compute_ratios():
    from agents.extractor import _compute_ratios
    data = {
        "net_income": 100,
        "revenue": 1000,
        "total_equity": 500,
        "total_assets": 2000,
        "total_debt": 300,
        "operating_cash_flow": 150,
        "free_cash_flow": None,
    }
    result = _compute_ratios(data)
    assert result["net_margin"] == pytest.approx(0.1, 0.01)
    assert result["roe"] == pytest.approx(0.2, 0.01)
    assert result["roa"] == pytest.approx(0.05, 0.01)
    assert result["debt_to_equity"] == pytest.approx(0.6, 0.01)


def test_settings_loads():
    from config.settings import get_settings
    settings = get_settings()
    assert settings.app_name == "Financial PDF Pipeline"
    assert settings.primary_llm_model == "claude-sonnet-4-6"


def test_bm25_retriever():
    from rag.retriever import BM25Retriever
    chunks = [
        {"text": "Apple revenue was 383 billion dollars in fiscal 2023"},
        {"text": "Net income for the quarter was 24 billion"},
        {"text": "The company paid dividends of 3.50 per share"},
        {"text": "Total assets stood at 352 billion as of year end"},
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.search("revenue 383 billion", k=2)
    assert len(results) >= 1
    # First result should be the revenue chunk
    assert "revenue" in results[0]["text"].lower() or "billion" in results[0]["text"].lower()


def test_rrf_fusion():
    from rag.retriever import reciprocal_rank_fusion
    vec_results = [
        {"text": "A", "score": 0.9},
        {"text": "B", "score": 0.8},
        {"text": "C", "score": 0.7},
    ]
    bm25_results = [
        {"text": "C", "bm25_score": 1.0},
        {"text": "A", "bm25_score": 0.8},
        {"text": "D", "bm25_score": 0.6},
    ]
    fused = reciprocal_rank_fusion(vec_results, bm25_results)
    assert len(fused) >= 3
    assert all("rrf_score" in r for r in fused)


def test_make_initial_state():
    from agents.orchestrator import make_initial_state
    state = make_initial_state(["file1.pdf", "file2.pdf"], task="classify")
    assert state["task"] == "classify"
    assert len(state["document_paths"]) == 2
    assert state["total_input_tokens"] == 0
    assert state["classifications"] == []
    assert isinstance(state["messages"], list)


# ── Marker: requires ANTHROPIC_API_KEY ───────────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
def test_guardrail_node_empty_paths():
    """Guardrail should reject empty file list."""
    from agents.orchestrator import make_initial_state
    from tools.guardrails import guardrail_check_node

    state = make_initial_state([], task="full_pipeline")
    result = guardrail_check_node(state)
    assert "guardrail_rejected" in result["completed_steps"]
