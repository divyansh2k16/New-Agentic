"""
Comparator Agent

CONCEPT: Cross-document analysis. Compares financials across multiple years/periods.
This demonstrates the 'multi-document reasoning' capability — a key interview topic.

Key patterns:
- Structured analytical reasoning (chain-of-thought)
- YoY (Year-over-Year) calculation
- Trend detection (improving / declining / volatile)
- Risk flagging (sudden drops, unusual ratios)
"""
import json
from loguru import logger
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import FinancialPipelineState, ComparisonResult, ExtractedFinancials
from config.settings import get_settings

settings = get_settings()


def _pct_change(old: float, new: float) -> float:
    if old and old != 0:
        return round(((new - old) / abs(old)) * 100, 2)
    return None


def _trend(pct: float) -> str:
    if pct is None:
        return "unknown"
    if pct > 10:
        return "strong_growth"
    if pct > 0:
        return "growth"
    if pct > -10:
        return "decline"
    return "strong_decline"


def _compute_yoy(extractions: list, classifications: list) -> dict:
    """
    Pure Python YoY comparison across all documents sorted by fiscal year.
    No LLM call needed for math — saves tokens.
    """
    # Sort by fiscal year
    paired = sorted(
        zip(classifications, extractions),
        key=lambda x: x[0].get("fiscal_year", "0")
    )

    metrics = [
        "revenue", "net_income", "ebitda", "operating_income",
        "total_assets", "total_debt", "operating_cash_flow",
        "free_cash_flow", "eps_diluted", "roe", "roa", "net_margin"
    ]

    yoy_changes = {}
    for metric in metrics:
        values = []
        for clf, ext in paired:
            v = ext.get(metric)
            if v is not None:
                values.append({
                    "year": clf.get("fiscal_year", "?"),
                    "company": clf.get("company_name", "?"),
                    "value": v,
                })
        if len(values) >= 2:
            changes = []
            for i in range(1, len(values)):
                pct = _pct_change(values[i-1]["value"], values[i]["value"])
                changes.append({
                    "from": values[i-1]["year"],
                    "to": values[i]["year"],
                    "from_value": values[i-1]["value"],
                    "to_value": values[i]["value"],
                    "pct_change": pct,
                    "trend": _trend(pct),
                })
            yoy_changes[metric] = {"series": values, "changes": changes}

    return yoy_changes


COMPARISON_SYSTEM = """You are a senior financial analyst. Based on year-over-year financial data,
provide:
1. Key insights (max 5 bullet points) — most important trends
2. Risk flags — concerning patterns that need attention

Be specific and quantitative. Use the numbers provided.
"""


def comparator_node(state: FinancialPipelineState) -> FinancialPipelineState:
    """
    LangGraph node: Compares extracted financials across documents.
    First does pure math (no LLM), then asks LLM for narrative insights.
    """
    logger.info(f"[COMPARATOR] Starting comparison for {len(state['extractions'])} documents")

    if len(state["extractions"]) < 2:
        logger.warning("[COMPARATOR] Only 1 document — skipping comparison")
        comparison: ComparisonResult = {
            "documents_compared": [state["document_paths"][0]] if state["document_paths"] else [],
            "yoy_changes": {},
            "key_insights": ["Only one document provided — no comparison possible."],
            "risk_flags": [],
        }
        return {
            **state,
            "comparison": comparison,
            "completed_steps": state["completed_steps"] + ["comparison"],
            "next_agent": "summarizer",
        }

    # Step 1: Math (no LLM, free)
    yoy_changes = _compute_yoy(state["extractions"], state["classifications"])

    # Step 2: LLM for narrative insights (fast model sufficient)
    llm = ChatAnthropic(
        model=settings.fast_llm_model,
        api_key=settings.anthropic_api_key,
        max_tokens=600,
    )

    # Build a concise summary of the YoY data for the LLM
    yoy_summary = []
    for metric, data in yoy_changes.items():
        for change in data.get("changes", []):
            if change.get("pct_change") is not None:
                yoy_summary.append(
                    f"{metric}: {change['from']} -> {change['to']}: "
                    f"{change['from_value']} -> {change['to_value']} "
                    f"({change['pct_change']:+.1f}%, {change['trend']})"
                )

    companies = list({c.get("company_name", "") for c in state["classifications"]})
    years = [c.get("fiscal_year", "") for c in state["classifications"]]

    prompt = f"""Companies: {', '.join(companies)}
Fiscal years covered: {', '.join(sorted(set(years)))}

Year-over-year changes:
{chr(10).join(yoy_summary[:30])}

Provide key_insights (list of 5 strings) and risk_flags (list of strings).
Return JSON: {{"key_insights": [...], "risk_flags": [...]}}"""

    try:
        response = llm.invoke([
            SystemMessage(content=COMPARISON_SYSTEM),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        parsed = json.loads(raw)
        key_insights = parsed.get("key_insights", [])
        risk_flags = parsed.get("risk_flags", [])
    except Exception as e:
        logger.error(f"[COMPARATOR] LLM narrative failed: {e}")
        key_insights = ["Comparison computed but narrative generation failed."]
        risk_flags = []

    comparison: ComparisonResult = {
        "documents_compared": state["document_paths"],
        "yoy_changes": yoy_changes,
        "key_insights": key_insights,
        "risk_flags": risk_flags,
    }

    return {
        **state,
        "comparison": comparison,
        "completed_steps": state["completed_steps"] + ["comparison"],
        "next_agent": "summarizer",
    }
