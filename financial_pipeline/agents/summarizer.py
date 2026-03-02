"""
Summarizer Agent

CONCEPT: Generates a human-readable executive summary combining all prior agent outputs.
This is the synthesis step — a classic RAG + agent output aggregation pattern.

Key patterns:
- Prompt engineering for concise, structured output
- Combining structured data + comparison narrative
- Audience-aware summary (executive vs technical)
"""
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import FinancialPipelineState
from config.llm_factory import get_llm


SUMMARY_SYSTEM = """You are a financial research analyst writing executive summaries for senior management.

Your summaries must be:
- Concise but comprehensive (400-600 words)
- Structured with clear sections
- Data-driven with specific numbers
- Written in professional financial language
- Highlighting key risks and opportunities

Use this structure:
## Executive Summary
## Key Financial Highlights
## Year-over-Year Performance
## Risk Assessment
## Conclusion
"""


def summarizer_node(state: FinancialPipelineState) -> FinancialPipelineState:
    """
    LangGraph node: Generates executive summary from all extracted/compared data.
    """
    logger.info(f"[SUMMARIZER] Generating summary for session {state['session_id']}")

    llm = get_llm(model_tier="primary", max_tokens=1200)

    # Compile context from all prior agents
    doc_summaries = []
    for i, (clf, ext) in enumerate(zip(state["classifications"], state["extractions"])):
        currency = ext.get("currency", "USD")
        unit = ext.get("unit", "millions")
        doc_summaries.append(f"""
Document {i+1}: {clf.get('company_name', 'Unknown')} - {clf.get('doc_type', '?')} ({clf.get('fiscal_year', '?')} {clf.get('fiscal_period', '')})
  Revenue: {ext.get('revenue')} {currency} {unit}
  Net Income: {ext.get('net_income')} {currency} {unit}
  EBITDA: {ext.get('ebitda')} {currency} {unit}
  EPS (diluted): {ext.get('eps_diluted')}
  Total Assets: {ext.get('total_assets')} {currency} {unit}
  Total Debt: {ext.get('total_debt')} {currency} {unit}
  Operating Cash Flow: {ext.get('operating_cash_flow')} {currency} {unit}
  Net Margin: {ext.get('net_margin')}
  ROE: {ext.get('roe')}
  Dual-Use Flag: {clf.get('is_dual_use_material', False)}
""")

    comparison_text = ""
    if state.get("comparison"):
        comp = state["comparison"]
        insights = "\n".join(f"- {i}" for i in comp.get("key_insights", []))
        risks = "\n".join(f"- {r}" for r in comp.get("risk_flags", []))
        comparison_text = f"""
YEAR-OVER-YEAR INSIGHTS:
{insights}

RISK FLAGS:
{risks}
"""

    prompt = f"""Analyse these financial documents and write a comprehensive executive summary.

DOCUMENTS ANALYSED:
{''.join(doc_summaries)}

{comparison_text}

Write the executive summary following the required structure."""

    try:
        response = llm.invoke([
            SystemMessage(content=SUMMARY_SYSTEM),
            HumanMessage(content=prompt),
        ])
        summary = response.content
        input_tokens = response.usage_metadata.get("input_tokens", 0)
        output_tokens = response.usage_metadata.get("output_tokens", 0)
    except Exception as e:
        logger.error(f"[SUMMARIZER] Failed: {e}")
        summary = "Summary generation failed. Please check logs."
        input_tokens = 0
        output_tokens = 0

    return {
        **state,
        "summary": summary,
        "total_input_tokens": state["total_input_tokens"] + input_tokens,
        "total_output_tokens": state["total_output_tokens"] + output_tokens,
        "completed_steps": state["completed_steps"] + ["summarization"],
        "next_agent": "query_agent",
    }
